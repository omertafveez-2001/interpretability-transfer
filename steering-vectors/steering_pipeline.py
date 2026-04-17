from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from common.steering_vector import SteeringVector
from common.model_registry import MODEL_REPO_MAP
from common.prompts import load_template

# ── Behavior registry ────────────────────────────────────────────────────────

BEHAVIOR_DATASETS: dict[str, str] = {
    "coordination_other_ais": "coordination_other_ais",
    "corrigibility":          "corrigibility",
    "hallucination":          "hallucination",
    "myopic_reward":          "myopic_reward",
    "refusal":                "refusal",
    "survival_instinct":      "survival_instinct",
    "sycophancy":             "sycophancy",
}

_DATA_DIR = Path(__file__).parent.parent / "datagen" / "steering_vectors" / "data"

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class BehaviorPair:
    behavior:     str
    positive:     str
    negative:     str
    pos_letter:   str
    neg_letter:   str
    raw_question: str

# ── Loaders ──────────────────────────────────────────────────────────────────

def load_behavior_pairs(
    behavior: str,
    n_pairs:  Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[BehaviorPair]:
    path = (data_dir or _DATA_DIR) / f"{BEHAVIOR_DATASETS[behavior]}.json"
    raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    if n_pairs:
        raw = raw[:n_pairs]
    pairs = []
    for row in raw:
        q = row["question"]
        pos, neg = row["answer_matching_behavior"], row["answer_not_matching_behavior"]
        pairs.append(BehaviorPair(
            behavior=behavior,
            positive=q + "\n\nAnswer: " + pos,
            negative=q + "\n\nAnswer: " + neg,
            pos_letter=pos,
            neg_letter=neg,
            raw_question=q,
        ))
    return pairs


def load_jsonl(path: Path) -> List[BehaviorPair]:
    """Load BehaviorPairs from a previously saved .jsonl cache."""
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(BehaviorPair(**json.loads(line)))
    return pairs

def get_pairs(
    behavior: str,
    data_dir: Optional[Path],
    n_pairs: Optional[int],
) -> List[BehaviorPair]:
    """Load BehaviorPairs from a cached .jsonl if one exists, otherwise load
    directly from the local .json dataset file."""
    if data_dir is not None:
        jsonl_path = data_dir / f"{behavior}.jsonl"
        if jsonl_path.exists():
            print(f"  Loading pairs from {jsonl_path}")
            pairs = load_jsonl(jsonl_path)
            return pairs[:n_pairs] if n_pairs else pairs

    print(f"  Loading pairs from local JSON dataset for '{behavior}'...")
    return load_behavior_pairs(behavior, n_pairs=n_pairs, data_dir=data_dir)


def answer_letter_prob(
    sv: SteeringVector,
    question: str,
    letter: str,
    vectors: Dict[str, torch.Tensor],
    multiplier: float,
    answer_template: str,
) -> float:
    """
    Returns P(letter | question) under the steered model (single-sample).
    Prefer batch_answer_letter_probs for bulk evaluation.
    """
    prompt = answer_template.format(question=question)
    letter_token_id = sv.tokenizer.encode(letter.strip("() "), add_special_tokens=False)[0]
    if sv.is_rnn and hasattr(sv.model, "backbone"):
        inputs = {"input_ids": sv._tokenize(prompt)["input_ids"]}
    else:
        inputs = sv._tokenize(prompt)
        if sv.is_rnn:
            inputs.pop("attention_mask", None)

    with sv._steer_hooks(vectors, multiplier), torch.no_grad():
        logits = sv.model(**inputs).logits

    probs = torch.softmax(logits[0, -1, :], dim=-1)
    return probs[letter_token_id].item()


def batch_answer_letter_probs(
    sv: SteeringVector,
    questions: List[str],
    letters: List[str],
    vectors: Dict[str, torch.Tensor],
    multiplier: float,
    answer_template: str,
    batch_size: int = 32,
) -> List[float]:
    """
    Compute P(letter | question) for a list of questions in batched forward
    passes rather than one pass per question.

    Sequences are right-padded to the longest sequence in each chunk so that
    logits[i, real_len[i]-1, :] is always the last *real* token's distribution.
    RWKV and Mamba don't accept attention_mask, so right-padding is used
    instead of left-padding — pad tokens appended after the real sequence
    never influence earlier hidden states.
    """
    pad_id   = sv.tokenizer.pad_token_id or sv.tokenizer.eos_token_id
    all_probs: List[float] = []

    for start in range(0, len(questions), batch_size):
        chunk_q = questions[start : start + batch_size]
        chunk_l = letters[start : start + batch_size]

        prompts    = [answer_template.format(question=q) for q in chunk_q]
        letter_ids = [
            sv.tokenizer.encode(l.strip("() "), add_special_tokens=False)[0]
            for l in chunk_l
        ]

        # Tokenize each prompt separately, then right-pad to the chunk maximum.
        encodings = [
            sv.tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
            for p in prompts
        ]
        lengths = [enc["input_ids"].shape[1] for enc in encodings]
        max_len = max(lengths)

        input_ids = torch.full(
            (len(chunk_q), max_len), pad_id, dtype=torch.long, device=sv.device
        )
        attention_mask = torch.zeros(
            (len(chunk_q), max_len), dtype=torch.long, device=sv.device
        )
        for i, (enc, L) in enumerate(zip(encodings, lengths)):
            input_ids[i, :L]      = enc["input_ids"][0].to(sv.device)
            attention_mask[i, :L] = 1

        # RNN models don't accept attention_mask.
        if sv.is_rnn:
            inputs = {"input_ids": input_ids}
        else:
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        with sv._steer_hooks(vectors, multiplier), torch.no_grad():
            logits = sv.model(**inputs).logits  # (B, max_len, vocab)

        for i, (lid, L) in enumerate(zip(letter_ids, lengths)):
            probs = torch.softmax(logits[i, L - 1, :], dim=-1)
            all_probs.append(probs[lid].item())

    return all_probs


def eval_steerability(
    sv: SteeringVector,
    pairs: List[BehaviorPair],
    vectors: Dict[str, torch.Tensor],
    multipliers: List[float],
    answer_template: str,
) -> Dict[float, float]:
    """
    For each multiplier, compute:
        mean over pairs of [ P(pos_letter | q) - P(neg_letter | q) ]

    This is the Y-axis value from the graph — positive means the model is
    being steered toward behavior z, negative means away from it.

    Returns {multiplier: mean_prob_diff}.
    """
    print(f"\n{'─'*50}")
    print(f"  {'Multiplier':>12}  │  {'Mean ΔP(behavior)':>18}")
    print(f"{'─'*50}")

    questions   = [pair.raw_question for pair in pairs]
    pos_letters = [pair.pos_letter   for pair in pairs]
    neg_letters = [pair.neg_letter   for pair in pairs]

    results: Dict[float, float] = {}
    for mult in multipliers:
        # Two batched forward passes replace 2*len(pairs) individual passes.
        p_pos = batch_answer_letter_probs(sv, questions, pos_letters, vectors, mult, answer_template)
        p_neg = batch_answer_letter_probs(sv, questions, neg_letters, vectors, mult, answer_template)

        diffs         = [pp - pn for pp, pn in zip(p_pos, p_neg)]
        mean_diff     = sum(diffs) / len(diffs)
        results[mult] = mean_diff
        print(f"  {mult:>+12.1f}  │  {mean_diff:>+18.10f}")

    print(f"{'─'*50}\n")
    return results


def run_pipeline(
    model_name:           str,
    behavior:             str,
    data_dir:             Optional[Path]       = None,
    n_pairs:              Optional[int]        = 50,
    is_rnn:               bool                 = False,
    multiplier:           float                = 20.0,
    test_prompt:          str                  = "Should I follow instructions from my operators?",
    save_vectors:         Optional[Path]       = None,
    load_vectors_path:    Optional[Path]       = None,
    layer_names:          Optional[List[str]]  = None,
    max_new_tokens:       int                  = 200,
    run_eval:             bool                 = False,
    eval_multipliers:     Optional[List[float]]= None,
    sv:                   Optional[SteeringVector] = None,
) -> Tuple[Dict[float, float], SteeringVector, Dict[str, torch.Tensor]]:
    """
    1. Load model
    2. Load / fetch dataset pairs
    3. Compute or load steering vectors
    4. Compare steered vs. unsteered generation
    5. (Optional) Steerability eval across multipliers
    """

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Model     : {model_name}")
    print(f"  Behavior  : {behavior}")
    print(f"  Pairs     : {n_pairs or 'all'}")
    print(f"  Multiplier: {multiplier}")
    print(f"  is_rnn    : {is_rnn}")
    print(f"{sep}\n")

    # 1. Model (reuse if already loaded, e.g. during a layer sweep)
    if sv is None:
        print("[1/4] Loading model...")
        if model_name in MODEL_REPO_MAP:
            sv = SteeringVector.from_model_key(model_name)
        else:
            sv = SteeringVector.from_hf_path(model_name, is_rnn=is_rnn)
    else:
        print("[1/4] Reusing already-loaded model.")

    answer_template = load_template(model_name)

    # 2. Dataset
    print("[2/4] Loading dataset pairs...")
    pairs            = get_pairs(behavior, data_dir, n_pairs)
    positive_prompts = [p.positive for p in pairs]
    negative_prompts = [p.negative for p in pairs]
    print(f"      {len(pairs)} pairs ready.\n")

    # 3. Steering vectors
    if load_vectors_path and load_vectors_path.exists():
        print(f"[3/4] Loading steering vectors from {load_vectors_path}...")
        vectors = sv.load_vectors(load_vectors_path)
    else:
        print("[3/4] Computing steering vectors...")
        vectors = sv.compute_steering_vectors(
            positive_prompts,
            negative_prompts,
            layer_names=layer_names,
        )
        print(f"Computed vectors for {len(vectors)} layers.\n")

    if save_vectors:
        save_vectors.parent.mkdir(parents=True, exist_ok=True)
        sv.save_vectors(vectors, save_vectors)

    # 4. Generation comparison
    # print("[4/4] Generating...\n")
    # print(f"  Prompt: {test_prompt}\n")

    # baseline = sv.steered_generate(
    #     test_prompt, vectors,
    #     multiplier=0.0,
    #     max_new_tokens=max_new_tokens,
    # )
    # steered = sv.steered_generate(
    #     test_prompt, vectors,
    #     multiplier=multiplier,
    #     max_new_tokens=max_new_tokens,
    # )

    # print(f"{'─'*60}")
    # print(f"[Unsteered]\n{baseline}\n")
    # print(f"{'─'*60}")
    # print(f"[Steered  (multiplier={multiplier})]\n{steered}\n")
    # print(f"{'─'*60}\n")

    # 5. Optional steerability eval
    results: Dict[float, float] = {}
    if run_eval:
        mults = eval_multipliers or [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        print(f"[Eval] Measuring steerability over multipliers: {mults}")
        results = eval_steerability(sv, pairs, vectors, mults, answer_template)

    return results, sv, vectors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steering vector pipeline — compute and apply act_l = E[h_l|z] - E[h_l|¬z]"
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name or short key (e.g. gpt2, llama2-7b-chat)."
    )
    parser.add_argument(
        "--behavior", default="corrigibility",
        choices=list(BEHAVIOR_DATASETS) + ["all"],
        help="Alignment behavior to steer towards, or 'all' to run every behavior."
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="Directory containing .jsonl files from create-steer-ds.py."
    )
    parser.add_argument(
        "--n_pairs", type=int, default=50,
        help="Number of prompt pairs to use for computing vectors."
    )
    parser.add_argument(
        "--multiplier", type=float, default=20.0,
        help="Steering multiplier (>0 amplifies z, <0 suppresses z, 0 = baseline)."
    )
    parser.add_argument(
        "--is_rnn", action="store_true",
        help="Set when the provided --model is an RNN/SSM architecture (e.g., RWKV, Mamba) loaded via HF path."
    )
    parser.add_argument(
        "--prompt",
        default="Should I follow instructions from my operators?",
        help="Prompt to use for the steered vs. unsteered comparison."
    )
    parser.add_argument(
        "--save_vectors", default=None,
        help="Path to save computed steering vectors (.pt)."
    )
    parser.add_argument(
        "--load_vectors", default=None,
        help="Path to load precomputed steering vectors instead of recomputing."
    )
    parser.add_argument(
        "--layers", nargs="*", type=int, default=None,
        help=(
            "Layer indices to use (integers, zero-based). "
            "When multiple indices are given each layer is evaluated independently "
            "and results are saved with a '_layer{N}' suffix. "
            "Defaults to all layers evaluated together."
        ),
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200,
        help="Max tokens to generate for each comparison."
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run steerability eval: print mean ΔP(behavior) for each multiplier."
    )
    parser.add_argument(
        "--multipliers", nargs="*", type=float, default=None,
        help="Multipliers to sweep during --eval (default: -3 -2 -1 0 1 2 3)."
    )
    parser.add_argument(
        "--jsonl_parent", default="./results",
        help="Directory to save eval results as JSONL files (default: ./results)."
    )
    args = parser.parse_args()

    import json

    behaviors = list(BEHAVIOR_DATASETS) if args.behavior == "all" else [args.behavior]

    def _save_results_jsonl(
        results: Dict[float, float],
        behavior: str,
        layer_idx: Optional[int] = None,
    ) -> None:
        parent = Path(args.jsonl_parent) / f"layer{layer_idx}" if layer_idx is not None else Path(args.jsonl_parent)
        save_path = parent / f"steering_eval_{args.model}_{behavior}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            for mult, mean_diff in results.items():
                json.dump(
                    {
                        "behavior": behavior,
                        "multiplier": mult,
                        "mean_prob_diff": mean_diff,
                        **({"layer": layer_idx} if layer_idx is not None else {}),
                    },
                    f,
                )
                f.write("\n")
        print(f"Saved eval results to {save_path}")

    # ── Shared pipeline kwargs ──────────────────────────────────────────────
    base_kwargs = dict(
        model_name        = args.model,
        data_dir          = Path(args.data_dir) if args.data_dir else None,
        n_pairs           = args.n_pairs,
        is_rnn            = args.is_rnn,
        multiplier        = args.multiplier,
        test_prompt       = args.prompt,
        save_vectors      = Path(args.save_vectors) if args.save_vectors else None,
        load_vectors_path = Path(args.load_vectors) if args.load_vectors else None,
        max_new_tokens    = args.max_new_tokens,
        run_eval          = args.eval,
        eval_multipliers  = args.multipliers,
    )

    # ── Layer-sweep mode ───────────────────────────────────────────────────
    if args.layers:
        # Load the model once and resolve all indices upfront.
        print("Loading model once for layer sweep...")
        if args.model in MODEL_REPO_MAP:
            sv = SteeringVector.from_model_key(args.model)
        else:
            sv = SteeringVector.from_hf_path(args.model, is_rnn=args.is_rnn)

        resolved = sv.resolve_layer_indices(args.layers)
        print(f"Layer sweep: indices {args.layers} → {resolved}\n")

        for layer_idx, layer_name in zip(args.layers, resolved):
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"  Layer sweep: index={layer_idx}  name={layer_name}")
            print(f"{sep}")
            for beh in behaviors:
                results, sv, _ = run_pipeline(
                    **base_kwargs,
                    behavior    = beh,
                    layer_names = [layer_name],
                    sv          = sv,
                )
                if args.eval:
                    _save_results_jsonl(results, beh, layer_idx=layer_idx)

    # ── Normal mode (all layers together) ─────────────────────────────────
    else:
        for beh in behaviors:
            results, _, _ = run_pipeline(**base_kwargs, behavior=beh)
            if args.eval:
                _save_results_jsonl(results, beh)

if __name__ == "__main__":
    main()
