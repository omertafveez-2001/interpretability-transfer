"""
run_logit_lens.py
=================
Entry point for the logit lens experiment.
Runs logit lens on Pythia 1.4B, Mamba 1.4B (mamba_ssm), and RWKV 1.5B.
Uses wikitext-2 as evaluation dataset (proxy for the Pile).

Usage
-----
# Single model
python run_logit_lens.py --model pythia-1.4b --n_texts 50

# All three models
python run_logit_lens.py --all --n_texts 50

# Save results to json
python run_logit_lens.py --all --n_texts 50 --save_path results/logit_lens.json

# Use Pile instead of wikitext (requires HF auth)
python run_logit_lens.py --all --dataset pile
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset

from common.model_registry import LOGIT_LENS_MODELS
from common.logit_lens import LogitLens
import pandas as pd


def run_logit_lens(
    model_key: str,
    texts: List[str],
    layer_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Instantiate LogitLens, run over texts, print and return results.
    Frees GPU memory after each model.
    """
    print(f"\n[{model_key}] Loading model...")
    ll = LogitLens.from_model_key(model_key)

    names = layer_names or ll._get_layer_names()
    print(f"[{model_key}] {len(names)} layers — running over {len(texts)} texts...")

    results = ll.compute(texts, layer_names=names)
    ll.print_results(results, model_name=model_key, layer_names=names)

    # free GPU memory before loading next model
    del ll
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit lens experiment for Pythia 1.4B, Mamba 1.4B, and RWKV 1.5B."
    )
    parser.add_argument(
        "--model", default=None,
        choices=list(LOGIT_LENS_MODELS),
        help="Single model to run."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all three models sequentially."
    )
    parser.add_argument(
        "--n_texts", type=int, default=50,
        help="Number of evaluation texts (default: 50)."
    )
    # parser.add_argument(
    #     "--dataset", default="wikitext",
    #     choices=["wikitext", "pile"],
    #     help="Evaluation dataset (default: wikitext)."
    # )
    parser.add_argument(
        "--save_path", default=None,
        help="Path to save all results as a single JSON (optional)."
    )
    parser.add_argument(
        "--save_dir", default=None,
        help="Directory to save one JSON file per model: <save_dir>/<model_key>.json"
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent.parent / "datagen/logit_lens/data/wikitext-600.jsonl"),
        help="Path to JSONL file with evaluation texts."
    )
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model <key> or --all.")

    models = list(LOGIT_LENS_MODELS) if args.all else [args.model]

    print(f"\nLoading {args.n_texts} texts from '{args.dataset}'...")
    df = pd.read_json(args.dataset, lines=True)
    texts = df["text"].tolist()[: args.n_texts]
    print(f"Loaded {len(texts)} texts.")

    all_results: Dict[str, Dict[str, float]] = {}
    for model_key in models:
        results = run_logit_lens(model_key, texts)
        all_results[model_key] = results

        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"{model_key}.json"
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved → {out}")

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {save_path}")


if __name__ == "__main__":
    main()
