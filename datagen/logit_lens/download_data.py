from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset


def load_texts(n_texts: int, dataset: str = "wikitext", offset: int = 0) -> List[str]:
    """
    Load evaluation texts, optionally skipping the first `offset` valid texts.
    Uses wikitext-2 as a proxy for the Pile since the Pile requires auth.
    Filters out short or empty lines.
    """
    if dataset == "wikitext":
        ds    = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [row["text"] for row in ds if len(row["text"].strip()) > 100]
        texts = texts[offset:]
    elif dataset == "pile":
        ds      = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        texts   = []
        skipped = 0
        for row in ds:
            if len(row["text"].strip()) <= 100:
                continue
            if skipped < offset:
                skipped += 1
                continue
            texts.append(row["text"])
            if len(texts) >= n_texts:
                break
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'wikitext' or 'pile'.")

    return texts[:n_texts]


def save_jsonl(texts: List[str], out_path: Path) -> None:
    """
    Save a list of texts to a JSONL file under the given path.
    Each line is a dict with a single \"text\" field.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"Wrote {len(texts)} texts → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and save logit lens evaluation texts")
    parser.add_argument("--n_texts", type=int, default=500, help="Number of texts to keep")
    parser.add_argument("--offset", type=int, default=0, help="Number of valid texts to skip before collecting")
    parser.add_argument(
        "--dataset",
        choices=["wikitext", "pile"],
        default="wikitext",
        help="Which dataset split to sample from",
    )
    default_out = Path(__file__).parent / "logit_lens" / "texts.jsonl"
    parser.add_argument(
        "--out_path",
        type=Path,
        default=default_out,
        help="Destination JSONL file (default: datagen/logit-lens/logit_lens/texts.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = load_texts(args.n_texts, dataset=args.dataset, offset=args.offset)
    save_jsonl(texts, args.out_path)


if __name__ == "__main__":
    main()
