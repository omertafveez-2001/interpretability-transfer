"""Convert tuned-lens aggregate_metrics.json to logit-lens BPB format.

Output format (same as logit-lens):
  {"layer_0": 1.52, "layer_1": 1.43, ..., "final": 1.21}
"""
import argparse
import json
from pathlib import Path


def convert(eval_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in sorted(eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        metrics_path = model_dir / "aggregate_metrics.json"
        if not metrics_path.exists():
            print(f"Skipping {model_dir.name} — no aggregate_metrics.json")
            continue

        metrics = json.loads(metrics_path.read_text())

        # tuned lens CE per layer
        layer_bpb: dict[str, float] = metrics["tuned"]["ce"]
        # final layer BPB from the baseline (model's own logits)
        final_bpb: float = metrics["baseline"]["ce"]["final"]

        result = dict(layer_bpb)
        result["final"] = final_bpb

        out_path = out_dir / f"{model_dir.name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--out_dir",  type=Path, required=True)
    args = parser.parse_args()
    convert(args.eval_dir, args.out_dir)


if __name__ == "__main__":
    main()
