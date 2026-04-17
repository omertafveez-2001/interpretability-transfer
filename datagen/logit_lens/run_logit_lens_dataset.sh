#!/usr/bin/env bash
set -euo pipefail

# Download a set of texts for logit-lens evaluation.
# Edit the defaults below as needed.

SCRIPT="download_data.py"
N_TEXTS="2000"
DATASET="pile"     # wikitext | pile
OUT_PATH="./data/pile.jsonl"

cmd=(python "$SCRIPT" --n_texts "$N_TEXTS" --dataset "$DATASET" --out_path "$OUT_PATH")

echo "Running: ${cmd[*]}"
"${cmd[@]}"
