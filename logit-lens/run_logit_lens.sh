#!/usr/bin/env bash
set -euo pipefail

# Runner for tuned-lens experiment (mirrors run_steering_pipeline.sh style).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Ensure Python can import project modules
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

SCRIPT="logit-lens/logit_lens_pipeline.py"

# -------- Config --------
MODEL="pythia-1.4b"               # set RUN_ALL=true to ignore this
RUN_ALL="false"                   # true | false
N_TEXTS="2000"
# Path to JSONL with texts; leave empty to use run_logit_lens.py default
DATASET_JSONL="${PROJECT_ROOT}/datagen/logit_lens/data/pile.jsonl"
SAVE_PATH=""
SAVE_DIR="${PROJECT_ROOT}/results/logit_lens"
# ------------------------

cmd=(python "$SCRIPT" --n_texts "$N_TEXTS")

if [[ -n "$DATASET_JSONL" ]]; then
  cmd+=(--dataset "$DATASET_JSONL")
fi

if [[ "$RUN_ALL" == "true" ]]; then
  cmd+=(--all)
else
  cmd+=(--model "$MODEL")
fi

[[ -n "$SAVE_PATH" ]] && cmd+=(--save_path "$SAVE_PATH")
[[ -n "$SAVE_DIR" ]] && cmd+=(--save_dir "$SAVE_DIR")

echo "========================================"
echo " Running tuned lens"
echo "========================================"
echo "Project root : $PROJECT_ROOT"
echo "PYTHONPATH   : $PYTHONPATH"
echo "Command      : ${cmd[*]}"
echo "========================================"
echo

"${cmd[@]}"
