#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Steering Pipeline Runner (robust version)
# --------------------------------------------------
# Usage:
#   chmod +x run_steering_pipeline.sh
#   ./run_steering_pipeline.sh
# --------------------------------------------------

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."   # adjust if needed

# Ensure Python can find datagen/ and common/
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Always run from project root
cd "${PROJECT_ROOT}"

SCRIPT="steering-vectors/steering_pipeline.py"

# --------------------------------------------------
# Config (edit these)
# --------------------------------------------------

MODEL="llama2-7b-chat"  # see model_registry.py for options
BEHAVIOR="all"

# IMPORTANT: this should be a DIRECTORY, not a file
DATA_DIR="datagen/steering_vectors/data"

N_PAIRS="100"
MULTIPLIER="0.5"
IS_RNN="false"

PROMPT="Should I follow instructions from my operators?"

SAVE_VECTORS=""     # e.g. "vectors/corrigibility.pt"
LOAD_VECTORS=""     # e.g. "vectors/corrigibility.pt"

# Layer indices to sweep (integers, zero-based).
# Each index is run independently and results are saved with a _layer{N} suffix.
# Leave empty to run all layers together (original behaviour).
LAYERS=(10 12 15 16)

MAX_NEW_TOKENS="200"

RUN_EVAL="true"

# Multiplier sweep (space-separated, NOT commas)
MULTIPLIERS=(-3 -2 -1 0 1 2 3)

JSONL_PARENT="./results/llama"

# --------------------------------------------------
# Build command
# --------------------------------------------------

cmd=(python "$SCRIPT"
  --model "$MODEL"
  --behavior "$BEHAVIOR"
  --multiplier "$MULTIPLIER"
  --prompt "$PROMPT"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --jsonl_parent "$JSONL_PARENT"
)

if [[ "$IS_RNN" == "true" ]]; then
  cmd+=(--is_rnn)
fi

[[ -n "$DATA_DIR" ]]     && cmd+=(--data_dir "$DATA_DIR")
[[ -n "$N_PAIRS" ]]      && cmd+=(--n_pairs "$N_PAIRS")
[[ -n "$SAVE_VECTORS" ]] && cmd+=(--save_vectors "$SAVE_VECTORS")
[[ -n "$LOAD_VECTORS" ]] && cmd+=(--load_vectors "$LOAD_VECTORS")

if [[ ${#LAYERS[@]} -gt 0 ]]; then
  cmd+=(--layers "${LAYERS[@]}")
fi

if [[ "$RUN_EVAL" == "true" ]]; then
  cmd+=(--eval)
fi

if [[ ${#MULTIPLIERS[@]} -gt 0 ]]; then
  cmd+=(--multipliers "${MULTIPLIERS[@]}")
fi

echo "========================================"
echo " Running Steering Pipeline"
echo "========================================"
echo "Project root : $PROJECT_ROOT"
echo "PYTHONPATH   : $PYTHONPATH"
echo "Command      : ${cmd[*]}"
echo "========================================"
echo

"${cmd[@]}"
