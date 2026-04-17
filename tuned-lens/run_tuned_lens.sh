#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"

# -------- Config --------
DATA="${PROJECT_ROOT}/datagen/logit_lens/data/pile.jsonl"
OUT_DIR="${PROJECT_ROOT}/results/tuned_lens"
NUM_STEPS=100
TOKENS_PER_STEP=32768   # 2^15 — 8x fewer grad-acc steps than default 2^18
LOSS="kl"
# ------------------------

MODELS=(
    "mamba-790m:state-spaces/mamba-790m"
    "mamba-1.4b:state-spaces/mamba-1.4b"
    "mamba-2.8b:state-spaces/mamba-2.8b"
    "rwkv-v4-3b:RWKV/rwkv-4-pile-3b"
    "btlm-3b:cerebras/btlm-3b-8k-base"
)

echo "========================================"
echo " Training Tuned Lens"
echo "========================================"
echo "Data     : $DATA"
echo "Output   : $OUT_DIR"
echo "Steps    : $NUM_STEPS"
echo "Loss     : $LOSS"
echo "========================================"
echo

for ENTRY in "${MODELS[@]}"; do
    NAME="${ENTRY%%:*}"
    REPO="${ENTRY##*:}"
    OUT="${OUT_DIR}/${NAME}"

    # RWKV-v4 has a hard 1024-token context limit
    if [[ "$NAME" == rwkv* ]]; then
        MAX_SEQ_LEN=1024
    else
        MAX_SEQ_LEN=2048
    fi

    echo "----------------------------------------"
    echo " Model      : $NAME ($REPO)"
    echo " Output     : $OUT"
    echo " Max seq len: $MAX_SEQ_LEN"
    echo "----------------------------------------"

    python -m tuned_lens train \
        --model.name "$REPO" \
        --output "$OUT" \
        --data.name "$DATA" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --num_steps "$NUM_STEPS" \
        --tokens_per_step "$TOKENS_PER_STEP" \
        --loss "$LOSS"

    echo "Done: $NAME"
    echo
done

echo "========================================"
echo " All tuned lenses saved to $OUT_DIR"
echo "========================================"
