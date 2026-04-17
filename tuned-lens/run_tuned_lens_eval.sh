#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"

# -------- Config --------
DATA="${PROJECT_ROOT}/datagen/logit_lens/data/pile.jsonl"
LENS_DIR="${PROJECT_ROOT}/results/tuned_lens"
EVAL_DIR="${PROJECT_ROOT}/results/tuned_lens_eval"
OUT_DIR="${PROJECT_ROOT}/results/tuned_lens_bpb"
# ------------------------

MODELS=(
    "mamba-790m:state-spaces/mamba-790m"
    "mamba-1.4b:state-spaces/mamba-1.4b"
    "mamba-2.8b:state-spaces/mamba-2.8b"
    "rwkv-v4-3b:RWKV/rwkv-4-pile-3b"
    "btlm-3b:cerebras/btlm-3b-8k-base"
)

echo "========================================"
echo " Evaluating Tuned Lens (BPB per layer)"
echo "========================================"

for ENTRY in "${MODELS[@]}"; do
    NAME="${ENTRY%%:*}"
    REPO="${ENTRY##*:}"
    LENS="${LENS_DIR}/${NAME}"
    EVAL_OUT="${EVAL_DIR}/${NAME}"

    if [ ! -d "$LENS" ]; then
        echo "Skipping $NAME — no trained lens at $LENS"
        continue
    fi

    # RWKV-v4 has a hard 1024-token context limit
    if [[ "$NAME" == rwkv* ]]; then
        MAX_SEQ_LEN=1024
    else
        MAX_SEQ_LEN=2048
    fi

    echo "----------------------------------------"
    echo " Model      : $NAME ($REPO)"
    echo " Lens       : $LENS"
    echo " Output     : $EVAL_OUT"
    echo " Max seq len: $MAX_SEQ_LEN"
    echo "----------------------------------------"

    python -m tuned_lens eval \
        --model.name "$REPO" \
        --lens_name "$LENS" \
        --output "$EVAL_OUT" \
        --data.name "$DATA" \
        --max_seq_len "$MAX_SEQ_LEN"

    echo "Done eval: $NAME"
    echo
done

echo "========================================"
echo " Converting to logit-lens BPB format"
echo "========================================"

python "${PROJECT_ROOT}/tuned-lens/convert_tuned_lens_bpb.py" \
    --eval_dir "$EVAL_DIR" \
    --out_dir  "$OUT_DIR"

echo "BPB results saved to $OUT_DIR"
