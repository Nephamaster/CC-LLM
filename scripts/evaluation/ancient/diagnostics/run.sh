#!/usr/bin/env bash
set -euo pipefail

CC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$CC_ROOT"

# 与正式评估相同的环境、完整基座及 ACT adapter。
PYTHON="${PYTHON:-$CC_ROOT/scripts/train/ms_swift/env/bin/python}"
MODEL="${MODEL:-$CC_ROOT/outputs/experiments/rnd2/training/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500}"
LORA="${LORA:-$CC_ROOT/outputs/experiments/rnd2/finetune/Qwen3-1.7B-Base-PGCA-CPT-ACT/v0-20261025-232029/checkpoint-1000}"
DEV="${DEV:-$CC_ROOT/data/downstream/ancient/train/final/dev.metadata.jsonl}"
MODE="${MODE:-baseline}"
case "$MODE" in baseline|followup) ;; *) echo 'MODE must be baseline or followup' >&2; exit 2 ;; esac
OUT="${OUT:-$CC_ROOT/outputs/diagnostics/act_${MODE}_$(date +%Y%m%d-%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-2}"
if [[ "$MODE" == followup ]]; then
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
else
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
fi
CACHE_MAX_TOKENS="${CACHE_MAX_TOKENS:-128}"
# BF16 诊断阈值；超限需结合误差数值复核，不直接判定模型需要重训。
ATOL="${ATOL:-0.05}"
RTOL="${RTOL:-0.01}"

export CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
printf '[diagnose] GPU=%s output=%s\n' "$CUDA_VISIBLE_DEVICES" "$OUT"
if [[ "$MODE" == followup ]]; then
    CASES="${CASES:-$CC_ROOT/outputs/diagnostics/act_checks_20261026-115931/selected_dev.json}"
    "$PYTHON" -B -m scripts.evaluation.ancient.diagnostics.followup \
        --model "$MODEL" --adapter "$LORA" --dev "$DEV" --cases "$CASES" --output "$OUT" \
        --max-new-tokens "$MAX_NEW_TOKENS" --candidate "${CANDIDATE_TOKEN:-𫟦}"
    exit
fi
"$PYTHON" -B -m scripts.evaluation.ancient.diagnostics.checks \
    --model "$MODEL" --adapter "$LORA" --dev "$DEV" --output "$OUT" \
    --samples-per-task "$SAMPLES_PER_TASK" --max-new-tokens "$MAX_NEW_TOKENS" \
    --cache-max-tokens "$CACHE_MAX_TOKENS" --atol "$ATOL" --rtol "$RTOL"
