#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/../common.sh"
DATASET="${DATASET:-c3}"
configure_evaluation "$TASK_DIR" "$DATASET" "shot${FEWSHOT:-0}"
STAGE="${STAGE:-all}"
case "$STAGE" in all|prepare|infer|judge|score) ;; *) echo "Invalid STAGE" >&2; exit 2 ;; esac
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
  "$PYTHON" -B -m scripts.evaluation.ancient.prepare --dataset "$DATASET" --fewshot "${FEWSHOT:-0}" --output "$OUT/input.jsonl"
fi
if [[ "$STAGE" == all || "$STAGE" == infer ]]; then
  : "${MODEL:?Set MODEL to the SFT model or base model for LORA}"
  adapter=()
  [[ -z "${LORA:-}" ]] || adapter=(--adapter "$LORA")
  "$PYTHON" -B -m scripts.evaluation.inference --input "$OUT/input.jsonl" --output "$OUT/predictions.jsonl" \
    --model "$MODEL" "${adapter[@]}" --batch-size "${BATCH_SIZE:-8}" --max-new-tokens "${MAX_NEW_TOKENS:-2048}"
fi
if [[ "$STAGE" == judge || ( "$STAGE" == all && "$DATASET" == fuxi && -n "${JUDGE_MODEL:-}" ) ]]; then
  [[ "$DATASET" == fuxi ]] || { echo 'Only Fuxi uses the judge' >&2; exit 2; }
  "$PYTHON" -B -m scripts.evaluation.ancient.judge --predictions "$OUT/predictions.jsonl" --output "$OUT/judgments.jsonl"
fi
if [[ "$STAGE" == all || "$STAGE" == score ]]; then
  judged=()
  [[ ! -f "$OUT/judgments.jsonl" ]] || judged=(--judgments "$OUT/judgments.jsonl")
  "$PYTHON" -B -m scripts.evaluation.ancient.score --input "$OUT/input.jsonl" --predictions "$OUT/predictions.jsonl" \
    --output "$OUT/metrics.json" "${judged[@]}"
fi
