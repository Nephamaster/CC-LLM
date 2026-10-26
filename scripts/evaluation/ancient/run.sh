#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/../common.sh"
DATASET="${DATASET:-c3}"
configure_evaluation "$TASK_DIR" "$DATASET" "shot${FEWSHOT:-0}"
STAGE="${STAGE:-all}"
case "$STAGE" in all|prepare|infer|judge|score) ;; *) echo "Invalid STAGE" >&2; exit 2 ;; esac
printf '[evaluation][%s/shot%s] stage=%s output=%s\n' "$DATASET" "${FEWSHOT:-0}" "$STAGE" "$OUT"
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
  printf '[evaluation][%s/shot%s] preparing inputs\n' "$DATASET" "${FEWSHOT:-0}"
  "$PYTHON" -B -m scripts.evaluation.ancient.prepare --dataset "$DATASET" --fewshot "${FEWSHOT:-0}" --output "$OUT/input.jsonl"
  printf '[evaluation][%s/shot%s] inputs ready\n' "$DATASET" "${FEWSHOT:-0}"
fi
if [[ "$STAGE" == all || "$STAGE" == infer ]]; then
  : "${MODEL:?Set MODEL to the SFT model or base model for LORA}"
  adapter=()
  [[ -z "${LORA:-}" ]] || adapter=(--adapter "$LORA")
  printf '[evaluation][%s/shot%s] starting inference\n' "$DATASET" "${FEWSHOT:-0}"
  "$PYTHON" -B -m scripts.evaluation.inference --input "$OUT/input.jsonl" --output "$OUT/predictions.jsonl" \
    --model "$MODEL" "${adapter[@]}" --batch-size "${BATCH_SIZE:-8}" --max-new-tokens "${MAX_NEW_TOKENS:-2048}"
  printf '[evaluation][%s/shot%s] inference complete\n' "$DATASET" "${FEWSHOT:-0}"
fi
if [[ "$STAGE" == judge || ( "$STAGE" == all && "$DATASET" == fuxi && -n "${JUDGE_MODEL:-}" ) ]]; then
  [[ "$DATASET" == fuxi ]] || { echo 'Only Fuxi uses the judge' >&2; exit 2; }
  printf '[evaluation][fuxi/shot%s] running model judge\n' "${FEWSHOT:-0}"
  "$PYTHON" -B -m scripts.evaluation.ancient.judge --predictions "$OUT/predictions.jsonl" --output "$OUT/judgments.jsonl"
  printf '[evaluation][fuxi/shot%s] model judge complete\n' "${FEWSHOT:-0}"
fi
if [[ "$STAGE" == all || "$STAGE" == score ]]; then
  judged=()
  [[ ! -f "$OUT/judgments.jsonl" ]] || judged=(--judgments "$OUT/judgments.jsonl")
  printf '[evaluation][%s/shot%s] scoring predictions\n' "$DATASET" "${FEWSHOT:-0}"
  "$PYTHON" -B -m scripts.evaluation.ancient.score --input "$OUT/input.jsonl" --predictions "$OUT/predictions.jsonl" \
    --output "$OUT/metrics.json" "${judged[@]}"
  printf '[evaluation][%s/shot%s] metrics written to %s/metrics.json\n' "$DATASET" "${FEWSHOT:-0}" "$OUT"
fi
