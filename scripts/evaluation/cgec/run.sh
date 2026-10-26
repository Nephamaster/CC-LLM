#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/../common.sh"
DATASET="${DATASET:-nacgec}"
SPLIT="${SPLIT:-test}"
configure_evaluation "$TASK_DIR" "$DATASET" "$SPLIT"
STAGE="${STAGE:-all}"
case "$STAGE" in all|prepare|infer|score) ;; *) echo "Invalid STAGE" >&2; exit 2 ;; esac
printf '[evaluation][%s/%s] stage=%s output=%s\n' "$DATASET" "$SPLIT" "$STAGE" "$OUT"
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
  printf '[evaluation][%s/%s] preparing inputs\n' "$DATASET" "$SPLIT"
  "$PYTHON" -B -m scripts.evaluation.cgec.prepare --dataset "$DATASET" --split "$SPLIT" --output "$OUT/input.jsonl"
  printf '[evaluation][%s/%s] inputs ready\n' "$DATASET" "$SPLIT"
fi
if [[ "$STAGE" == all || "$STAGE" == infer ]]; then
  : "${MODEL:?Set MODEL to the SFT model or base model for LORA}"
  adapter=()
  [[ -z "${LORA:-}" ]] || adapter=(--adapter "$LORA")
  printf '[evaluation][%s/%s] starting inference\n' "$DATASET" "$SPLIT"
  "$PYTHON" -B -m scripts.evaluation.inference --input "$OUT/input.jsonl" --output "$OUT/predictions.jsonl" \
    --model "$MODEL" "${adapter[@]}" --batch-size "${BATCH_SIZE:-8}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
    --backend "${INFERENCE_BACKEND:-auto}" --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  printf '[evaluation][%s/%s] inference complete\n' "$DATASET" "$SPLIT"
fi
if [[ "$STAGE" == all || "$STAGE" == score ]]; then
  printf '[evaluation][%s/%s] scoring predictions\n' "$DATASET" "$SPLIT"
  "$PYTHON" -B -m scripts.evaluation.cgec.score --input "$OUT/input.jsonl" --predictions "$OUT/predictions.jsonl" --output "$OUT/metrics.json"
  printf '[evaluation][%s/%s] metrics written to %s/metrics.json\n' "$DATASET" "$SPLIT" "$OUT"
fi
