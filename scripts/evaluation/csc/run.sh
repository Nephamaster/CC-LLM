#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/../common.sh"
configure_evaluation "$TASK_DIR" csc
STAGE="${STAGE:-all}"
case "$STAGE" in all|prepare|infer|score) ;; *) echo "Invalid STAGE" >&2; exit 2 ;; esac
printf '[evaluation][csc] stage=%s output=%s\n' "$STAGE" "$OUT"
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
  printf '[evaluation][csc] preparing inputs\n'
  "$PYTHON" -B -m scripts.evaluation.csc.prepare --output "$OUT/input.jsonl"
  printf '[evaluation][csc] inputs ready\n'
fi
if [[ "$STAGE" == all || "$STAGE" == infer ]]; then
  : "${MODEL:?Set MODEL to the SFT model or base model for LORA}"
  adapter=()
  [[ -z "${LORA:-}" ]] || adapter=(--adapter "$LORA")
  printf '[evaluation][csc] starting inference\n'
  "$PYTHON" -B -m scripts.evaluation.inference --input "$OUT/input.jsonl" --output "$OUT/predictions.jsonl" \
    --model "$MODEL" "${adapter[@]}" --batch-size "${BATCH_SIZE:-8}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
    --backend "${INFERENCE_BACKEND:-auto}" --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  printf '[evaluation][csc] inference complete\n'
fi
if [[ "$STAGE" == all || "$STAGE" == score ]]; then
  printf '[evaluation][csc] scoring predictions\n'
  "$PYTHON" -B -m scripts.evaluation.csc.score --input "$OUT/input.jsonl" --predictions "$OUT/predictions.jsonl" --output "$OUT/metrics.json"
  printf '[evaluation][csc] metrics written to %s/metrics.json\n' "$OUT"
fi
