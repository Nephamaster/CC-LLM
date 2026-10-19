#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/../common.sh"
configure_evaluation "$TASK_DIR" ppl
: "${MODEL:?Set MODEL to a full HF model checkpoint, not a tokenizer-only or LoRA directory}"
"$PYTHON" -B -m scripts.evaluation.ppl.evaluate \
  --model "$MODEL" --input-dir "${DATA_DIR:-$ROOT/data/ppl}" --output "$OUT/metrics.json" \
  --window "${WINDOW:-2048}" --stride "${STRIDE:-1024}" \
  --dtype "${DTYPE:-auto}" --device "${DEVICE:-auto}" --attention "${ATTENTION:-sdpa}" \
  --loss-chunk-size "${LOSS_CHUNK_SIZE:-128}"
