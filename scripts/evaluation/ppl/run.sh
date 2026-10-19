#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
if [[ -z "${OUT:-}" ]]; then
  : "${MODEL_NAME:?Set MODEL_NAME to the evaluated model name, or supply OUT explicitly}"
  for component in "$MODEL_NAME" "${RUN_NAME:-}"; do
    [[ "$component" != */* && "$component" != . && "$component" != .. ]] || {
      echo "Evaluation names must be single directory components" >&2; exit 2;
    }
  done
  OUT="$ROOT/outputs/evaluation/ppl/$MODEL_NAME"
  [[ -z "${RUN_NAME:-}" ]] || OUT="$OUT/$RUN_NAME"
fi
cd "$ROOT"
: "${MODEL:?Set MODEL to a full HF model checkpoint, not a tokenizer-only or LoRA directory}"
python -B -m scripts.evaluation.ppl.evaluate \
  --model "$MODEL" --input-dir "${DATA_DIR:-$ROOT/data/ppl}" --output "$OUT/metrics.json" \
  --window "${WINDOW:-2048}" --stride "${STRIDE:-1024}" \
  --dtype "${DTYPE:-auto}" --device "${DEVICE:-auto}" --attention "${ATTENTION:-sdpa}" \
  --loss-chunk-size "${LOSS_CHUNK_SIZE:-128}"
