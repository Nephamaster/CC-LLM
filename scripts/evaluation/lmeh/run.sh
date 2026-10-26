#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$TASK_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/evaluation/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT"
args=(--benchmark "${BENCHMARK:-all}" --stage "${STAGE:-run}"
      --output-root "${OUT:-$ROOT/outputs/evaluation}"
      --device "${DEVICE:-cuda:0}" --dtype "${DTYPE:-bfloat16}"
      --batch-size "${BATCH_SIZE:-1}" --max-length "${MAX_LENGTH:-8192}"
      --backend "${INFERENCE_BACKEND:-auto}" --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}")
[[ -z "${MODEL:-}" ]] || args+=(--model "$MODEL")
[[ -z "${MODEL_NAME:-}" ]] || args+=(--model-name "$MODEL_NAME")
[[ -z "${RUN_NAME:-}" ]] || args+=(--run-name "$RUN_NAME")
[[ -z "${DATA_DIR:-}" ]] || args+=(--data-dir "$DATA_DIR")
[[ -z "${LIMIT:-}" ]] || args+=(--limit "$LIMIT")
"$PYTHON" -B -m scripts.evaluation.lmeh.run "${args[@]}"
