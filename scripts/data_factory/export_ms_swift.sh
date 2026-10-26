#!/usr/bin/env bash
set -euo pipefail

CC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-$CC_ROOT/scripts/train/env/bin/python}"
export PYTHONDONTWRITEBYTECODE=1

: "${MODEL_PATH:?set MODEL_PATH}"
: "${FINAL_DIR:?set FINAL_DIR to the V2 final directory}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
CACHED_DIR="${CACHED_DIR:-${FINAL_DIR}/ms_swift_cached_${MAX_LENGTH}}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-32}"

"$PYTHON_CMD" -B -m swift.cli.main export \
  --model "${MODEL_PATH}" \
  --dataset "${FINAL_DIR}/ms_swift/train" \
  --val_dataset "${FINAL_DIR}/ms_swift/validation" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --to_cached_dataset true \
  --truncation_strategy split \
  --max_length "${MAX_LENGTH}" \
  --use_chat_template false \
  --loss_scale all \
  --output_dir "${CACHED_DIR}"

printf 'cached_dataset=%s/train\ncached_val_dataset=%s/val\n' "${CACHED_DIR}" "${CACHED_DIR}"
