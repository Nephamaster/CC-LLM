#!/usr/bin/env bash
set -euo pipefail
CC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CC_ROOT"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_CMD="${PYTHON_CMD:-python}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3-1.7B-Base}"
CHAR_MODEL="${CHAR_MODEL:-$CC_ROOT/models/${BASE_MODEL_NAME}-Char}"
PGCA_MODEL="${PGCA_MODEL:-$CC_ROOT/models/${BASE_MODEL_NAME}-PGCA}"

"$PYTHON_CMD" -B -m src.pgca.build_pgca \
    --model-path "$CHAR_MODEL" --output-path "$PGCA_MODEL" \
    --pgca-layers 11,12,13,14,15,16 \
    --context-encoder --auxiliary-heads --zero-init-output

"$PYTHON_CMD" -B -m src.pgca.validate_pgca --model-path "$PGCA_MODEL"
