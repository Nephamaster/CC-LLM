#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-$ROOT/.venv/train/bin/python}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/base}"
CHAR_MODEL="${CHAR_MODEL:-$ROOT/models/char}"

cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_CMD" -B -m src.embedding.build_embedding all \
    --base-model-path "$BASE_MODEL" \
    --char-model-path "$CHAR_MODEL" \
    --skip-migrate
