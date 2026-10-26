#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-$ROOT/.venv/train/bin/python}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/base}"
CHAR_MODEL="${CHAR_MODEL:-$ROOT/models/char}"
BUILD_RESOURCES="${BUILD_RESOURCES:-$ROOT/models/.build/hanzi}"
UNIHAN_PATH="${UNIHAN_PATH:-$ROOT/data/model_build/unihan/Unihan.zip}"
UNIHAN_DIR="${UNIHAN_DIR:-$ROOT/data/model_build/unihan}"

cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_CMD" -B -m src.vocab.build_vocab all \
    --base-model-path "$BASE_MODEL" \
    --char-model-path "$CHAR_MODEL" \
    --hanzi-dir "$BUILD_RESOURCES" \
    --features-dir "$CHAR_MODEL/features" \
    --unihan-path "$UNIHAN_PATH" \
    --fallback-unihan-dir "$UNIHAN_DIR" \
    --strict
