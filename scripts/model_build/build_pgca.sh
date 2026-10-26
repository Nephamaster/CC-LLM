#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_CMD="${PYTHON_CMD:-$ROOT/.venv/train/bin/python}"
CHAR_MODEL="${CHAR_MODEL:-$ROOT/models/char}"
PGCA_MODEL="${PGCA_MODEL:-$ROOT/models/pgca}"
AUX_PHONETIC_WEIGHT="${AUX_PHONETIC_WEIGHT:-0.03}"
AUX_GLYPH_WEIGHT="${AUX_GLYPH_WEIGHT:-0.03}"

"$PYTHON_CMD" -B -m src.pgca.build_pgca \
    --model-path "$CHAR_MODEL" --output-path "$PGCA_MODEL" \
    --context-encoder --auxiliary-heads --zero-init-output \
    --aux-phonetic-weight "$AUX_PHONETIC_WEIGHT" --aux-glyph-weight "$AUX_GLYPH_WEIGHT"

"$PYTHON_CMD" -B -m src.pgca.validate_pgca --model-path "$PGCA_MODEL"
