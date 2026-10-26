#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/base}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-$(basename -- "$BASE_MODEL")}"
CHAR_MODEL="${CHAR_MODEL:-$ROOT/models/${BASE_MODEL_NAME}-Char}"
PGCA_MODEL="${PGCA_MODEL:-$ROOT/models/${BASE_MODEL_NAME}-PGCA}"
BUILD_RESOURCES="${BUILD_RESOURCES:-$ROOT/models/.build/${BASE_MODEL_NAME}/hanzi}"
HANZI_SOURCE_DIR="${HANZI_SOURCE_DIR:-$DATA_ROOT/model_build/hanzi}"
UNIHAN_PATH="${UNIHAN_PATH:-$DATA_ROOT/model_build/unihan/Unihan.zip}"
UNIHAN_DIR="${UNIHAN_DIR:-$DATA_ROOT/model_build/unihan}"
PYTHON_CMD="${PYTHON_CMD:-$ROOT/.venv/train/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

[[ -x "$PYTHON_CMD" ]] || { echo "Run scripts/environment/run.sh first" >&2; exit 1; }
[[ -f "$BASE_MODEL/config.json" ]] || { echo "Place the local Base model at $BASE_MODEL or set BASE_MODEL" >&2; exit 1; }
for name in tghz2013.txt common_traditional.txt structure.tsv; do
    [[ -f "$HANZI_SOURCE_DIR/$name" ]] || { echo "Missing model-build resource: $HANZI_SOURCE_DIR/$name" >&2; exit 1; }
done
[[ -f "$UNIHAN_PATH" || -f "$UNIHAN_DIR/Unihan_Readings.txt" ]] || {
    echo "Missing Unihan.zip or extracted Unihan files under $UNIHAN_DIR" >&2; exit 1;
}
[[ ! -e "$CHAR_MODEL" && ! -e "$PGCA_MODEL" ]] || {
    echo "CHAR_MODEL or PGCA_MODEL already exists; choose new output paths" >&2; exit 1;
}

mkdir -p "$ROOT/models"
mkdir -p "$BUILD_RESOURCES"
cp -- "$HANZI_SOURCE_DIR/tghz2013.txt" "$BUILD_RESOURCES/tghz2013.txt"
cp -- "$HANZI_SOURCE_DIR/common_traditional.txt" "$BUILD_RESOURCES/common_traditional.txt"
cp -- "$HANZI_SOURCE_DIR/structure.tsv" "$BUILD_RESOURCES/structure.tsv"
if [[ -f "$HANZI_SOURCE_DIR/rare_high_freq.txt" ]]; then
    cp -- "$HANZI_SOURCE_DIR/rare_high_freq.txt" "$BUILD_RESOURCES/rare_high_freq.txt"
fi

export ROOT DATA_ROOT BASE_MODEL BASE_MODEL_NAME CHAR_MODEL PGCA_MODEL BUILD_RESOURCES
export UNIHAN_PATH UNIHAN_DIR PYTHON_CMD
bash "$ROOT/scripts/model_build/build_vocab.sh"
bash "$ROOT/scripts/model_build/build_embedding.sh"
bash "$ROOT/scripts/model_build/build_pgca.sh"

printf '%s\n' "$PGCA_MODEL" > "$ROOT/models/next_stage_model.txt"
printf '%s\n' "${BASE_MODEL_NAME}-PGCA" > "$ROOT/models/model_name.txt"
echo "Model build complete: $PGCA_MODEL"
