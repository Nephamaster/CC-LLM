#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_WHEEL_URL="${TORCH_WHEEL_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu121/}"
WHEELHOUSE="${WHEELHOUSE:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

install_environment() {
    local target="$1" requirements="$2"
    if [[ ! -x "$target/bin/python" ]]; then
        "$PYTHON_BIN" -m venv "$target"
    fi

    local -a source_args=(--index-url "$PIP_INDEX_URL")
    if [[ -n "$WHEELHOUSE" ]]; then
        [[ -d "$WHEELHOUSE" ]] || { echo "WHEELHOUSE does not exist: $WHEELHOUSE" >&2; return 1; }
        source_args=(--no-index --find-links "$WHEELHOUSE")
    fi

    if ! "$target/bin/python" -c 'import torch' >/dev/null 2>&1; then
        if [[ -n "$WHEELHOUSE" ]]; then
            "$target/bin/python" -m pip install "${source_args[@]}" torch
        else
            "$target/bin/python" -m pip install --no-index --find-links "$TORCH_WHEEL_URL" torch
        fi
    fi
    "$target/bin/python" -m pip install "${source_args[@]}" -r "$requirements"
    "$target/bin/python" -m pip check
}

install_environment "$ROOT/.venv/train" "$ROOT/requirements/train.txt"
install_environment "$ROOT/.venv/evaluation" "$ROOT/requirements/evaluation.txt"

echo "Training environment:   $ROOT/.venv/train"
echo "Evaluation environment: $ROOT/.venv/evaluation"
echo "HF endpoint:            $HF_ENDPOINT"
