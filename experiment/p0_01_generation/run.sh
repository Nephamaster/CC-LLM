#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GPU="${GPU:-1}"
PYTHON_CMD="${PYTHON_CMD:-$ROOT/.venv/evaluation/bin/python}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname -- "$PYTHON_CMD"):$PATH"
"$PYTHON_CMD" -B -m experiment.p0_01_generation.run "$@"
