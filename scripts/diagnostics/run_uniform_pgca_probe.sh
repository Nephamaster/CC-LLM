#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# 只运行离线推理，不接管或停止训练进程。参数集中在这里。
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an idle GPU}"
PYTHON_CMD="${PYTHON_CMD:-$ROOT/scripts/train/env/bin/python}"
P0_DIR="${P0_DIR:-$ROOT/outputs/experiments/rnd2/p0_pgca_probe}"
OUT="${OUT:-$ROOT/outputs/experiments/rnd2/p0_uniform_pgca}"
MODEL="${MODEL:-}"  # 留空：使用原 P0 manifest 的 checkpoint；可指定同一权重的迁移副本。
LIMIT="${LIMIT:-0}" # 0：全部符合条件的原 P0 样本；正数：仅前几条，用于预检。

model_args=()
if [[ -n "$MODEL" ]]; then
    model_args=(--model "$MODEL")
fi
export CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
env -u PGCA_ENABLED -u NPROC_PER_NODE "$PYTHON_CMD" -B -m scripts.diagnostics.uniform_pgca_probe \
    --p0 "$P0_DIR" --output "$OUT" --limit "$LIMIT" --device cuda:0 "${model_args[@]}"
