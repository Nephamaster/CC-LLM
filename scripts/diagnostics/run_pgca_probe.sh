#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# 只使用空闲卡；此脚本不查找、停止或接管正在训练的进程。
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an idle GPU before running the probe}"
PYTHON_CMD="${PYTHON_CMD:-$ROOT/scripts/train/ms_swift/env/bin/python}"
MODEL="${MODEL:-$ROOT/outputs/train/Qwen3-1.7B-Base-PGCA-CPT/for_exp}"
CACHE_DIR="${CACHE_DIR:-$ROOT/data/cc_llm_multitask/phase2_4096}"
OUT="${OUT:-$ROOT/outputs/experiments/rnd2/p0_pgca_probe}"
POLYPHONES="${POLYPHONES:-$ROOT/scripts/diagnostics/polyphones_review.tsv}"
SAMPLES="${SAMPLES:-512}"
SEED="${SEED:-42}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
GRADIENT_ROW_INDEX="${GRADIENT_ROW_INDEX:-0}"
LAYERS="${LAYERS:-11,12,13,14,15,16}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION="${ATTENTION:-sdpa}"

export CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
env -u PGCA_ENABLED -u NPROC_PER_NODE "$PYTHON_CMD" -B -m scripts.diagnostics.pgca_probe \
    --model "$MODEL" --cache "$CACHE_DIR" --output "$OUT" --polyphones "$POLYPHONES" \
    --samples "$SAMPLES" --seed "$SEED" --bootstrap "$BOOTSTRAP" \
    --gradient-row-index "$GRADIENT_ROW_INDEX" --layers "$LAYERS" \
    --dtype "$DTYPE" --attention "$ATTENTION" --device cuda:0
