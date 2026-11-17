#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# 必填：完整 Base 模型路径。也可作为脚本的第一个参数。
BASE_MODEL="${BASE_MODEL:-${1:-}}"
: "${BASE_MODEL:?Usage: bash scripts/smoke/run.sh /path/to/Base}"

# 每条路线独占一张卡；卡够时 Base/Char/PGCA 并行，单卡时自动串行。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$ROOT/.venv/train/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/.venv/evaluation/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
SMOKE_ROOT="${SMOKE_ROOT:-$ROOT/outputs/smoke}"

# 冒烟只验证运行能力；原始训练集和评测集不会被改写。
TRAIN_STEPS="${TRAIN_STEPS:-2}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1}"     # 本地每域/任务、LMEH 每个叶子任务的样本数
CPT_MAX_LENGTH="${CPT_MAX_LENGTH:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"  # Base/Char=vLLM，PGCA=HF

export PYTHONDONTWRITEBYTECODE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT"
"$TRAIN_PYTHON" -B -m scripts.smoke.run \
    --base-model "$BASE_MODEL" --devices "$CUDA_VISIBLE_DEVICES" \
    --train-python "$TRAIN_PYTHON" --eval-python "$EVAL_PYTHON" \
    --data-root "$DATA_ROOT" --output-root "$SMOKE_ROOT" \
    --train-steps "$TRAIN_STEPS" --train-samples "$TRAIN_SAMPLES" \
    --eval-samples "$EVAL_SAMPLES" --cpt-max-length "$CPT_MAX_LENGTH" \
    --max-new-tokens "$MAX_NEW_TOKENS" --backend "$INFERENCE_BACKEND"
