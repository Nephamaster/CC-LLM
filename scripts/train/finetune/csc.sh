#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# ==================== CSC 参数 ====================
# MODEL 优先；留空时读取 MODEL_NAME 对应训练根目录的完成指针，再尝试 models/MODEL_NAME。
MODEL="${MODEL:-}"
TRAIN="${TRAIN:-$CC_ROOT/data/finetune/csc/train.jsonl}"
DEV="${DEV:-$CC_ROOT/data/finetune/csc/validation.jsonl}"
OUT="${OUT:-$TRAIN_ROOT/${MODEL_NAME%-ALIGN}-CSC}"
MAX_LENGTH="${MAX_LENGTH:-768}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"

run_finetuning
