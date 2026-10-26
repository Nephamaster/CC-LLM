#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# ==================== CGEC 参数 ====================
# 使用用户已准备好的 train/validation；不从 FCGEC、NaCGEC 测试目录另取训练数据。
MODEL="${MODEL:-}"
TRAIN="${TRAIN:-$CC_ROOT/data/finetune/cgec/train.jsonl}"
DEV="${DEV:-$CC_ROOT/data/finetune/cgec/validation.jsonl}"
OUT="${OUT:-$TRAIN_ROOT/${MODEL_NAME%-ALIGN}-CGEC}"
MAX_LENGTH="${MAX_LENGTH:-768}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"

run_finetuning
