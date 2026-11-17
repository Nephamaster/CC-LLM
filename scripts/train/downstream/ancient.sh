#!/usr/bin/env bash
set -euo pipefail

# 古汉语样本更长；四卡时默认有效 batch = 4 × 2 × 4 = 32。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-4}"
ADD_NON_THINKING_PREFIX="${ADD_NON_THINKING_PREFIX:-true}"
DISABLE_IGNORE_EMPTY_THINK="${DISABLE_IGNORE_EMPTY_THINK:-false}"
# ms-swift CLI 用 delete 表示模板内部 raise；配合 strict=true 不会静默跳过。
TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-delete}"
ENABLE_CHANNEL_LOSS="${ENABLE_CHANNEL_LOSS:-true}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_macro_task_loss}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-cc_llm_ancient}"

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# ==================== 古汉语多任务 SFT 参数 ====================
MODEL="${MODEL:-}"
ANCIENT_DATA_DIR="${ANCIENT_DATA_DIR:-$CC_ROOT/data/downstream/ancient}"
OUT="${OUT:-$TRAIN_ROOT/${MODEL_NAME%-ALIGN}-ACT}"
ANCIENT_VIEW_DIR="${ANCIENT_VIEW_DIR:-$OUT/.inputs}"
TRAIN="$ANCIENT_VIEW_DIR/train.jsonl"
DEV="$ANCIENT_VIEW_DIR/dev.jsonl"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"

export CC_LLM_MACRO_CHANNELS="${CC_LLM_MACRO_CHANNELS:-c2m,m2c,punctuate}"

"$PYTHON_CMD" -B "$CC_ROOT/scripts/train/runtime.py" ancient/prepare python \
    "$DOWNSTREAM_DIR/prepare_ancient_views.py" \
    --data-dir "$ANCIENT_DATA_DIR" \
    --output-dir "$ANCIENT_VIEW_DIR"

run_sft ancient
