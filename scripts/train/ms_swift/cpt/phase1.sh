#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# ==================== Phase1 参数：通常只改这里 ====================
# GPU、batch、日志间隔、优化器等共享参数在 common.sh 顶部；环境变量覆盖优先。

# 输入、输出与缓存。
MODEL="${MODEL:-$CC_ROOT/models/$MODEL_NAME}"
DATA_DIR="${DATA_DIR:-$CC_ROOT/data/corpus/runs/phase1/phase1-1b-default-c97f9387391d/final/ms_swift}"
OUT="${OUT:-$TRAIN_ROOT/${MODEL_NAME}-ALIGN}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
CACHE_DIR="${CACHE_DIR:-$CC_ROOT/data/cc_llm_multitask/phase1_${MAX_LENGTH}}"

# 训练 token 上限与学习率；数据不足时使用全部可用数据，tokens=0 表示不设上限。
PHASE1_TOKENS="${PHASE1_TOKENS:-1000000000}"
PHASE1_LR="${PHASE1_LR:-1e-5}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"  # 留空为新训练，否则填本阶段 checkpoint

# ==================== 执行：主干训练，音形分支旁路且冻结 ====================
export CC_LLM_STAGE=phase1 PGCA_ENABLED=false

prepare_cache "$MODEL" "$DATA_DIR" "$CACHE_DIR" "$MAX_LENGTH" 0 "$PHASE1_TOKENS"
CC_RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
run_training "$MODEL" "$CACHE_DIR/train" "$CACHE_DIR/val" "$OUT" "$MAX_LENGTH" "$PHASE1_LR" true \
    --freeze_parameters_ratio 0 \
    --freeze_parameters_regex '(^model\.layers\.[0-9]+\.pgca_attn\.|^model\.feature_memory_builder\.|^auxiliary_heads\.)'
