#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# ==================== Phase2 参数：通常只改这里 ====================
# GPU、batch、日志间隔、优化器等共享参数在 common.sh 顶部；环境变量覆盖优先。

# MODEL 优先；留空时，读取 ALIGN_OUT 根目录发布的最近完成的 Phase1 checkpoint。
MODEL="${MODEL:-}"
ALIGN_OUT="${ALIGN_OUT:-$TRAIN_ROOT/${MODEL_NAME}-ALIGN}"
DATA_DIR="${DATA_DIR:-$CC_ROOT/data/corpus/runs/phase2/phase2-10b-default-6d14a3c4e453/final/ms_swift}"
OUT="${OUT:-$TRAIN_ROOT/${MODEL_NAME}-CPT}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
CACHE_DIR="${CACHE_DIR:-$CC_ROOT/data/cc_llm_multitask/phase2_${MAX_LENGTH}}"

# A/B 共享总 token 上限；数据不足时用实际可用量，B 使用扣除 A 后的剩余目标。
PHASE2_TOKENS="${PHASE2_TOKENS:-10000000000}"
ADAPTATION_TOKENS="${ADAPTATION_TOKENS:-100000000}"
PHASE2_A_LR="${PHASE2_A_LR:-5e-5}"                 # A：仅新增分支
PHASE2_B_LR="${PHASE2_B_LR:-1e-5}"                 # B：主干
PHASE2_B_BRANCH_LR="${PHASE2_B_BRANCH_LR:-3e-5}"   # B：新增分支，由插件建立独立参数组
# 辅助任务权重/warmup 来自模型 config 的 pgca_aux_*，与上面的学习率不是同一参数。

RUN_STAGE="${RUN_STAGE:-all}"                     # all / a / b
RESUME_A="${RESUME_A:-}"                         # 留空为新训练，否则填 A 的 checkpoint
RESUME_B="${RESUME_B:-}"                         # 留空为新训练，否则填 B 的 checkpoint

# ==================== 执行：A 分支适配 → B 全参数联合训练 ====================
[[ "$RUN_STAGE" == all || "$RUN_STAGE" == a || "$RUN_STAGE" == b ]] || { echo "RUN_STAGE must be all/a/b" >&2; exit 1; }
if [[ -z "$MODEL" ]]; then
    [[ -f "$ALIGN_OUT/next_stage_model.txt" ]] || { echo "Set MODEL to the Phase1 checkpoint" >&2; exit 1; }
    IFS= read -r MODEL < "$ALIGN_OUT/next_stage_model.txt"
fi
export PHASE2_B_BRANCH_LR
export CC_LLM_STAGE=phase2a PGCA_ENABLED=true
prepare_cache "$MODEL" "$DATA_DIR" "$CACHE_DIR" "$MAX_LENGTH" "$ADAPTATION_TOKENS" "$PHASE2_TOKENS"

# A 冻结主干，保存最后 checkpoint 作为 B 的起点。
if [[ "$RUN_STAGE" != b ]]; then
    CC_RESUME_FROM_CHECKPOINT="$RESUME_A" \
    run_training "$MODEL" "$CACHE_DIR/train_a" "$CACHE_DIR/val" "$OUT/adaptation" "$MAX_LENGTH" "$PHASE2_A_LR" false \
        --freeze_parameters_ratio 1 \
        --trainable_parameters_regex '(^model\.layers\.[0-9]+\.pgca_attn\.|^model\.feature_memory_builder\.|^auxiliary_heads\.)'
fi
[[ "$RUN_STAGE" != a ]] || exit 0
[[ -f "$OUT/adaptation/next_stage_model.txt" ]] || { echo "Phase2-A must finish before Phase2-B" >&2; exit 1; }
IFS= read -r MODEL < "$OUT/adaptation/next_stage_model.txt"
export CC_LLM_STAGE=phase2b
# B 新建优化器/调度器，主干和新增分支使用不同学习率，并按 eval_clm_loss 选优。
CC_RESUME_FROM_CHECKPOINT="$RESUME_B" \
run_training "$MODEL" "$CACHE_DIR/train_b" "$CACHE_DIR/val" "$OUT/joint" "$MAX_LENGTH" "$PHASE2_B_LR" true \
    --freeze_parameters_ratio 0
cp -- "$OUT/joint/next_stage_model.txt" "$OUT/next_stage_model.txt"
