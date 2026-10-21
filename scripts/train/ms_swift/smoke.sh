#!/usr/bin/env bash
set -euo pipefail

CC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$CC_ROOT"

# 第二轮小规模验收；仅选择空闲 GPU。所有产物自动放入本次独立目录。
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to idle GPUs, preferably two for DDP}"
PYTHON_CMD="${PYTHON_CMD:-$CC_ROOT/scripts/train/ms_swift/env/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-Base-PGCA}"
PGCA_MODEL="${PGCA_MODEL:-$CC_ROOT/models/rnd2/$MODEL_NAME}"
SMOKE_PARENT="${SMOKE_PARENT:-$CC_ROOT/outputs/experiments/rnd2/smoke}"
IFS=, read -r -a smoke_devices <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#smoke_devices[@]}}"
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-2}"
SMOKE_STEPS="${SMOKE_STEPS:-4}" # 按满长序列估算预算，实际优化步数可能略多。
DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"
LOGGING_STEPS=1
EVAL_STEPS=2
SAVE_STEPS=2
SAVE_TOTAL_LIMIT=1
TORCH_DTYPE=bfloat16
WEIGHT_DECAY=0.01
GRADIENT_CHECKPOINTING=true
LOAD_FROM_CACHE_FILE=true
MAX_GRAD_NORM=1.0
SEED=42
DATA_SEED=42

[[ -n "$MODEL_NAME" && "$MODEL_NAME" != */* && "$MODEL_NAME" != . && "$MODEL_NAME" != .. ]] || {
    echo "MODEL_NAME must be a single directory component" >&2; exit 2;
}
for name in NPROC_PER_NODE GRAD_ACCUMULATION SMOKE_STEPS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
(( SMOKE_STEPS >= 4 && NPROC_PER_NODE == ${#smoke_devices[@]} )) || {
    echo "Need at least four budgeted steps and NPROC_PER_NODE equal to the visible GPU count" >&2; exit 2;
}
[[ -f "$PGCA_MODEL/config.json" ]] || { echo "Round-two initial model missing: $PGCA_MODEL" >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
export PYTHON_CMD CUDA_VISIBLE_DEVICES NPROC_PER_NODE MODEL_NAME PGCA_MODEL
export TRAIN_BATCH_SIZE EVAL_BATCH_SIZE GRAD_ACCUMULATION DATASET_NUM_PROC
export LOGGING_STEPS EVAL_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT TORCH_DTYPE WEIGHT_DECAY
export GRADIENT_CHECKPOINTING LOAD_FROM_CACHE_FILE MAX_GRAD_NORM SEED DATA_SEED

mkdir -p -- "$SMOKE_PARENT"
SMOKE_ROOT="$(mktemp -d "$SMOKE_PARENT/pgca-smoke.XXXXXX")"
SMOKE_ALIGN="$SMOKE_ROOT/training/${MODEL_NAME}-ALIGN"
SMOKE_CPT="$SMOKE_ROOT/training/${MODEL_NAME}-CPT"
phase1_targets=$((NPROC_PER_NODE * TRAIN_BATCH_SIZE * GRAD_ACCUMULATION * SMOKE_STEPS * 2048))
adaptation_targets=$((NPROC_PER_NODE * TRAIN_BATCH_SIZE * GRAD_ACCUMULATION * SMOKE_STEPS * 4096))
phase2_targets=$((2 * adaptation_targets))
echo "本次冒烟目录：$SMOKE_ROOT"
echo "只检查训练链路；合成训练/验证内容有重叠，不能用于效果评估。"
echo "需预留完整模型/优化器 checkpoint 空间；冒烟 checkpoint 不用于正式第二轮训练。"

"$PYTHON_CMD" -m pip check 2>&1 | tee "$SMOKE_ROOT/environment.log"
env -u PGCA_ENABLED -u CC_LLM_STAGE "$PYTHON_CMD" -B -m scripts.train.ms_swift.smoke_check prepare \
    --model "$PGCA_MODEL" --root "$SMOKE_ROOT" --world-size "$NPROC_PER_NODE" \
    --batch-size "$TRAIN_BATCH_SIZE" --accumulation "$GRAD_ACCUMULATION" --steps "$SMOKE_STEPS" \
    2>&1 | tee "$SMOKE_ROOT/preflight.console.log"

MODEL="$PGCA_MODEL" \
DATA_DIR="$SMOKE_ROOT/dataset" \
OUT="$SMOKE_ALIGN" \
CACHE_DIR="$SMOKE_ROOT/cache_phase1" \
MAX_LENGTH=2048 PHASE1_TOKENS="$phase1_targets" PHASE1_LR=1e-5 WARMUP_RATIO=0.03 \
RESUME_FROM_CHECKPOINT="" \
bash scripts/train/ms_swift/cpt/phase1.sh 2>&1 | tee "$SMOKE_ROOT/phase1.console.log"

MODEL="" ALIGN_OUT="$SMOKE_ALIGN" \
DATA_DIR="$SMOKE_ROOT/dataset" \
CPT_ROOT="$SMOKE_CPT" OUT="$SMOKE_CPT/adaptation" \
CACHE_DIR="$SMOKE_ROOT/cache_phase2" \
MAX_LENGTH=4096 PHASE2_TOKENS="$phase2_targets" ADAPTATION_TOKENS="$adaptation_targets" \
PHASE2_A_LR=5e-5 WARMUP_RATIO=0.03 \
RESUME_FROM_CHECKPOINT="" \
bash scripts/train/ms_swift/cpt/phase2_a.sh 2>&1 | tee "$SMOKE_ROOT/phase2a.console.log"

MODEL="" DATA_DIR="$SMOKE_ROOT/dataset" \
CPT_ROOT="$SMOKE_CPT" ADAPTATION_OUT="$SMOKE_CPT/adaptation" OUT="$SMOKE_CPT/joint" \
CACHE_DIR="$SMOKE_ROOT/cache_phase2" \
MAX_LENGTH=4096 PHASE2_TOKENS="$phase2_targets" ADAPTATION_TOKENS="$adaptation_targets" \
PHASE2_B_LR=1e-5 PHASE2_B_BRANCH_LR=3e-5 WARMUP_RATIO=0.01 \
RESUME_FROM_CHECKPOINT="" \
bash scripts/train/ms_swift/cpt/phase2_b.sh 2>&1 | tee "$SMOKE_ROOT/phase2b.console.log"

"$PYTHON_CMD" -B -m scripts.train.ms_swift.smoke_check summarize \
    --root "$SMOKE_ROOT" --model-name "$MODEL_NAME" 2>&1 | tee "$SMOKE_ROOT/summary.console.log"
echo "冒烟完成，验收报告：$SMOKE_ROOT/smoke_report.json"
