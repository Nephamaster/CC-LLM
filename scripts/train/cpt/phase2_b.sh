#!/usr/bin/env bash
set -euo pipefail

CC_PLUGIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CC_ROOT="${CC_ROOT:-$(cd -- "$CC_PLUGIN_DIR/../../.." && pwd)}"
MS_SWIFT_DIR="$(cd -- "$CC_PLUGIN_DIR/.." && pwd)"

# ==================== Phase2-B 独立参数：通常只改这里 ====================
# 修改 :- 后的默认值即可；命令行传入的同名环境变量优先。

# 环境、模型名称与产物根目录。本文件包含本阶段全部配置，不依赖 common.sh。
PYTHON_CMD="${PYTHON_CMD:-$MS_SWIFT_DIR/env/bin/python}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-${BASE_MODEL_NAME}-PGCA}"
TRAIN_ROOT="${TRAIN_ROOT:-$CC_ROOT/outputs/train}"

# 本阶段输入、输出及训练参数。MODEL 优先；留空时读取 Phase2-A 完成指针。
MODEL="${MODEL:-}"
CPT_ROOT="${CPT_ROOT:-$TRAIN_ROOT/${MODEL_NAME}-CPT}"
ADAPTATION_OUT="${ADAPTATION_OUT:-$CPT_ROOT/adaptation}"
OUT="${OUT:-$CPT_ROOT/joint}"
DATA_DIR="${DATA_DIR:-$CC_ROOT/data/corpus/runs/phase2/phase2-10b-default-6d14a3c4e453/final/ms_swift}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
CACHE_DIR="${CACHE_DIR:-$CC_ROOT/data/cc_llm_multitask/phase2_${MAX_LENGTH}}"
# 与 A 保持相同数据、长度、DATA_SEED 和预算；B 只使用扣除 A 后的剩余监督目标。
PHASE2_TOKENS="${PHASE2_TOKENS:-10000000000}"
ADAPTATION_TOKENS="${ADAPTATION_TOKENS:-100000000}"
PHASE2_B_LR="${PHASE2_B_LR:-1e-5}"                # 主干学习率
PHASE2_B_BRANCH_LR="${PHASE2_B_BRANCH_LR:-3e-5}"   # 新增分支学习率
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# 单机 GPU 与 batch；有效 batch = 卡数 × TRAIN_BATCH_SIZE × GRAD_ACCUMULATION。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"              # 留空时按可见 GPU 数推导
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"         # 每卡训练 batch
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"           # 每卡验证 batch
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-32}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-32}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

# ms-swift 原生数据预处理缓存复用；不是模型 KV cache，也不清理已准备的 CACHE_DIR。
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"

# 优化器；学习率在本文件上方的阶段参数区。
OPTIM="${OPTIM:-adamw_torch_fused}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
ADAM_EPSILON="${ADAM_EPSILON:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"              # 学习率 warmup，不是辅助任务权重 warmup
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 日志、保存与随机种子。选优要求 SAVE_STEPS 是 EVAL_STEPS 的整数倍。
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"                      # 同时控制准备缓存时的样本打乱


# ==================== 1. 初始化与输入模型 ====================
[[ -n "$MODEL_NAME" && "$MODEL_NAME" != */* && "$MODEL_NAME" != . && "$MODEL_NAME" != .. ]] || {
    echo "MODEL_NAME must be a model name, not a checkpoint path" >&2; exit 2;
}
export PYTHONDONTWRITEBYTECODE=1
IFS=, read -r -a cc_devices <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#cc_devices[@]}}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
export CC_LLM_STAGE=phase2b PGCA_ENABLED=true
export PHASE2_B_BRANCH_LR
if [[ -z "$MODEL" ]]; then
    [[ -f "$ADAPTATION_OUT/next_stage_model.txt" ]] || {
        echo "Set MODEL to a Phase2-A checkpoint or finish phase2_a.sh first" >&2; exit 1;
    }
    IFS= read -r MODEL < "$ADAPTATION_OUT/next_stage_model.txt"
fi

# ==================== 2. 准备或校验训练数据 ====================
RAW_CACHE="${CACHE_DIR}.raw"
prepare_args=(
    --input "$RAW_CACHE" --output "$CACHE_DIR"
    --model "$MODEL" --max-length "$MAX_LENGTH"
    --adaptation-tokens "$ADAPTATION_TOKENS" --total-tokens "$PHASE2_TOKENS"
    --world-size "$NPROC_PER_NODE" --batch-size "$TRAIN_BATCH_SIZE" --seed "$DATA_SEED"
)
shopt -s nullglob
train_files=("$DATA_DIR"/train/*.jsonl)
validation_files=("$DATA_DIR"/validation/*.jsonl)
shopt -u nullglob
[[ -f "$MODEL/config.json" && ${#train_files[@]} -gt 0 && ${#validation_files[@]} -gt 0 ]] || {
    echo "Model config or train/validation JSONL files missing" >&2; exit 1;
}
if [[ -f "$CACHE_DIR/preparation_report.json" ]]; then
    "$PYTHON_CMD" -B "$CC_PLUGIN_DIR/prepare_data.py" "${prepare_args[@]}" --verify
else
    [[ ! -e "$CACHE_DIR" && ! -e "$RAW_CACHE" ]] || {
        echo "Partial cache exists; inspect it and choose a new CACHE_DIR" >&2; exit 1;
    }
    env -u NPROC_PER_NODE "$PYTHON_CMD" -B -m swift.cli.main export \
        --external_plugins "$CC_PLUGIN_DIR/plugin.py" \
        --model "$MODEL" --model_type qwen3 --template cc_llm_multitask --load_args false \
        --dataset "${train_files[@]}" --val_dataset "${validation_files[@]}" \
        --dataset_num_proc "$DATASET_NUM_PROC" --data_seed "$DATA_SEED" --to_cached_dataset true \
        --strict true --load_from_cache_file "$LOAD_FROM_CACHE_FILE" --split_dataset_ratio 0 \
        --truncation_strategy split --max_length "$MAX_LENGTH" \
        --use_chat_template false --loss_scale all --disable_ignore_empty_think true --output_dir "$RAW_CACHE"
    "$PYTHON_CMD" -B "$CC_PLUGIN_DIR/prepare_data.py" "${prepare_args[@]}"
fi

# ==================== 3. 本阶段训练 / 断点恢复 ====================
training_output="$OUT"
add_version=true
resume_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    [[ -f "$RESUME_FROM_CHECKPOINT/trainer_state.json" ]] || {
        echo "Invalid resume checkpoint: $RESUME_FROM_CHECKPOINT" >&2; exit 1;
    }
    case "$(realpath -- "$RESUME_FROM_CHECKPOINT")/" in
        "$(realpath -- "$OUT")/"*) ;;
        *) echo "Resume checkpoint must belong to this stage output directory" >&2; exit 1 ;;
    esac
    MODEL="$RESUME_FROM_CHECKPOINT"
    training_output="$(dirname -- "$(realpath -- "$RESUME_FROM_CHECKPOINT")")"
    add_version=false
    resume_args=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

CC_LLM_OUTPUT_ROOT="$OUT" \
"$PYTHON_CMD" -B -m swift.cli.main pt \
    --external_plugins "$CC_PLUGIN_DIR/plugin.py" \
    --model "$MODEL" --model_type qwen3 --template cc_llm_multitask \
    --check_model false --load_args false --tuner_type full \
    --torch_dtype "$TORCH_DTYPE" --max_length "$MAX_LENGTH" \
    --cached_dataset "$CACHE_DIR/train_b" --cached_val_dataset "$CACHE_DIR/val" \
    --load_from_cache_file "$LOAD_FROM_CACHE_FILE" \
    --remove_unused_columns false --use_chat_template false \
    --loss_scale all --disable_ignore_empty_think true --truncation_strategy split \
    --loss_type cc_llm_multitask --optimizer cc_llm_multitask --callbacks cc_llm_multitask \
    --average_tokens_across_devices true \
    --packing false --padding_free false --sequence_parallel_size 1 \
    --use_liger_kernel false --use_logits_to_keep false --attn_impl sdpa \
    --num_train_epochs 1 \
    --learning_rate "$PHASE2_B_LR" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUMULATION" \
    --optim "$OPTIM" --adam_beta1 "$ADAM_BETA1" --adam_beta2 "$ADAM_BETA2" \
    --adam_epsilon "$ADAM_EPSILON" --weight_decay "$WEIGHT_DECAY" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --seed "$SEED" --data_seed "$DATA_SEED" \
    --logging_steps "$LOGGING_STEPS" --logging_nan_inf_filter false \
    --eval_on_start true --eval_strategy steps --eval_steps "$EVAL_STEPS" \
    --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --save_only_model false --load_best_model_at_end "true" \
    --metric_for_best_model eval_clm_loss --greater_is_better false \
    --add_version "$add_version" --output_dir "$training_output" \
    --freeze_parameters_ratio 0 \
    "${resume_args[@]}"

# 发布最终 CPT 结果，供下游微调自动读取。
mkdir -p -- "$CPT_ROOT"
cp -- "$OUT/next_stage_model.txt" "$CPT_ROOT/next_stage_model.txt"
