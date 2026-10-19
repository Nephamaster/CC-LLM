#!/usr/bin/env bash
set -euo pipefail

CC_PLUGIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CC_ROOT="${CC_ROOT:-$(cd -- "$CC_PLUGIN_DIR/../../../.." && pwd)}"
MS_SWIFT_DIR="$(cd -- "$CC_PLUGIN_DIR/.." && pwd)"

# ==================== 两阶段共享参数：通常只改这里 ====================
# 修改 :- 后的默认值即可；命令行传入的同名环境变量优先。

# 环境、模型名称与产物根目录。各阶段的具体模型/数据路径在 phase1.sh、phase2.sh。
PYTHON_CMD="${PYTHON_CMD:-$MS_SWIFT_DIR/env/bin/python}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-${BASE_MODEL_NAME}-PGCA}"
TRAIN_ROOT="${TRAIN_ROOT:-$CC_ROOT/outputs/train}"

# 单机 GPU 与 batch；有效 batch = 卡数 × TRAIN_BATCH_SIZE × GRAD_ACCUMULATION。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"              # 留空时按可见 GPU 数推导
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"         # 每卡训练 batch
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"           # 每卡验证 batch
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-32}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

# ms-swift 原生数据预处理缓存复用；不是模型 KV cache，也不清理已准备的 CACHE_DIR。
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"

# 优化器；各阶段的学习率在对应 phase 脚本顶部。
OPTIM="${OPTIM:-adamw_torch_fused}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
ADAM_EPSILON="${ADAM_EPSILON:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
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

# ==================== 以下为公共执行逻辑，通常无需修改 ====================
[[ -n "$MODEL_NAME" && "$MODEL_NAME" != */* && "$MODEL_NAME" != . && "$MODEL_NAME" != .. ]] || {
    echo "MODEL_NAME must be a model name, not a checkpoint path" >&2; exit 2;
}
export PYTHONDONTWRITEBYTECODE=1
IFS=, read -r -a cc_devices <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#cc_devices[@]}}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE

prepare_cache() {
    local model="$1" data_dir="$2" cache="$3"
    local max_length="$4" adaptation_tokens="$5" total_tokens="$6"
    local raw="${cache}.raw"
    local -a train_files validation_files
    local -a prepare_args=(
        --input "$raw" --output "$cache"
        --model "$model" --max-length "$max_length"
        --adaptation-tokens "$adaptation_tokens" --total-tokens "$total_tokens"
        --world-size "$NPROC_PER_NODE" --batch-size "$TRAIN_BATCH_SIZE" --seed "$DATA_SEED"
    )
    shopt -s nullglob
    train_files=("$data_dir"/train/*.jsonl)
    validation_files=("$data_dir"/validation/*.jsonl)
    shopt -u nullglob
    [[ -f "$model/config.json" && ${#train_files[@]} -gt 0 && ${#validation_files[@]} -gt 0 ]] || {
        echo "Model config or train/validation JSONL files missing" >&2; return 1;
    }
    if [[ -f "$cache/preparation_report.json" ]]; then
        "$PYTHON_CMD" -B "$CC_PLUGIN_DIR/prepare_data.py" "${prepare_args[@]}" --verify
        return
    fi
    [[ ! -e "$cache" && ! -e "$raw" ]] || {
        echo "Partial cache exists; inspect it and choose a new CACHE_DIR" >&2; return 1;
    }
    env -u NPROC_PER_NODE "$PYTHON_CMD" -B -m swift.cli.main export \
        --external_plugins "$CC_PLUGIN_DIR/plugin.py" \
        --model "$model" --model_type qwen3 --template cc_llm_multitask --load_args false \
        --dataset "${train_files[@]}" --val_dataset "${validation_files[@]}" \
        --dataset_num_proc "$DATASET_NUM_PROC" --data_seed "$DATA_SEED" --to_cached_dataset true \
        --strict true --load_from_cache_file "$LOAD_FROM_CACHE_FILE" --split_dataset_ratio 0 \
        --truncation_strategy split --max_length "$max_length" \
        --use_chat_template false --loss_scale all --disable_ignore_empty_think true --output_dir "$raw"
    "$PYTHON_CMD" -B "$CC_PLUGIN_DIR/prepare_data.py" "${prepare_args[@]}"
}

run_training() {
    local model="$1" train="$2" validation="$3" output="$4"
    local max_length="$5" learning_rate="$6" load_best="$7"
    local output_root="$output" add_version=true
    local resume="${CC_RESUME_FROM_CHECKPOINT:-}"
    local -a resume_args=()
    shift 7
    if [[ -n "$resume" ]]; then
        [[ -f "$resume/trainer_state.json" ]] || { echo "Invalid resume checkpoint: $resume" >&2; return 1; }
        case "$(realpath -- "$resume")/" in
            "$(realpath -- "$output")/"*) ;;
            *) echo "Resume checkpoint must belong to this stage output directory" >&2; return 1 ;;
        esac
        model="$resume"
        output="$(dirname -- "$(realpath -- "$resume")")"
        add_version=false
        resume_args=(--resume_from_checkpoint "$resume")
    fi
    local -a training_args=(
        # 模型与数据。
        --external_plugins "$CC_PLUGIN_DIR/plugin.py"
        --model "$model" --model_type qwen3 --template cc_llm_multitask
        --check_model false --load_args false --tuner_type full
        --torch_dtype "$TORCH_DTYPE" --max_length "$max_length"
        --cached_dataset "$train" --cached_val_dataset "$validation"
        --load_from_cache_file "$LOAD_FROM_CACHE_FILE"

        # 当前链路固定约束：完整 logits、非 packing、单轮消耗指定 token 预算。
        --remove_unused_columns false --use_chat_template false
        --loss_scale all --disable_ignore_empty_think true --truncation_strategy split
        --loss_type cc_llm_multitask --optimizer cc_llm_multitask --callbacks cc_llm_multitask
        --average_tokens_across_devices true
        --packing false --padding_free false --sequence_parallel_size 1
        --use_liger_kernel false --use_logits_to_keep false --attn_impl sdpa
        --num_train_epochs 1

        # 训练超参数：数值统一来自顶部参数区或对应 phase 脚本。
        --learning_rate "$learning_rate"
        --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
        --per_device_eval_batch_size "$EVAL_BATCH_SIZE"
        --gradient_accumulation_steps "$GRAD_ACCUMULATION"
        --optim "$OPTIM" --adam_beta1 "$ADAM_BETA1" --adam_beta2 "$ADAM_BETA2"
        --adam_epsilon "$ADAM_EPSILON" --weight_decay "$WEIGHT_DECAY"
        --lr_scheduler_type "$LR_SCHEDULER_TYPE" --warmup_ratio "$WARMUP_RATIO"
        --max_grad_norm "$MAX_GRAD_NORM"
        --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
        --gradient_checkpointing_kwargs '{"use_reentrant": false}'
        --seed "$SEED" --data_seed "$DATA_SEED"

        # 验证、保存和选优；保留优化器状态以支持断点恢复。
        --logging_steps "$LOGGING_STEPS" --logging_nan_inf_filter false
        --eval_on_start true --eval_strategy steps --eval_steps "$EVAL_STEPS"
        --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT"
        --save_only_model false --load_best_model_at_end "$load_best"
        --metric_for_best_model eval_clm_loss --greater_is_better false
        --add_version "$add_version" --output_dir "$output"
    )
    # Swift 分配版本目录及日志目录；完成后插件向稳定根目录发布 checkpoint 指针。
    CC_LLM_OUTPUT_ROOT="$output_root" \
    "$PYTHON_CMD" -B -m swift.cli.main pt "${training_args[@]}" "${resume_args[@]}" "$@"
}
