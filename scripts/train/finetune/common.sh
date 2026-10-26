#!/usr/bin/env bash
set -euo pipefail

FINETUNE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CC_ROOT="${CC_ROOT:-$(cd -- "$FINETUNE_DIR/../../.." && pwd)}"

# ==================== 下游 LoRA 共享参数 ====================
# 修改 :- 后的默认值；命令行同名环境变量优先。任务路径、长度、学习率在任务脚本顶部。
PYTHON_CMD="${PYTHON_CMD:-$CC_ROOT/scripts/train/env/bin/python}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-${BASE_MODEL_NAME}-PGCA-CPT}"  # 输入模型名称，不是 checkpoint-N
TRAIN_ROOT="${TRAIN_ROOT:-$CC_ROOT/outputs/train}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-32}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-true}"
ADD_NON_THINKING_PREFIX="${ADD_NON_THINKING_PREFIX:-false}"
DISABLE_IGNORE_EMPTY_THINK="${DISABLE_IGNORE_EMPTY_THINK:-true}"
TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-delete}"
ENABLE_CHANNEL_LOSS="${ENABLE_CHANNEL_LOSS:-false}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-cc_llm_correction}"

# 主干与 PGCA 都走框架的 all-linear LoRA；不用 CPT 的辅助 loss/优化器插件。
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
OPTIM="${OPTIM:-adamw_torch}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPSILON="${ADAM_EPSILON:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
MAX_STEPS="${MAX_STEPS:--1}"                    # 正数用于短跑；默认按任务脚本的 epoch 数
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_loss}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

# ==================== 执行逻辑，通常无需修改 ====================
[[ -n "$MODEL_NAME" && "$MODEL_NAME" != */* && "$MODEL_NAME" != . && "$MODEL_NAME" != .. ]] || {
    echo "MODEL_NAME must be the input model name, not a checkpoint path" >&2; exit 2;
}
IFS=, read -r -a finetune_devices <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#finetune_devices[@]}}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PYTHONDONTWRITEBYTECODE=1

run_finetuning() {
    local model="$MODEL" output="$OUT" add_version=true
    local -a resume_args=()
    if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
        [[ -f "$RESUME_FROM_CHECKPOINT/trainer_state.json" && -f "$RESUME_FROM_CHECKPOINT/adapter_config.json" ]] || {
            echo "RESUME_FROM_CHECKPOINT must be an SFT LoRA training checkpoint" >&2; return 1;
        }
        case "$(realpath -- "$RESUME_FROM_CHECKPOINT")/" in
            "$(realpath -- "$OUT")/"*) ;;
            *) echo "Resume checkpoint must belong to OUT" >&2; return 1 ;;
        esac
        # 恢复优先读取 adapter 记录的原基座，不能误用后来更新的 CPT 完成指针。
        if [[ -z "$model" ]]; then
            model="$("$PYTHON_CMD" -B -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get("base_model_name_or_path")
if not isinstance(value, str) or not value:
    raise SystemExit("Adapter has no base_model_name_or_path; set MODEL explicitly")
print(value)' "$RESUME_FROM_CHECKPOINT/adapter_config.json")"
        fi
        output="$(dirname -- "$(realpath -- "$RESUME_FROM_CHECKPOINT")")"
        add_version=false
        resume_args=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
    fi
    if [[ -z "$model" ]]; then
        local pointer="$TRAIN_ROOT/$MODEL_NAME/next_stage_model.txt"
        if [[ -f "$pointer" ]]; then
            IFS= read -r model < "$pointer"
        else
            model="$CC_ROOT/models/$MODEL_NAME"
        fi
    fi
    [[ -f "$model/config.json" ]] || {
        echo "Set MODEL to a full base-model checkpoint and MODEL_NAME to its canonical name: $model" >&2; return 1;
    }
    [[ -s "$TRAIN" && -s "$DEV" ]] || {
        echo "Training/validation JSONL missing or empty: TRAIN=$TRAIN DEV=$DEV" >&2; return 1;
    }
    local -a args=(
        --external_plugins "$FINETUNE_DIR/plugin.py"
        --model "$model" --model_type qwen3 --check_model false --load_args false
        # 任务脚本选择已验收的模板；插件仅处理 PGCA 前向及闲置辅助头。
        --template "$TEMPLATE_TYPE" --use_chat_template true --enable_thinking false
        --add_non_thinking_prefix "$ADD_NON_THINKING_PREFIX" \
        --disable_ignore_empty_think "$DISABLE_IGNORE_EMPTY_THINK" --loss_scale default
        --callbacks cc_llm_correction
        --dataset "$TRAIN" --val_dataset "$DEV" --split_dataset_ratio 0
        --dataset_shuffle true --train_dataloader_shuffle true --val_dataset_shuffle false
        --dataset_num_proc "$DATASET_NUM_PROC" --load_from_cache_file "$LOAD_FROM_CACHE_FILE"
        --strict true --truncation_strategy "$TRUNCATION_STRATEGY" --max_length "$MAX_LENGTH"
        --tuner_type lora --tuner_backend peft --target_modules all-linear
        --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT"
        --torch_dtype "$TORCH_DTYPE" --attn_impl sdpa
        --learning_rate "$LEARNING_RATE" --num_train_epochs "$NUM_TRAIN_EPOCHS" --max_steps "$MAX_STEPS"
        --per_device_train_batch_size "$TRAIN_BATCH_SIZE" --per_device_eval_batch_size "$EVAL_BATCH_SIZE"
        --gradient_accumulation_steps "$GRAD_ACCUMULATION" --max_grad_norm "$MAX_GRAD_NORM"
        --optim "$OPTIM" --adam_beta1 "$ADAM_BETA1" --adam_beta2 "$ADAM_BETA2"
        --adam_epsilon "$ADAM_EPSILON" --weight_decay "$WEIGHT_DECAY"
        --lr_scheduler_type "$LR_SCHEDULER_TYPE" --warmup_ratio "$WARMUP_RATIO"
        --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
        --gradient_checkpointing_kwargs '{"use_reentrant": false}'
        --ddp_find_unused_parameters "$DDP_FIND_UNUSED_PARAMETERS"
        --enable_channel_loss "$ENABLE_CHANNEL_LOSS"
        --packing false --padding_free false --sequence_parallel_size 1
        --use_liger_kernel false --use_logits_to_keep false
        --logging_steps "$LOGGING_STEPS" --logging_nan_inf_filter false --eval_on_start true
        --eval_strategy steps --eval_steps "$EVAL_STEPS" --save_strategy steps --save_steps "$SAVE_STEPS"
        --save_total_limit "$SAVE_TOTAL_LIMIT" --save_only_model false --load_best_model_at_end true
        --metric_for_best_model "$METRIC_FOR_BEST_MODEL" --greater_is_better false --max_new_tokens "$MAX_NEW_TOKENS"
        --seed "$SEED" --data_seed "$DATA_SEED" --add_version "$add_version" --output_dir "$output"
    )
    # 不导出 .raw、不另建准备缓存，直接使用 ms-swift 原生 JSONL 加载/缓存机制。
    PGCA_ENABLED=true CC_LLM_SFT_OUTPUT_ROOT="$OUT" \
    "$PYTHON_CMD" -B -m swift.cli.main sft "${args[@]}" "${resume_args[@]}"
}
