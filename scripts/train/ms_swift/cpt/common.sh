#!/usr/bin/env bash
set -euo pipefail

CC_PLUGIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CC_ROOT="${CC_ROOT:-$(cd -- "$CC_PLUGIN_DIR/../../../.." && pwd)}"
MS_SWIFT_DIR="$(cd -- "$CC_PLUGIN_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-$MS_SWIFT_DIR/env/bin/python}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-${BASE_MODEL_NAME}-PGCA}"
TRAIN_ROOT="${TRAIN_ROOT:-$CC_ROOT/outputs/train}"
[[ -n "$MODEL_NAME" && "$MODEL_NAME" != */* && "$MODEL_NAME" != . && "$MODEL_NAME" != .. ]] || {
    echo "MODEL_NAME must be a model name, not a checkpoint path" >&2; exit 2;
}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=, read -r -a cc_devices <<< "$CUDA_VISIBLE_DEVICES"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#cc_devices[@]}}"

prepare_cache() {
    local model="$1" data_dir="$2" cache="$3" length="$4" adaptation="$5" budget="$6"
    local raw="${cache}.raw"
    local -a train_files validation_files
    shopt -s nullglob
    train_files=("$data_dir"/train/*.jsonl)
    validation_files=("$data_dir"/validation/*.jsonl)
    shopt -u nullglob
    [[ -f "$model/config.json" && ${#train_files[@]} -gt 0 && ${#validation_files[@]} -gt 0 ]] || {
        echo "Model config or train/validation JSONL files missing" >&2; return 1;
    }
    if [[ -f "$cache/preparation_report.json" ]]; then
        "$PYTHON_CMD" -B "$CC_PLUGIN_DIR/prepare_data.py" --input "$raw" --output "$cache" \
            --model "$model" --max-length "$length" --adaptation-tokens "$adaptation" \
            --total-tokens "$budget" --world-size "$NPROC_PER_NODE" --batch-size "${TRAIN_BATCH_SIZE:-1}" --verify
        return
    fi
    [[ ! -e "$cache" && ! -e "$raw" ]] || {
        echo "Partial cache exists; inspect it and choose a new CACHE_DIR" >&2; return 1;
    }
    env -u NPROC_PER_NODE "$PYTHON_CMD" -B -m swift.cli.main export \
        --external_plugins "$CC_PLUGIN_DIR/plugin.py" \
        --model "$model" --model_type qwen3 --template cc_llm_multitask --load_args false \
        --dataset "${train_files[@]}" --val_dataset "${validation_files[@]}" \
        --dataset_num_proc "${DATASET_NUM_PROC:-8}" --to_cached_dataset true \
        --strict true --load_from_cache_file false --split_dataset_ratio 0 \
        --truncation_strategy split --max_length "$length" \
        --use_chat_template false --loss_scale all --disable_ignore_empty_think true --output_dir "$raw"
    "$PYTHON_CMD" -B "$CC_PLUGIN_DIR/prepare_data.py" --input "$raw" --output "$cache" \
        --model "$model" --max-length "$length" --adaptation-tokens "$adaptation" --total-tokens "$budget" \
        --world-size "$NPROC_PER_NODE" --batch-size "${TRAIN_BATCH_SIZE:-1}"
}

run_training() {
    local model="$1" train="$2" validation="$3" output="$4" length="$5" lr="$6" best="$7"
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
        resume_args=(--resume_from_checkpoint "$resume")
    else
        [[ ! -e "$output" ]] || { echo "Refusing to overwrite training directory: $output" >&2; return 1; }
    fi
    "$PYTHON_CMD" -B -m swift.cli.main pt \
        --external_plugins "$CC_PLUGIN_DIR/plugin.py" \
        --model "$model" --model_type qwen3 --template cc_llm_multitask \
        --check_model false --load_args false --tuner_type full --torch_dtype bfloat16 \
        --cached_dataset "$train" --cached_val_dataset "$validation" \
        --remove_unused_columns false \
        --use_chat_template false --loss_scale all --disable_ignore_empty_think true --truncation_strategy split \
        --loss_type cc_llm_multitask --optimizer cc_llm_multitask --callbacks cc_llm_multitask \
        --average_tokens_across_devices true --max_length "$length" \
        --packing false --padding_free false --sequence_parallel_size 1 \
        --use_liger_kernel false --use_logits_to_keep false --attn_impl sdpa \
        --learning_rate "$lr" --num_train_epochs 1 \
        --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-1}" \
        --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
        --gradient_accumulation_steps "${GRAD_ACCUMULATION:-8}" \
        --optim adamw_torch_fused --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 \
        --weight_decay 0.0 --lr_scheduler_type linear --max_grad_norm 1.0 --warmup_ratio 0.03 \
        --logging_steps "${LOGGING_STEPS:-10}" --gradient_checkpointing true \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --eval_on_start true --logging_nan_inf_filter false \
        --eval_strategy steps --save_strategy steps --eval_steps "${EVAL_STEPS:-500}" --save_steps "${SAVE_STEPS:-500}" \
        --save_total_limit 3 --save_only_model false --load_best_model_at_end "$best" \
        --metric_for_best_model eval_clm_loss --greater_is_better false \
        --seed 42 --data_seed 42 --add_version false --output_dir "$output" \
        --logging_dir "$output/logs" "${resume_args[@]}" "$@"
}
