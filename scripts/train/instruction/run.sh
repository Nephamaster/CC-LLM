#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$TASK_DIR/../../.." && pwd)"
cd "$ROOT"

# 同名环境变量优先。MODEL 是 Base 或已经训练完的 Char-CPT / PGCA-CPT 完整权重。
MODEL_NAME="${MODEL_NAME:-}"
MODEL="${MODEL:-}"
INSTRUCTION_DATA="${INSTRUCTION_DATA:-}"
SUPERVISED_TOKENS="${SUPERVISED_TOKENS:-0}"  # 0 只表示全量一遍；正式预算待三条路线统计后确定
MAX_LENGTH="${MAX_LENGTH:-4096}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-8}"
ADD_NON_THINKING_PREFIX=false
DISABLE_IGNORE_EMPTY_THINK=true
TEMPLATE_TYPE=cc_llm_instruction
if [[ -z "$MODEL_NAME" ]]; then
    [[ -s "$ROOT/models/model_name.txt" ]] || { echo "Set MODEL_NAME to the General-SFT input model name" >&2; exit 1; }
    IFS= read -r MODEL_NAME < "$ROOT/models/model_name.txt"
    MODEL_NAME="${MODEL_NAME}-CPT"
fi
source "$TASK_DIR/../finetune/common.sh"

if [[ -z "$MODEL" ]]; then
    pointer="$TRAIN_ROOT/$MODEL_NAME/next_stage_model.txt"
    if [[ -s "$pointer" ]]; then IFS= read -r MODEL < "$pointer"; else MODEL="$ROOT/models/$MODEL_NAME"; fi
fi
if [[ -z "$INSTRUCTION_DATA" && ( -z "${TRAIN:-}" || -z "${DEV:-}" ) ]]; then
    IFS= read -r INSTRUCTION_DATA < "$ROOT/data/instruction/prepared/latest.txt"
fi
TRAIN="${TRAIN:-$INSTRUCTION_DATA/train.jsonl}"
DEV="${DEV:-$INSTRUCTION_DATA/dev.jsonl}"
INS_OUT="${INS_OUT:-${OUT:-$TRAIN_ROOT/${MODEL_NAME}-INS}}"
OUT="$INS_OUT/adapters"
mkdir -p "$OUT"
source_train="$TRAIN"
source_dev="$DEV"
view="$("$PYTHON_CMD" -B -m scripts.evaluation.runs "$INS_OUT/inputs")"
TRAIN="$view/train.jsonl"
DEV="$view/dev.jsonl"
# 完整保留多轮答案和 EOS；超长对话排除并计数，不从答案中间截断。
"$PYTHON_CMD" -B -m scripts.train.instruction.budget --model "$MODEL" --input "$source_train" \
  --output "$TRAIN" --tokens "$SUPERVISED_TOKENS" --max-length "$MAX_LENGTH"
"$PYTHON_CMD" -B -m scripts.train.instruction.budget --model "$MODEL" --input "$source_dev" \
  --output "$DEV" --tokens 0 --max-length "$MAX_LENGTH"
if [[ "$SUPERVISED_TOKENS" != 0 ]]; then NUM_TRAIN_EPOCHS=1; fi
run_finetuning instruction
"$PYTHON_CMD" -B -m scripts.train.instruction.finish --model "$MODEL" --adapter-root "$OUT" \
    --dev "$DEV" --output "$INS_OUT"
