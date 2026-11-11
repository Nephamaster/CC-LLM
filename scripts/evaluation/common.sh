#!/usr/bin/env bash

configure_evaluation() {
    local task_dir="$1" benchmark="$2" protocol="${3:-}"
    local component
    ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    PYTHON="${PYTHON:-$ROOT/.venv/evaluation/bin/python}"
    export PYTHONDONTWRITEBYTECODE=1
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    cd "$ROOT"
    if [[ -n "${RUN_DIR:-}" ]]; then
        OUT="$RUN_DIR"
        [[ -d "$OUT" ]] || { echo "RUN_DIR does not exist: $OUT" >&2; return 2; }
        return
    fi
    if [[ -z "${OUT:-}" ]]; then
        : "${MODEL_NAME:?Set MODEL_NAME to the evaluated model name, or supply OUT explicitly}"
        for component in "$benchmark" "$MODEL_NAME" "$protocol" "${RUN_NAME:-}"; do
            [[ "$component" != */* && "$component" != . && "$component" != .. ]] || {
                echo "Evaluation names must be single directory components" >&2; return 2;
            }
        done
        OUT="$ROOT/outputs/evaluation/$benchmark/$MODEL_NAME"
        [[ -z "$protocol" ]] || OUT="$OUT/$protocol"
        [[ -z "${RUN_NAME:-}" ]] || OUT="$OUT/$RUN_NAME"
    fi
    case "${STAGE:-all}" in
        all|prepare) OUT="$("$PYTHON" -B -m scripts.evaluation.runs "$OUT")" ;;
        infer|score|judge) OUT="$("$PYTHON" -B -m scripts.evaluation.runs "$OUT" --latest)" ;;
        *) echo "Invalid STAGE: $STAGE" >&2; return 2 ;;
    esac
}

check_sft_dev() {
    local task="$1"
    : "${MODEL:?Set MODEL to the full SFT base model}"
    local -a args=(--model "$MODEL" --task "$task" --output "$OUT/dev_check")
    [[ -z "${LORA:-}" ]] || args+=(--adapter "$LORA")
    printf '[evaluation][%s] fixed-Dev generation/LoRA/cache acceptance\n' "$task"
    "$PYTHON" -B -m scripts.train.finetune.dev_check "${args[@]}"
    export DEV_CHECK_MODE="${DEV_CHECK_MODE:-required}"
    export DEV_CHECK_REPORT_ROOT="$OUT/dev_check"
}
