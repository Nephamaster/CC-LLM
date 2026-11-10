#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/evaluation/bin/python}"
BENCHMARKS="${BENCHMARKS:-ppl,lmeh,csc,fcgec_valid,fcgec_test,nacgec,ancient_c3,ancient_fuxi}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

[[ -x "$PYTHON" ]] || { echo "Run scripts/environment/run.sh first" >&2; exit 1; }
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT/outputs/train}"
MODEL_NAME="${MODEL_NAME:-${CPT_MODEL_NAME:-}}"
if [[ -z "$MODEL_NAME" ]]; then
    [[ -s "$ROOT/models/model_name.txt" ]] || { echo "Set MODEL_NAME or run model_build/run.sh" >&2; exit 1; }
    IFS= read -r MODEL_NAME < "$ROOT/models/model_name.txt"
    MODEL_NAME="${MODEL_NAME}-CPT"
fi
if [[ -z "${MODEL:-}" ]]; then
    pointer="$TRAIN_ROOT/$MODEL_NAME/next_stage_model.txt"
    if [[ -s "$pointer" ]]; then
        IFS= read -r MODEL < "$pointer"
    else
        MODEL="$ROOT/models/$MODEL_NAME"
    fi
fi
IFS=, read -r -a requested_benchmarks <<< "$BENCHMARKS"
for benchmark in "${requested_benchmarks[@]}"; do
    case "$benchmark" in
        ppl|lmeh|csc|fcgec_valid|fcgec_test|nacgec|ancient_c3|ancient_fuxi) ;;
        *) echo "Unknown evaluation target: $benchmark" >&2; exit 2 ;;
    esac
done

selected() {
    [[ ",$BENCHMARKS," == *",$1,"* ]]
}

adapter_path() {
    local suffix="$1"
    local pointer="$TRAIN_ROOT/${MODEL_NAME}-${suffix}/next_stage_model.txt"
    [[ -s "$pointer" ]] || { echo "Missing $suffix adapter pointer: $pointer" >&2; return 1; }
    cat "$pointer"
}

if selected ppl; then
    MODEL="$MODEL" MODEL_NAME="$MODEL_NAME" PYTHON="$PYTHON" \
        bash "$ROOT/scripts/evaluation/ppl/run.sh"
fi
if selected lmeh; then
    MODEL="$MODEL" MODEL_NAME="$MODEL_NAME" PYTHON="$PYTHON" BENCHMARK=all \
        bash "$ROOT/scripts/evaluation/lmeh/run.sh"
fi
if selected csc; then
    adapter="$(adapter_path CSC)"
    LORA="$adapter" MODEL="$MODEL" MODEL_NAME="${MODEL_NAME}-CSC" PYTHON="$PYTHON" \
        bash "$ROOT/scripts/evaluation/csc/run.sh"
fi
if selected fcgec_valid; then
    adapter="$(adapter_path CGEC)"
    LORA="$adapter" MODEL="$MODEL" MODEL_NAME="${MODEL_NAME}-CGEC" PYTHON="$PYTHON" \
        DATASET=fcgec SPLIT=valid bash "$ROOT/scripts/evaluation/cgec/run.sh"
fi
if selected fcgec_test; then
    adapter="$(adapter_path CGEC)"
    LORA="$adapter" MODEL="$MODEL" MODEL_NAME="${MODEL_NAME}-CGEC" PYTHON="$PYTHON" \
        DATASET=fcgec SPLIT=test bash "$ROOT/scripts/evaluation/cgec/run.sh"
fi
if selected nacgec; then
    adapter="$(adapter_path CGEC)"
    LORA="$adapter" MODEL="$MODEL" MODEL_NAME="${MODEL_NAME}-CGEC" PYTHON="$PYTHON" \
        DATASET=nacgec SPLIT=test bash "$ROOT/scripts/evaluation/cgec/run.sh"
fi
if selected ancient_c3; then
    adapter="$(adapter_path ACT)"
    LORA="$adapter" MODEL="$MODEL" MODEL_NAME="${MODEL_NAME}-ACT" PYTHON="$PYTHON" \
        DATASET=c3 bash "$ROOT/scripts/evaluation/ancient/run.sh"
fi
if selected ancient_fuxi; then
    adapter="$(adapter_path ACT)"
    LORA="$adapter" MODEL="$MODEL" MODEL_NAME="${MODEL_NAME}-ACT" PYTHON="$PYTHON" \
        DATASET=fuxi bash "$ROOT/scripts/evaluation/ancient/run.sh"
fi
