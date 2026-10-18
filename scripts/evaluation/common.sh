#!/usr/bin/env bash

# The calling task supplies its own environment directory and evaluation protocol.
configure_evaluation() {
    local task_dir="$1" benchmark="$2" protocol="${3:-}"
    local component
    ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    PYTHON="${PYTHON:-$task_dir/env/bin/python}"
    export PYTHONDONTWRITEBYTECODE=1
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
    cd "$ROOT"
}
