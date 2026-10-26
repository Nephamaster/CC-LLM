#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-$ROOT/.venv/train/bin/python}"
STAGES="${STAGES:-phase1,phase2_a,phase2_b,csc,cgec,ancient}"
FORCE="${FORCE:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

[[ -x "$PYTHON_CMD" ]] || { echo "Run scripts/environment/run.sh first" >&2; exit 1; }
if [[ -z "${MODEL_NAME:-}" ]]; then
    [[ -s "$ROOT/models/model_name.txt" ]] || { echo "Set MODEL_NAME or run scripts/model_build/run.sh" >&2; exit 1; }
    IFS= read -r MODEL_NAME < "$ROOT/models/model_name.txt"
fi
[[ "$MODEL_NAME" != */* ]] || { echo "MODEL_NAME must be a name, not a path" >&2; exit 1; }
IFS=, read -r -a requested_stages <<< "$STAGES"
for stage in "${requested_stages[@]}"; do
    case "$stage" in
        phase1|phase2_a|phase2_b|csc|cgec|ancient) ;;
        *) echo "Unknown training stage: $stage" >&2; exit 2 ;;
    esac
done

selected() {
    [[ ",$STAGES," == *",$1,"* ]]
}

run_stage() {
    local name="$1" pointer="$2"
    shift 2
    if [[ "$FORCE" != true && -s "$pointer" ]]; then
        echo "[train] skip completed stage $name: $(<"$pointer")"
        return
    fi
    echo "[train] start $name"
    "$@"
    [[ -s "$pointer" ]] || { echo "Stage $name finished without publishing $pointer" >&2; return 1; }
    echo "[train] complete $name: $(<"$pointer")"
}

ALIGN_ROOT="$ROOT/outputs/train/${MODEL_NAME}-ALIGN"
CPT_ROOT="$ROOT/outputs/train/${MODEL_NAME}-CPT"
FINETUNE_MODEL_NAME="${MODEL_NAME}-CPT"

if selected phase1; then
    run_stage phase1 "$ALIGN_ROOT/next_stage_model.txt" \
        env MODEL_NAME="$MODEL_NAME" MODEL="${PHASE1_MODEL:-}" PYTHON_CMD="$PYTHON_CMD" \
        bash "$ROOT/scripts/train/cpt/phase1.sh"
fi
if selected phase2_a; then
    run_stage phase2_a "$CPT_ROOT/adaptation/next_stage_model.txt" \
        env MODEL_NAME="$MODEL_NAME" MODEL="${PHASE2_A_MODEL:-}" PYTHON_CMD="$PYTHON_CMD" \
        bash "$ROOT/scripts/train/cpt/phase2_a.sh"
fi
if selected phase2_b; then
    run_stage phase2_b "$CPT_ROOT/next_stage_model.txt" \
        env MODEL_NAME="$MODEL_NAME" MODEL="${PHASE2_B_MODEL:-}" PYTHON_CMD="$PYTHON_CMD" \
        bash "$ROOT/scripts/train/cpt/phase2_b.sh"
fi
if selected csc; then
    run_stage csc "$ROOT/outputs/train/${FINETUNE_MODEL_NAME}-CSC/next_stage_model.txt" \
        env MODEL_NAME="$FINETUNE_MODEL_NAME" MODEL="${CSC_MODEL:-}" PYTHON_CMD="$PYTHON_CMD" \
        bash "$ROOT/scripts/train/finetune/csc.sh"
fi
if selected cgec; then
    run_stage cgec "$ROOT/outputs/train/${FINETUNE_MODEL_NAME}-CGEC/next_stage_model.txt" \
        env MODEL_NAME="$FINETUNE_MODEL_NAME" MODEL="${CGEC_MODEL:-}" PYTHON_CMD="$PYTHON_CMD" \
        bash "$ROOT/scripts/train/finetune/cgec.sh"
fi
if selected ancient; then
    run_stage ancient "$ROOT/outputs/train/${FINETUNE_MODEL_NAME}-ACT/next_stage_model.txt" \
        env MODEL_NAME="$FINETUNE_MODEL_NAME" MODEL="${ANCIENT_MODEL:-}" PYTHON_CMD="$PYTHON_CMD" \
        bash "$ROOT/scripts/train/finetune/ancient.sh"
fi
