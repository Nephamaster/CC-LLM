#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-$ROOT/.venv/train/bin/python}"
STAGES="${STAGES:-phase1,phase2_a,phase2_b,csc,cgec,ancient}"
FORCE="${FORCE:-false}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT/outputs/train}"
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
    local name="$1" output="$2" input="$3" refresh="$4" model_name="$5" script="$6"
    if [[ ${#requested_stages[@]} -eq 1 ]]; then
        output="${OUT:-$output}"
    fi
    stage_pointer="$output/next_stage_model.txt"
    stage_ran=false
    if [[ "$FORCE" != true && "$refresh" != true && -z "$input" && -z "${RESUME_FROM_CHECKPOINT:-}" && -s "$stage_pointer" ]]; then
        local checkpoint="$(<"$stage_pointer")"
        if [[ -d "$checkpoint" ]]; then
            echo "[train] skip completed stage $name: $checkpoint"
            return
        fi
        echo "[train] stage $name has a stale completion pointer; run it again"
    fi
    echo "[train] start $name input=${input:-previous stage} output=$output"
    env MODEL_NAME="$model_name" MODEL="$input" PYTHON_CMD="$PYTHON_CMD" \
        TRAIN_ROOT="$TRAIN_ROOT" CPT_ROOT="$CPT_ROOT" ALIGN_OUT="$ALIGN_ROOT" \
        ADAPTATION_OUT="$ADAPTATION_OUT" OUT="$output" bash "$script"
    [[ -s "$stage_pointer" ]] || { echo "Stage $name finished without publishing $stage_pointer" >&2; return 1; }
    stage_ran=true
    echo "[train] complete $name: $(<"$stage_pointer")"
}

ALIGN_ROOT="${ALIGN_OUT:-$TRAIN_ROOT/${MODEL_NAME}-ALIGN}"
CPT_ROOT="${CPT_ROOT:-$TRAIN_ROOT/${MODEL_NAME}-CPT}"
ADAPTATION_OUT="${ADAPTATION_OUT:-$CPT_ROOT/adaptation}"
FINETUNE_MODEL_NAME="${MODEL_NAME}-CPT"
align_updated=false
adaptation_updated=false
cpt_updated=false

if selected phase1; then
    run_stage phase1 "$ALIGN_ROOT" "${PHASE1_MODEL:-}" false "$MODEL_NAME" "$ROOT/scripts/train/cpt/phase1.sh"
    align_updated="$stage_ran"
fi
if selected phase2_a; then
    run_stage phase2_a "$ADAPTATION_OUT" "${PHASE2_A_MODEL:-}" "$align_updated" "$MODEL_NAME" "$ROOT/scripts/train/cpt/phase2_a.sh"
    adaptation_updated="$stage_ran"
fi
if selected phase2_b; then
    run_stage phase2_b "$CPT_ROOT/joint" "${PHASE2_B_MODEL:-}" "$adaptation_updated" "$MODEL_NAME" "$ROOT/scripts/train/cpt/phase2_b.sh"
    cpt_updated="$stage_ran"
    if [[ "$stage_ran" != true ]]; then
        mkdir -p -- "$CPT_ROOT"
        if [[ ! "$stage_pointer" -ef "$CPT_ROOT/next_stage_model.txt" ]]; then
            cp -- "$stage_pointer" "$CPT_ROOT/next_stage_model.txt"
        fi
    fi
fi
if selected csc; then
    run_stage csc "$TRAIN_ROOT/${FINETUNE_MODEL_NAME}-CSC" "${CSC_MODEL:-}" "$cpt_updated" "$FINETUNE_MODEL_NAME" "$ROOT/scripts/train/finetune/csc.sh"
fi
if selected cgec; then
    run_stage cgec "$TRAIN_ROOT/${FINETUNE_MODEL_NAME}-CGEC" "${CGEC_MODEL:-}" "$cpt_updated" "$FINETUNE_MODEL_NAME" "$ROOT/scripts/train/finetune/cgec.sh"
fi
if selected ancient; then
    run_stage ancient "$TRAIN_ROOT/${FINETUNE_MODEL_NAME}-ACT" "${ANCIENT_MODEL:-}" "$cpt_updated" "$FINETUNE_MODEL_NAME" "$ROOT/scripts/train/finetune/ancient.sh"
fi
