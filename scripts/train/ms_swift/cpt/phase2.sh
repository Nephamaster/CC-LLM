#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

ALIGN_OUT="${ALIGN_OUT:-$TRAIN_ROOT/${MODEL_NAME}-ALIGN}"
if [[ -z "${MODEL:-}" ]]; then
    [[ -f "$ALIGN_OUT/next_stage_model.txt" ]] || { echo "Set MODEL to the Phase1 checkpoint" >&2; exit 1; }
    IFS= read -r MODEL < "$ALIGN_OUT/next_stage_model.txt"
fi
DATA_DIR="${DATA_DIR:-$CC_ROOT/data/corpus/runs/phase2/phase2-10b-default-6d14a3c4e453/final/ms_swift}"
OUT="${OUT:-$TRAIN_ROOT/${MODEL_NAME}-CPT}"
CACHE_DIR="${CACHE_DIR:-$CC_ROOT/data/cc_llm_multitask/phase2_4096}"
RUN_STAGE="${RUN_STAGE:-all}"
[[ "$RUN_STAGE" == all || "$RUN_STAGE" == a || "$RUN_STAGE" == b ]] || { echo "RUN_STAGE must be all/a/b" >&2; exit 1; }
export CC_LLM_STAGE=phase2a PGCA_ENABLED=true
prepare_cache "$MODEL" "$DATA_DIR" "$CACHE_DIR" 4096 "${ADAPTATION_TOKENS:-100000000}" "${PHASE2_TOKENS:-10000000000}"

if [[ "$RUN_STAGE" != b ]]; then
    CC_RESUME_FROM_CHECKPOINT="${RESUME_A:-}" \
    run_training "$MODEL" "$CACHE_DIR/train_a" "$CACHE_DIR/val" "$OUT/adaptation" 4096 5e-5 false \
        --freeze_parameters_ratio 1 \
        --trainable_parameters_regex '(^model\.layers\.[0-9]+\.pgca_attn\.|^model\.feature_memory_builder\.|^auxiliary_heads\.)'
fi
[[ "$RUN_STAGE" != a ]] || exit 0
[[ -f "$OUT/adaptation/next_stage_model.txt" ]] || { echo "Phase2-A must finish before Phase2-B" >&2; exit 1; }
IFS= read -r MODEL < "$OUT/adaptation/next_stage_model.txt"
export CC_LLM_STAGE=phase2b
# A new Trainer/optimizer/scheduler, with the remaining disjoint targets, not a second full epoch.
CC_RESUME_FROM_CHECKPOINT="${RESUME_B:-}" \
run_training "$MODEL" "$CACHE_DIR/train_b" "$CACHE_DIR/val" "$OUT/joint" 4096 1e-5 true \
    --freeze_parameters_ratio 0
cp -- "$OUT/joint/next_stage_model.txt" "$OUT/next_stage_model.txt"
