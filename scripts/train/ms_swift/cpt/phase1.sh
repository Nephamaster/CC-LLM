#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

MODEL="${MODEL:-$CC_ROOT/models/$MODEL_NAME}"
DATA_DIR="${DATA_DIR:-$CC_ROOT/data/corpus/runs/phase1/phase1-1b-default-c97f9387391d/final/ms_swift}"
OUT="${OUT:-$TRAIN_ROOT/${MODEL_NAME}-ALIGN}"
CACHE_DIR="${CACHE_DIR:-$CC_ROOT/data/cc_llm_multitask/phase1_2048}"
export CC_LLM_STAGE=phase1 PGCA_ENABLED=false

# Enforce the proposed budget; PHASE1_TOKENS=0 explicitly accepts the existing corpus once.
prepare_cache "$MODEL" "$DATA_DIR" "$CACHE_DIR" 2048 0 "${PHASE1_TOKENS:-1000000000}"
CC_RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}" \
run_training "$MODEL" "$CACHE_DIR/train" "$CACHE_DIR/val" "$OUT" 2048 1e-5 true \
    --freeze_parameters_ratio 0 \
    --freeze_parameters_regex '(^model\.layers\.[0-9]+\.pgca_attn\.|^model\.feature_memory_builder\.|^auxiliary_heads\.)'
