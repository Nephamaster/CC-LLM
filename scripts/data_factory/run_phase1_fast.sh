#!/usr/bin/env bash
set -euo pipefail

CONFIG="${PHASE1_CONFIG:-scripts/data_factory/phase1_config.json}"
WORKERS="${PHASE1_WORKERS:-32}"

python -m scripts.data_factory.build_phase1 cache \
  --config "${CONFIG}" \
  --workers "${WORKERS}"

python -m scripts.data_factory.build_phase1 calibrate \
  --config "${CONFIG}" \
  --workers "${WORKERS}"

python -m scripts.data_factory.build_phase1 fast_sample \
  --config "${CONFIG}" \
  --workers "${WORKERS}"
