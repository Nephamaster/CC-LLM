#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

CONFIG="${PHASE1_CONFIG:-scripts/data_factory/phase1_config.json}"
python -m scripts.data_factory.build_phase1 "${ACTION}" --config "${CONFIG}" "$@"

