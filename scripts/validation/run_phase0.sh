#!/usr/bin/env bash
set -euo pipefail

python -m scripts.validation.build_validation_data
python -m scripts.validation.validate_tokenizer
python -m scripts.validation.validate_forward
