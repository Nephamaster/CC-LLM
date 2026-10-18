python -m src.vocab.build_vocab all \
  --base-model-path /share/project/wuhaiming/data/models/Qwen3-1.7B-Base/ \
  --char-model-path models/Qwen3-1.7B-Base-Char \
  --hanzi-dir resources/hanzi \
  --features-dir models/Qwen3-1.7B-Base-Char/features \
  --unihan-path resources/unihan/Unihan.zip \
  --fallback-unihan-dir resources/raw/Unihan \
  --strict
#   --skip-migrate