# PPL

```bash
MODEL=/path/to/full-checkpoint MODEL_NAME=Qwen3-Base-PGCA-CPT \
bash scripts/evaluation/ppl/run.sh
```

默认读取 `data/evaluation/ppl/*.jsonl`。每行必须包含字符串字段 `text`，可选 `id` 和与文件名一致的 `domain`。

PPL 使用原始文本、无 chat template、无 BOS/EOS 拼接，按文档独立滑窗。不同 tokenizer 的 token PPL 不直接横比，应同时查看字符和 UTF-8 字节归一化结果。
