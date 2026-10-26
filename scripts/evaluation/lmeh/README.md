# 通用 benchmark

```bash
MODEL=/path/to/model MODEL_NAME=Qwen3-Base-PGCA-CPT \
BENCHMARK=all bash scripts/evaluation/lmeh/run.sh
```

可选任务：`mmlu`、`arc`、`squadv2`、`gsm8k`、`cmmlu`、`clue`。MMLU、CMMLU、GSM8K 使用 5-shot，其余使用 0-shot；ARC 仅运行 Challenge。

全部数据从 `data/evaluation/lmeh` 的本地 Parquet/CSV 加载。SQuAD v2 评分实现在仓库内，不下载在线 metric。GSM8K 最多生成 512 token，SQuAD v2 最多生成 256 token。

常用参数：

```bash
INFERENCE_BACKEND=hf|vllm|auto
TENSOR_PARALLEL_SIZE=1
BATCH_SIZE=auto
MAX_LENGTH=8192
LIMIT=10                 # 仅冒烟
```

每次运行自动创建新的 `vN` 结果目录。
