# 通用 benchmark

```bash
MODEL=/path/to/model MODEL_NAME=Qwen3-Base-PGCA-CPT \
BENCHMARK=all bash scripts/evaluation/lmeh/run.sh
```

可选任务：`mmlu`、`arc`、`squadv2`、`gsm8k`、`cmmlu`、`clue`。MMLU、CMMLU、GSM8K 使用 5-shot，其余使用 0-shot；ARC 仅运行 Challenge。

任务数据和 SQuAD v2 评分器通过 `HF_ENDPOINT=https://hf-mirror.com` 获取，下载后复用 Hugging Face 缓存。代码不固定数据 revision，也不要求在项目数据包中保存通用 benchmark。GSM8K 最多生成 512 token，SQuAD v2 最多生成 256 token。

只检查任务数据能否通过镜像加载：

```bash
STAGE=check BENCHMARK=all bash scripts/evaluation/lmeh/run.sh
```

常用参数：

```bash
INFERENCE_BACKEND=hf|vllm|auto
TENSOR_PARALLEL_SIZE=1
BATCH_SIZE=auto
MAX_LENGTH=8192
LIMIT=10                 # 仅冒烟
```

每次运行自动创建新的 `vN` 结果目录。
