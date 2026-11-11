# 通用 benchmark

```bash
MODEL=/path/to/merged-ins MODEL_NAME=Qwen3-Base-PGCA-CPT-INS \
EVAL_MODE=instruct BENCHMARK=all bash scripts/evaluation/lmeh/run.sh
```

可选任务：`mmlu`、`arc`、`squadv2`、`gsm8k`、`cmmlu`、`clue`。MMLU、CMMLU、GSM8K 使用 5-shot，其余使用 0-shot；ARC 仅运行 Challenge。

默认 `EVAL_MODE=instruct`：所有通用主结果使用训练后的 Chat 模型，通过模型保存的 `args.json` 恢复 Swift 非思考模板。Few-shot 固定为 user/assistant 多轮消息，不使用 tokenizer 的默认思考开关猜测格式。每个 Prompt 都核对渲染后的 Token IDs；前 32 个写入 `chat_prompts.jsonl`，harness 的逐样本结果另保留实际上下文。生成统一增加 im_end/endoftext 停止串，任务自身的停止串和生成上限继续保留；manifest 记录模式、消息组织和实际检查数量。

原始 Base/CPT 的补充对照显式设置 `EVAL_MODE=base`，此时 `apply_chat_template=false`、`fewshot_as_multiturn=false`；不能把这套结果与 Instruct 协议混为同一组。两种模式均保持相同题目、shot 数和随机种子。

任务数据和 SQuAD v2 评分器通过 `HF_ENDPOINT=https://hf-mirror.com` 获取，下载后复用 Hugging Face 缓存。代码不固定数据 revision，也不要求在项目数据包中保存通用 benchmark。GSM8K 最多生成 512 token，SQuAD v2 最多生成 256 token。

只检查任务数据能否通过镜像加载：

```bash
STAGE=check BENCHMARK=all bash scripts/evaluation/lmeh/run.sh
```

常用参数：

```bash
INFERENCE_BACKEND=hf|vllm|auto
TENSOR_PARALLEL_SIZE=4
BATCH_SIZE=32
MAX_LENGTH=8192
LIMIT=10                 # 仅冒烟
```

每次运行自动创建新的 `vN` 结果目录。

单卡 vLLM 评估需设置 `CUDA_VISIBLE_DEVICES=<索引> TENSOR_PARALLEL_SIZE=1`；HF 后端不使用该 TP 参数。
