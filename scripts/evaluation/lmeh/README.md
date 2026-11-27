# 通用 benchmark

```bash
MODEL=/path/to/merged-ins MODEL_NAME=Qwen3-Base-PGCA-CPT-INS \
EVAL_MODE=instruct BENCHMARK=all bash scripts/evaluation/lmeh/run.sh
```

可选任务：`mmlu`、`arc`、`squadv2`、`gsm8k`、`cmmlu`、`clue`。MMLU、CMMLU、GSM8K 使用 5-shot，其余使用 0-shot；ARC 仅运行 Challenge。

默认 `EVAL_MODE=instruct`：所有通用主结果使用训练后的 Chat 模型，通过模型保存的 `args.json` 恢复 Swift 非思考模板。Few-shot 固定为 user/assistant 多轮消息，不使用 tokenizer 的默认思考开关猜测格式。每个 Prompt 都核对渲染后的 Token IDs；前 32 个写入 `chat_prompts.jsonl`，harness 的逐样本结果另保留实际上下文。生成统一增加 im_end/endoftext 停止串，任务自身的停止串和生成上限继续保留；manifest 记录模式、消息组织和实际检查数量。

原始 Base/CPT 的补充对照显式设置 `EVAL_MODE=base`，此时 `apply_chat_template=false`、`fewshot_as_multiturn=false`；不能把这套结果与 Instruct 协议混为同一组。两种模式均保持相同题目、shot 数和随机种子。

任务数据和 SQuAD v2 评分器通过 `HF_ENDPOINT=https://hf-mirror.com` 获取，下载后复用 Hugging Face 缓存。代码不固定数据 revision，也不要求在项目数据包中保存通用 benchmark。GSM8K 最多生成 512 token，SQuAD v2 最多生成 256 token。

CMMLU 通过 HF 缓存获取官方 `cmmlu_v1_0_1.zip`，用内置 CSV 加载器直接读取各学科的 Dev/Test，不执行已被新版 datasets 移除支持的 `cmmlu.py`。保持 67 学科、Dev 5-shot、原提示与指标；忽略 CSV 索引列，并保持题目、四个选项和答案均为字符串。指令数据去污染也复用这套任务配置，无需降级 datasets 或更换数据集来源。

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

## 生成策略、seed 与配置记录

所有自由生成请求在共用后端入口明确使用 greedy、单返回序列。SQuAD 的实际 `construct_requests` 也显式设置 `do_sample=false, temperature=0`，不依赖 Python 任务的汇总配置。预算、任务停止串及候选 loglikelihood 评分保持各自协议。

实验 seed 统一为42：evaluator/few-shot 与 vLLM 引擎、生成请求分别显式设置。manifest 的 `seeds` 区分这些层次，`actual_model.engine_seed` 记录 harness 后端接收的引擎值。HF 的 RNG 由 evaluator 设置；不将 seed 作为不受支持的 HF generate 参数传入。

每轮新增 `generation_config.jsonl`，按真实生成 dispatch 追加关键生效参数。HF 在长度及停止条件解析后捕获；vLLM 在合入引擎默认值后捕获，并同时保留传入的采样参数。日志不复制 Prompt 或完整训练 args。`phase=dispatch_end` 的 `effective_config_observed` 说明本次是否实际捕获，异常只记录类型，钩子会恢复。

manifest 的 `generation_protocol` 标识解码协议，`generation_trace` 给出日志路径和本进程捕获覆盖。候选概率任务没有自由生成时，`generation_calls=0, all_calls_observed=null`；未捕获时不会用请求值冒充生效值。HF 最终温度字段为1但 `do_sample=false` 仍表示 greedy。
