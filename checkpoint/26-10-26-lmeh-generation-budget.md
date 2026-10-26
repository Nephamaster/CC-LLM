# lm-evaluation-harness 生成预算修复

## 问题

评估检查点的 `generation_config.json` 带有 `max_new_tokens: 2048`。HFLM 虽为每个请求传入任务相关的 `max_length`，但 Transformers 优先采用 `max_new_tokens`，导致 GSM8K/SQuAD 请求可能逐题生成 2048 token，并反复输出两种长度参数同时存在的警告。

## 修复

- `StrictHFLM` 加载模型后清除内存中的检查点级 `max_new_tokens`，不修改模型文件。
- GSM8K 明确使用 512-token 预算，SQuAD v2 明确使用 256-token 预算。
- manifest 记录被忽略的检查点级配置值，任务预算记录在 `protocol`。

该修复只约束生成长度，不改变题目、few-shot、评分协议或模型权重。吞吐仍由 `BATCH_SIZE` 控制。
