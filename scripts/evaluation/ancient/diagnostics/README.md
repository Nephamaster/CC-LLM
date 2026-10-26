# ACT 诊断与数据抽查

所有古汉语专用诊断集中于本目录。原公共目录的 `diagnose.sh` / `diagnose_act.py` 已迁移，
没有旧入口兼容包装。公共推理实现仍为 `scripts/evaluation/inference.py`。

## P0：结束位置概率与真实 Cache On/Off

在服务器仓库根目录执行，默认使用 CPT checkpoint-12500 与 ACT checkpoint-1000：

```bash
CUDA_VISIBLE_DEVICES=2 MODE=followup MAX_NEW_TOKENS=96 \
  bash scripts/evaluation/ancient/diagnostics/run.sh
```

`CASES` 默认指向已完成诊断的
`outputs/diagnostics/act_checks_20261026-115931/selected_dev.json`。
该文件与当前 Dev 的 ID/messages 必须相符，固定复用六条，不重新抽样。
可覆盖 `MODEL`、`LORA`、`DEV`、`PYTHON`、`CASES`、`OUT`。
每次默认新建 `outputs/diagnostics/act_followup_<时间戳>/`，显式 `OUT` 必须尚不存在。

1. 用实际 `cc_llm_ancient` 与保存的 loss_scale 编码六条 Dev。
   正确因果位置是 EOS 之前：输入包含参考答案但不包含目标 EOS，取最后位置的 logits。
   在同一模型上关闭／开启 LoRA 各前向一次（均关闭 cache），记录 EOS 与 `𫟦` 的概率、logit、排名和 top-10。
   默认候选字符可通过 `CANDIDATE_TOKEN` 指定，ID 从实际 tokenizer 查询，并检查单 token 往返。
2. 三条 Dev（各任务第一条），固定 LoRA 开启、batch=1、无 padding，Cache On/Off 各生成最多 96 token。
   在最终 template.generate 调用同时覆盖 GenerationConfig 和 use_cache kwargs；不依赖引擎默认值。
   记录两次完全相同的 prompt IDs、实际 forward 输入长度／cache 长度、原始输出 token、停止原因。
   Cache Off 必须逐步计算增长的完整前缀且返回空 cache；On 必须出现单 token 输入及两套 cache 同步增长。

`report.json` 保存全部结果和汇总。`status=completed` 仅表示检查执行完毕；
`cache_control_verified` 才表示两种开关路径被实际观测。只有一步生成时，增量行为无法验证。
重复 4-gram 比例是诊断线索，不能当作翻译质量分数。正常停止但答案错误，也不能称作训练质量通过。

六条中的 Dev index 25 已确认参考错配，保留以复现既有结果，但其 teacher-forcing 上下文不能当作正确译文。
单条候选概率高于 EOS 不证明系统性 token 替代；需比较六条、两种权重状态、原始 token 和映射。

## P1：三任务 × Train/Dev × 100 条

只用 CPU，标准 Python 即可；原 final/raw 数据只读：

```bash
python -B -m scripts.evaluation.ancient.diagnostics.audit_data \
  --data data/downstream/ancient/train/final \
  --raw data/downstream/ancient/raw \
  --per-task 100 --seed 42 \
  --output "outputs/diagnostics/act_data_audit_$(date +%Y%m%d-%H%M%S)"
```

逐 task/split 随机抽取，共 600 条，不按长度／质量预筛。
全量核对 final 与 metadata 对齐，对抽样逐条回查 raw 来源；标点核对正文字符不变，翻译记录长度／复制风险。
输出 `samples.jsonl`（含来源证据）、`review.tsv`（语义审查表）、`report.json`（自动核验与抽样分母）。

必须区分来源一致与译文正确：错误配对可能已经存在于 raw。
`semantic_verdict` 默认为 pending，人工逐条填 ok/mismatch/uncertain 并说明理由后，才可估计语义错误率。
随机抽样与用户指出的已知错误应分别报告，不将刻意选中的坏例混入随机样本分母。
前置标点、长度比异常均为待审线索，不自动删除。不要用测试集修订训练答案。

## 已有 baseline 检查

```bash
CUDA_VISIBLE_DEVICES=2 MODE=baseline \
  bash scripts/evaluation/ancient/diagnostics/run.sh
```

baseline 全 Dev 检查最终 assistant EOS labels；三类各选 2 条，LoRA On/Off 比较 logits、生成、恢复；
三对左 padding 输入比较完整前向与逐 token 前向全词表 logits（默认只取前 128 tokens）。
默认 ATOL=0.05/RTOL=0.01，报告记录数值误差；BF16 超限需复核，不能自动判定需要重训。
输出到 `outputs/diagnostics/act_baseline_<时间戳>/`；`report.json`、`eos_labels.jsonl`、
`generations.jsonl`、`selected_dev.json` 保留证据。失败返回非零，异常尽量保留部分结果。

## 决策

- 同输入 Cache Off 正常停止、On 重复：优先调查推理路径；检查差异是否来自数值或缓存逻辑。
- 两种 cache 都重复且多个 EOS 位置候选概率偏高：调查权重与 EOS 学习，继续核验词表／权重行映射。
- 编码、映射、adapter、推理路径都确认正确后，ACT 仍系统性不终止：再考虑从 CPT checkpoint-12500 重训 ACT。
- 若数据错配广泛：先修数据、重新验收，再重训；调 LR 或延长训练不能修正参考答案。

这些工具不修改模型、训练数据、正式评估协议，不做源码或权重哈希。
本地 CPU 小模型验证工具行为，真实 1.7B＋LoRA 数值结论必须依据服务器报告。
