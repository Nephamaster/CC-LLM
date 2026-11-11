# 下游 LoRA

CSC、CGEC 和古汉语均使用 ms-swift all-linear LoRA，PGCA 分支参与 LoRA，字音/字形辅助损失关闭。

```bash
bash scripts/train/finetune/csc.sh
bash scripts/train/finetune/cgec.sh
bash scripts/train/finetune/ancient.sh
```

数据分别位于 `data/finetune/{csc,cgec,ancient}`。CSC 默认读取 `csc_mix_dedup.jsonl` 和 `cscd_dev.jsonl`；其余任务使用各目录下的 train/validation 或 train/dev。任务脚本顶部直接设置长度、学习率和 epoch；GPU、batch、LoRA 和优化器公共参数位于 `common.sh` 顶部。

默认读取完成的 CPT 指针。显式固定基座时同时传入名称：

```bash
MODEL=/path/to/checkpoint MODEL_NAME=Qwen3-Base-PGCA-CPT \
bash scripts/train/finetune/csc.sh
```

产物是 LoRA adapter，并在任务输出根目录发布 `next_stage_model.txt`。

集中冒烟会设置 `MAX_STEPS=2 SAVE_ONLY_MODEL=true`；正式微调默认仍按 epoch 训练并保存优化器状态。

单机训练默认通过 torchrun 自动分配通信端口；指定端口时在原命令前加 `MASTER_PORT=23457`。启动时打印模型、输出路径、world size、每卡 batch、梯度累积及有效 batch，并显示阶段日志文件路径。

每个版本目录的 `checkpoint_selection.json` 说明最终选择 best 还是 last。验证日志增加有效监督 token 总数；没有可用的有限 best 指标时，会明确记录回退到最后检查点的原因。原有 LoRA、辅助损失与数据编码并发设置保持原值。

## 固定 Dev 生成验收

三个任务、Base/Char/PGCA 三条路线共用同一检查逻辑：

- 每次保存 checkpoint，主进程按 Dev 文件顺序取每个任务前 1 条，使用实际训练模板检查答案结束符监督，并执行自由生成。ACT 分别检查 `c2m`、`m2c`、`punctuate`。结果保存到 `checkpoint-N/dev_check/vN/`；未通过只记录，不中断训练，不改变 loss 选模。
- 正式下游评估的 `STAGE=all|infer` 自动重新加载选中模型与已保存 adapter，每任务检查前 10 条。每个任务的第一条另做 LoRA 开关对照、完整前缀与逐 token KV Cache logits 对照、Cache 开关自由生成对照。PGCA 同时检查主干和音形 Cache 长度。
- 默认各任务正常停止率及 `<|im_end|>` 结束率均须至少 90%，长度截断、重复、异常输出的样本占比各不超过 10%。结束符未被监督、模板不匹配、adapter 未生效或 Cache 检查不通过，也会阻止正式下游推理。

每个检查目录有 `report.json`（阈值、各任务比例及对照结果）和 `samples.jsonl`（输入、参考、原始生成、token ID 和逐条标记）。合法罕见汉字只作提示，不能仅凭出现罕见字就判定输出损坏。检查不评价翻译正确性，也不能替代数据质量审查。

配置集中在 `dev_metrics.py`，可直接用以下环境变量覆盖，训练保存和评估使用同一套默认值：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `DEV_CHECK_SAVE_SAMPLES` | 1 | 每次保存时每任务样本数 |
| `DEV_CHECK_SAMPLES` | 10 | 完整验收时每任务样本数 |
| `DEV_CHECK_MAX_NEW_TOKENS` | 512 | 生成上限；单条预算按参考长度计算，最多取此值 |
| `DEV_CHECK_MIN_STOP_RATE` | 0.9 | 正常停止及 im_end 结束的最低比例 |
| `DEV_CHECK_MAX_BAD_RATE` | 0.1 | 截断、重复、异常各自的最高比例 |
| `DEV_CHECK_CACHE_STEPS` | 16 | Cache 数值对照的参考续写 token 上限 |
| `DEV_CHECK_CACHE_ATOL` / `DEV_CHECK_CACHE_RTOL` | auto | logits 对照绝对/相对容差；实际值写入报告 |
| `DEV_CHECK_MODE` | required | 完整验收模式；`report` 只豁免生成质量门槛，结构错误仍失败 |
| `DEV_CHECK_DEV` | 自动读取 | 覆盖选中 checkpoint 记录的 Dev 路径 |
| `DEV_CHECK_DEVICE` | cuda:0 | 完整验收设备，按 `CUDA_VISIBLE_DEVICES` 映射后的编号 |

每条生成预算为 `min(上限, max(64, 2×参考token数+16))`，参考答案不进入自由生成输入。先确认 `budget_covers_reference` 再解释截断；过长样本可提高上限。重复检测使用重复 4-gram 和尾部循环；异常检测覆盖空回答、替换字符、非法控制字符和泄漏的 ChatML 标记，均为启发式检查。

自动 logits 容差：FP32 `atol=1e-4, rtol=1e-4`，FP16 `0.02, 0.005`，BF16 `0.1, 0.02`。Cache 两路生成可因近似并列 logits 出现不同文字；要求数值在容差内、停止/截断/重复/异常标记一致，不要求全文逐字相同。不一致应调查，不能仅据此断言 Cache 实现错误。验收使用单序列、无 padding 的 HF 路径，不覆盖 vLLM 或批量 padding 的数值一致性。

也可提前独立验收，无需开始整套评估：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/evaluation/bin/python -B -m scripts.train.finetune.dev_check \
  --task ancient --model /path/to/full-model --adapter /path/to/adapter-checkpoint \
  --output outputs/evaluation/dev_check
```

CSC/CGEC 将 `--task` 改为 `csc`/`cgec`。迁移训练目录后若原 Dev 路径不可用，加 `--dev /path/to/dev.jsonl`；古汉语 Dev 需自带 `channel` 或保留同目录同名 `.metadata.jsonl`。
