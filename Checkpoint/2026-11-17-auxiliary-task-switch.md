# CPT 独立辅助任务开关

工作区基线：master `518a578`，保留此前未提交的 Hierarchical-12L 改动。本轮只小改训练插件、Phase2-A/B 脚本参数及相关说明/测试，没有修改模型结构、数据、CLM/辅助损失公式或分母，无提交/合并/推送。

## 行为

- 新增 ENABLE_AUXILIARY_TASKS，默认 true，仅 Phase2-A/B 可以开启；Phase1 和 Char CPT 始终不启用辅助任务。只接受 true/false（忽略大小写和首尾空白）。
- false 时模板 return_aux_logits=false，纯 CLM 损失；辅助头在 optimizer/DDP 初始化前冻结，避免即使梯度为零仍因 weight decay 更新。音形 Embedding、PGCA、全部 context 编码器继续按原阶段冻结或训练。
- PGCA 开关按训练阶段检查，不再误用 auxiliary_enabled 判断，允许“PGCA 开启、辅助关闭”。无辅助头的完整模型在 false 下也可运行。
- 不产生辅助监督指标、不增加辅助 warmup Token；保留原有梯度组中的零值诊断。Checkpoint 的 cc_llm_state.json 新增 auxiliary_tasks_enabled。模型 pgca_aux_enabled 仍表示结构是否有头，两者不混同。
- 默认 true 的辅助加权损失和 Token 归一化保持原样。切换开关需要新的 optimizer 分组；从 MODEL 权重开始对照实验，不直接 resume 另一开关状态。

## 使用

```bash
ENABLE_AUXILIARY_TASKS=false bash scripts/train/cpt/phase2_a.sh
ENABLE_AUXILIARY_TASKS=false bash scripts/train/cpt/phase2_b.sh
```

现有总训练入口也继承该变量。Shared 和 Hierarchical 均适用，无需重建模型。

## 验证

测试环境沿用 `/tmp/cc-hierarchical-20261117/env`（Torch 2.15.0+cpu、Transformers 5.22.2、ms-swift 4.6.1、PEFT 0.20.0、Python 3.14.6）；证据在 `/tmp/cc-auxiliary-switch-20261117/`。仅测试子进程使用前轮的 fork 设置匹配正式 Python 3.12 Linux worker 语义，没有修改生产运行时。

- 19 项定向检查通过：阶段/开关默认值和非法值、三阶段开启/关闭的可训练性与真实优化器组、Shared/Hierarchical 有/无头的纯 CLM 等价性、辅助头 forward 禁止调用、辅助头无梯度/不在优化器中、context 仍有梯度、counter 不累计、默认开启的加权损失保持原公式。
- 真实关闭开关的 ms-swift CPU 链路通过：Phase1/Phase2-A/Phase2-B 各四步，验证 loss 有限；逐项比较保存后的辅助头及 counter 与初始值完全一致，三份 cc_llm_state.json 均 enabled=false/counter=0；辅助损失指标未产生，Phase2-A/B 三个 context 梯度均非零。
- 综合快速回归 63 passed、1 skipped、6 个 unittest 子测试通过；跳过的单个真实 Swift 慢用例已以上述 false 模式另行执行并通过。默认开启的真实三阶段 CPU 短跑在前轮架构测试已执行，当前默认开启另有数学/冻结/优化器回归，没有声明本轮重复执行了它。

- Shell、Ruff F/E9、Python AST 与 diff 空白检查通过。未执行 GPU 大模型训练或多卡对照，原有目录/数据/环境未重建。

2026-11-23 的实际运行对照见 [Phase2-B 单卡吞吐对照](2026-11-23-phase2b-throughput.md)。新下载的日志确认 Shared 本轮 A 记录了字音、字形辅助损失，B 的辅助头梯度为零且无辅助损失字段。当前 B 的辅助任务关闭只适用于 B。两轮配置与损失对比已纳入上述记录。
