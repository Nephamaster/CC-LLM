# 继续预训练

阶段顺序：

```bash
bash scripts/train/cpt/phase1.sh
bash scripts/train/cpt/phase2_a.sh
bash scripts/train/cpt/phase2_b.sh
```

每个脚本顶部直接列出该阶段全部训练参数。Phase1 训练主干并冻结 PGCA；Phase2-A 训练音形分支和辅助头；Phase2-B 进行全参数联合训练。

单阶段脚本的 `MODEL_NAME` 是输入模型名称。Phase1 例如传 `B-Char-PGCA`，输出 `B-Char-PGCA-Align`；Phase2-A/B 沿用同一个 CPT 输入名 `B-Char-PGCA-Align`，共同输出 `B-Char-PGCA-Align-CPT/{adaptation,joint}`。Char CPT 同样在输入名后追加 `-CPT`。总入口仍使用构建模型名，自动完成阶段名称衔接。

Phase2-A/B 的 `ENABLE_AUXILIARY_TASKS` 默认 true；设为 false 时只使用 CLM 损失，不执行辅助头前向或辅助监督统计，冻结辅助头且不累计辅助 warmup Token。PGCA、Feature Embedding、C1/C2/C3 继续按原阶段规则训练；无需修改模型结构或重建模型。

```bash
ENABLE_AUXILIARY_TASKS=false bash scripts/train/cpt/phase2_a.sh
ENABLE_AUXILIARY_TASKS=false bash scripts/train/cpt/phase2_b.sh
```

也可在 `scripts/train/run.sh` 前设置相同环境变量。Phase1、Char CPT 保持辅助任务关闭，INS/下游策略不变。关闭开关不同于把辅助权重设为 0；每个 checkpoint 的 cc_llm_state.json 记录实际开关状态。更改开关会改变可训练参数/优化器分组，切换实验时用 MODEL 从完整权重开始新运行，不直接 resume 另一开关状态的优化器。

`pgca_aux_enabled` 表示模型结构是否包含辅助头，新开关控制本次训练是否使用它们。false 不删除现有辅助头；原本不含辅助头的完整 PGCA 模型也可在 false 下训练。

Char 对照路线：`MODEL_VARIANT=char MODEL_NAME=B-Char PHASE1_MODEL=/path/to/B-Char STAGES=phase1,cpt bash scripts/train/run.sh`。Phase1 使用相同语料训练标准 Char 主干；随后 `char.sh` 自动读取 Char-Align 指针，以同一 Phase2 共享池进行全参数 CLM，默认监督上限 10B，不划分只训练分支的 A 段，也不启用辅助任务。`CPT_MODEL` 可覆盖 Char 的 Phase2 输入。标准 Qwen3 不能使用 PGCA 专属的多任务 loss，因此复用插件中的 CLM 分支与既有零目标过滤、token 归一化、检查点发布逻辑。

数据预处理默认 `DATASET_NUM_PROC=4`，编码单批默认 `ENCODING_BATCH_SIZE=32` 篇文档。两者都能在命令前覆盖；编码单批与 GPU 训练 batch 无关。长文档先编码再切分，一批可能扩展为许多训练片段，因此不沿用 Swift 的 1000 篇编码批次。减少单批文档数用于控制中间结果内存，不保证对所有机器都提速；并发应根据实际 CPU、内存和磁盘吞吐选择，不是越大越好。

`DATASET_NUM_PROC=1` 时，CPT 插件将编码预处理的 `num_proc` 转为 `None`，让编码在当前进程执行，避免额外的进程池数据传递。编码启动时会显示实际进程数和编码单批文档数。原生预处理缓存默认复用（包括 Phase2-A）；这些设置不改变样本内容、切分长度或训练 Token 预算。

默认数据：

```text
data/pretrain/phase1/{train,validation}
data/pretrain/phase2/{train,validation}
```

数据包中的 `train/*.tar.xz` 必须先按根目录 README 给出的命令原地解压；训练脚本不会自动解压或删除压缩包。训练缓存写入 `outputs/cache/<MODEL_NAME>/`，不同模型名称默认使用不同缓存。A/B 使用同一数据、token 预算、长度和随机种子，保证监督区间不重叠。

Phase1、Char CPT 默认每卡 batch 8、累积 4。PGCA-A 默认每卡 batch 8、累积 1。PGCA-B 默认每卡 batch 1、目标有效 batch 256，按实际进程数向上取整计算梯度累积步数。两卡累积 128，八卡累积 32，显式传入 `GRAD_ACCUMULATION` 可覆盖。PGCA-B 的 warmup 默认 1%，总更新步数由单遍训练数据量和有效 batch 决定。GPU 用 `CUDA_VISIBLE_DEVICES` 选择，数据预处理进程数默认 4。

A 的默认 prepared 目录为 `phase2_<长度>_adaptation`，B 为 `phase2_<长度>_joint`，各自按卡数和每卡 batch 补齐。两者默认复用既有的 `phase2_<长度>.raw` 原始编码；显式指定 `CACHE_DIR` 时仍默认使用其对应的 `.raw`。后续更改卡数或每卡 batch 时，脚本自动从原始编码重建 prepared 视图，无需删除缓存或重新编码；真实样本的顺序、监督 Token 区间保持不变，仅重新计算补齐样本数。只改梯度累积可直接复用同一 prepared 视图。

原生编码导出和后续准备可分开恢复：`.raw/train`、`.raw/val` 已完成时直接复用；准备在临时目录完成后才发布，旧的未完成准备目录保留为 `.incomplete-*`。自动重建失败时保留已有 prepared 缓存，成功后替换并清理旧视图。原生导出不完整时会提示具体 split，原文件保留；可设置 `RAW_CACHE` 指向新的导出目录。长度、随机种子或 Token 预算变化仍需使用对应的新缓存目录。

所有参数仍在阶段脚本顶部。共享的 `scripts/train/runtime.py` 只负责启动、阶段日志和检查点发布；单机默认自动分配通信端口，显式 `MASTER_PORT` 优先。启动时显示日志位置，训练版本目录的 `checkpoint_selection.json` 记录 best/last 及原因，Phase2-A 仍选择 last。

集中冒烟入口见 `scripts/smoke/run.sh`。CPT 支持 `MAX_STEPS`（默认 -1，仍按单遍数据训练）与 `SAVE_ONLY_MODEL`（默认 false）；冒烟用少步、只存模型来减少运行时间，后者不保存恢复所需的优化器状态。

断点恢复：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N bash scripts/train/cpt/phase2_b.sh
```
