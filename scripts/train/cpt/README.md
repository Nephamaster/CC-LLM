# 继续预训练

阶段顺序：

```bash
bash scripts/train/cpt/phase1.sh
bash scripts/train/cpt/phase2_a.sh
bash scripts/train/cpt/phase2_b.sh
```

每个脚本顶部直接列出该阶段全部训练参数。Phase1 训练主干并冻结 PGCA；Phase2-A 训练音形分支和辅助头；Phase2-B 进行全参数联合训练。

Char 对照路线：`MODEL_VARIANT=char MODEL_NAME=B-Char PHASE1_MODEL=/path/to/B-Char STAGES=phase1,cpt bash scripts/train/run.sh`。Phase1 使用相同语料训练标准 Char 主干；随后 `char.sh` 自动读取 Char-ALIGN 指针，以同一 Phase2 共享池进行全参数 CLM，默认监督上限 10B，不划分只训练分支的 A 段，也不启用辅助任务。`CPT_MODEL` 可覆盖 Char 的 Phase2 输入。标准 Qwen3 不能使用 PGCA 专属的多任务 loss，因此复用插件中的 CLM 分支与既有零目标过滤、token 归一化、检查点发布逻辑。

`DATASET_NUM_PROC=1` 时，CPT 插件仅将编码预处理的 `num_proc` 转为 `None`，让编码在当前进程执行，避免额外的进程池数据传递。数据读取与字段转换仍使用原进程设置，编码 batch 保持原值；显式设置大于 1 时仍使用所选并发数。

默认数据：

```text
data/pretrain/phase1/{train,validation}
data/pretrain/phase2/{train,validation}
```

数据包中的 `train/*.tar.xz` 必须先按根目录 README 给出的命令原地解压；训练脚本不会自动解压或删除压缩包。训练缓存写入 `outputs/cache/<MODEL_NAME>/`，不同模型名称默认使用不同缓存。A/B 使用同一数据、token 预算、长度和随机种子，保证监督区间不重叠。

正式默认参数与 `experiment.xlsx` 的参数一览一致：Phase1、Char CPT 每卡 batch 8 / 累积 4；PGCA-A 每卡 batch 8 / 累积 1；PGCA-B 每卡 batch 16 / 累积 2；编码进程均为 32。8 卡时有效 batch 分别为 256、256、64、256。GPU 仍用 `CUDA_VISIBLE_DEVICES` 选择，进程数自动按可见卡数推导。

A 的默认 prepared 目录为 `phase2_<长度>_adaptation`，B 为 `phase2_<长度>_joint`，分别满足 batch 8/16 的缓存几何。两者默认复用既有的 `phase2_<长度>.raw` 原始编码；显式指定 `CACHE_DIR` 时仍默认使用其对应的 `.raw`。后续更改 batch 时，prepared 视图仍需按原有校验规则重新准备。

原生编码导出和后续准备可分开恢复：`.raw/train`、`.raw/val` 已完成时直接复用；准备在临时目录完成后才发布，旧的未完成准备目录保留为 `.incomplete-*`。原生导出不完整时会提示具体 split，原文件保留；可设置 `RAW_CACHE` 指向新的导出目录。卡数或每卡 batch 改变后的缓存几何校验保持原逻辑。

所有参数仍在阶段脚本顶部。共享的 `scripts/train/runtime.py` 只负责启动、阶段日志和检查点发布；单机默认自动分配通信端口，显式 `MASTER_PORT` 优先。启动时显示日志位置，训练版本目录的 `checkpoint_selection.json` 记录 best/last 及原因，Phase2-A 仍选择 last。

集中冒烟入口见 `scripts/smoke/run.sh`。CPT 支持 `MAX_STEPS`（默认 -1，仍按单遍数据训练）与 `SAVE_ONLY_MODEL`（默认 false）；冒烟用少步、只存模型来减少运行时间，后者不保存恢复所需的优化器状态。

断点恢复：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N bash scripts/train/cpt/phase2_b.sh
```
