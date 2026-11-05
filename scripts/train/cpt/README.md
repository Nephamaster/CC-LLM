# 继续预训练

阶段顺序：

```bash
bash scripts/train/cpt/phase1.sh
bash scripts/train/cpt/phase2_a.sh
bash scripts/train/cpt/phase2_b.sh
```

每个脚本顶部直接列出该阶段全部训练参数。Phase1 训练主干并冻结 PGCA；Phase2-A 训练音形分支和辅助头；Phase2-B 进行全参数联合训练。

`DATASET_NUM_PROC=1` 时，CPT 插件仅将编码预处理的 `num_proc` 转为 `None`，让编码在当前进程执行，避免额外的进程池数据传递。数据读取与字段转换仍使用原进程设置，编码 batch 保持原值；显式设置大于 1 时仍使用所选并发数。

默认数据：

```text
data/pretrain/phase1/{train,validation}
data/pretrain/phase2/{train,validation}
```

数据包中的 `train/*.tar.xz` 必须先按根目录 README 给出的命令原地解压；训练脚本不会自动解压或删除压缩包。训练缓存写入 `outputs/cache/<MODEL_NAME>/`，不同模型名称默认使用不同缓存。A/B 使用同一数据、token 预算、长度和随机种子，保证监督区间不重叠。

断点恢复：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N bash scripts/train/cpt/phase2_b.sh
```
