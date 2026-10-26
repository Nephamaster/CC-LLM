# 继续预训练

阶段顺序：

```bash
bash scripts/train/cpt/phase1.sh
bash scripts/train/cpt/phase2_a.sh
bash scripts/train/cpt/phase2_b.sh
```

每个脚本顶部直接列出该阶段全部训练参数。Phase1 训练主干并冻结 PGCA；Phase2-A 训练音形分支和辅助头；Phase2-B 进行全参数联合训练。

默认数据：

```text
data/pretrain/phase1/{train,validation}
data/pretrain/phase2/{train,validation}
```

数据缓存写入 `outputs/cache/`，不会改动下载的数据。A/B 使用同一数据、token 预算、长度和随机种子，保证监督区间不重叠。

断点恢复：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N bash scripts/train/cpt/phase2_b.sh
```
