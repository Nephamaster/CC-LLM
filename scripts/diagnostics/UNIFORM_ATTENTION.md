# Uniform-Attention PGCA：最小补充实验

对应 [P0 观察报告第六节](../../agent/observation/refactor-p0-ob.md)。不训练、不改模型源码、不改权重，不运行 P1/P2，也不重跑原 P0 的梯度或多音字诊断。

## 运行

在 CC-LLM 根目录，确认 GPU 0 空闲后：

    CUDA_VISIBLE_DEVICES=0 bash scripts/diagnostics/run_uniform_pgca_probe.sh

默认读取已有的 outputs/experiments/rnd2/p0_pgca_probe。MODEL 留空时使用原 manifest 中的 checkpoint 路径，本轮是 for_exp；**不要换成继续训练后新保存的 checkpoint**。路径迁移可通过 MODEL 指定相同权重的副本。

输出是独立的 outputs/experiments/rnd2/p0_uniform_pgca，不改、不覆盖原 P0 结果。已有同名输出时需换 OUT，不会自动删除。参数均在 run_uniform_pgca_probe.sh 顶部，不依赖 common 脚本。

仅做服务器预检时，可运行：

    CUDA_VISIBLE_DEVICES=0 LIMIT=2 \
    OUT=outputs/experiments/rnd2/p0_uniform_preflight \
    bash scripts/diagnostics/run_uniform_pgca_probe.sh

这不需要重新导出/加载全量数据缓存或重新分词。使用已保存的 probe_inputs.jsonl、probe_samples.jsonl、probe_manifest.json 和 probe_summary.json；精度、Attention backend、种子和 bootstrap 次数沿用原 P0。正常执行只对选中的输入新增一轮 Uniform 前向，额外几个首样本前向用于检查恢复和重复性。

## 固定实验定义

- 从原 P0 验证序列中选择 on.phon_count > 0 的样本，保持原顺序、input_ids、labels、attention_mask、position_ids。**本轮恰好 314 条**；不按改善/恶化结果筛选，不重新抽样。LIMIT 只用于前缀预检。
- Normal/Off 逐样本结果复用原 P0；**汇总必须重算这 314 条的分母**，不能拿旧 512 条的 1.89425 / 2.07371 作基线。本轮该子集的 Normal/Off NLL 应约为 1.97647 / 2.23860。
- 对全部 PGCA 层（本轮 0-based 11–16），每个 token、每个 head 的权重替换为：

  a_m = mask_m / max(1, sum(mask))

  拼音和字形的所有有效 slots 一起均分。无效 slot 为 0；全部无效时整行权重为 0。不是拼音/字形各 50%，也不是只在拼音候选内均分。
- 用 forward pre-hook 读取实际 feature mask，在现有 PGCA dropout 的输出处替换 attention。V、O Projection、feature memory、mask 和残差相加路径不改；Q/K 仍可计算，但其分数不再决定 Uniform 权重。BF16 仍按原路径精度表示均匀权重，和为 1 可能有正常舍入误差。
- 固定 eval、use_cache=False、单卡 batch=1，无 optimizer、无梯度步骤。继承原损失计数方法，NLL 为有效目标上的 loss 总和 / token 总数。

## 配对差异与输出

三个条件输出 normal_nll、uniform_nll、off_nll。每项差异按原 P0 的 seed 和 2,000 次配对序列 bootstrap 重算 95% percentile CI：

| 字段 | 定义 | 为正表示 |
| --- | --- | --- |
| uniform_minus_normal | Uniform NLL − Normal NLL | 学到的 Attention 分配优于均匀分配 |
| off_minus_normal | Off NLL − Normal NLL | 正常分支优于关闭分支 |
| off_minus_uniform | Off NLL − Uniform NLL | 均匀分支优于关闭分支 |

输出目录中的 uniform_manifest.json 保存样本行号、模型状态及干预协议；uniform_samples.jsonl 保存三条件的逐样本损失和计数；uniform_summary.json 保存同子集汇总、差异、CI、胜负样本数和检查结果。

保障配对有效性：

- 检查保存的输入/labels 与原样本对应，当前内存模型状态与原 P0 相同。
- 开始前在首个选中样本上复算 Normal/Off，要求与原结果一致；若不一致则停止，不能把另一权重或不同数值路径悄悄当成配对条件。
- Uniform 首样本重复检查；退出干预后 Normal 恢复检查；全部持久化模型状态前后保持不变。
- 不计算或校验源码哈希，不依赖 Git，不重复扫描 checkpoint 源码或权重文件作哈希；这里只复用原 P0 的模型状态和数据指纹。

如果换了设备/框架，首样本的精确比较可能不一致，此时应恢复原 P0 的数值环境再跑，而不是取消检查或混用基线。LIMIT 未覆盖完整相关子集时，summary 标为 preflight_subset，不当作正式结果。

## 解释边界

Uniform **同时改变模态总份额和模态内部候选分配**，因此结果针对整体学习到的 Attention 分配，不单独证明“多音字选择”或“拼音/字形配比”的作用。

保留原 Causal Feature Encoder，所以 Uniform 仍可能利用上下文相关的 Value；不能把它等同于静态特征或 Char-only 基线。置信区间包含零也不等于已证明两条件等效，需结合效应大小与区间宽度判断。这里不自动判定 Normal≈Uniform，也不自动修改第二轮训练配置。
