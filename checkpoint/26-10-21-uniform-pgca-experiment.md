# Uniform-Attention PGCA 最小补充实验

## 范围与决策

- 已通读 agent/observation/refactor-p0-ob.md，按第六节实现独立推理实验；没有修改模型源码、训练脚本、权重或 P1/P2。
- 仅将全部 PGCA 层实际用于 attention @ V 的权重替换为有效 slot 上的均匀分布。拼音和字形一起等分，不是两模态各 50%。Mask、Causal Feature Encoder、Value 和 O Projection 保持原路径。
- 使用临时 forward hook；正常退出和异常退出都清理。全无效 slot 的权重保持全零，仍使用模型原有的零输出 mask。

## 数据与比较

- 新入口：scripts/diagnostics/run_uniform_pgca_probe.sh；实现和说明在同目录 uniform_pgca_probe.py / UNIFORM_ATTENTION.md。
- 从原 P0 保存的输入和逐样本结果选择 on.phon_count > 0，不重新抽样或 tokenize，不加载完整训练缓存。
- 已只读核验本轮子集为 314 条、540,315 个有效目标；其 Normal NLL=1.976471630912693，Off NLL=2.2385985787177396。不能与原 512 条的汇总直接比较。
- 模型默认来自 P0 manifest（本轮 for_exp 副本），不是继续训练后的最新 checkpoint。原 Normal/Off 逐样本结果复用，仅新增一轮 Uniform 前向和少量首样本检查。
- 输出为独立的 outputs/experiments/rnd2/p0_uniform_pgca，包含 uniform_manifest.json、uniform_samples.jsonl、uniform_summary.json；保留原 P0 数据不动。
- 三个配对差：Uniform−Normal、Off−Normal、Off−Uniform；以有效 token 加权，按原 seed=42 做 2,000 次配对序列 bootstrap。置信区间包含零不自动解释为两条件等效。

## 必须保留的边界

- Uniform 同时改变模态份额、候选分配和 head-specific 分配，不能单独归因于多音字消歧。
- Uniform 仍使用上下文相关的特征记忆，不能当作静态特征或 Char-only 对照。
- 不计算或校验源码哈希，不依赖 Git；保留模型状态与原 P0 一致、固定输入、重复性和干预后状态不变检查。
- 单样本 Normal/Off 锚点需复现原结果，退出 hook 后 Normal 也需恢复；不混用不同权重/数值环境的旧基线。

## 验证状态

- 40 项相关本地 CPU 测试通过，包含 BF16、1/4/17 slots、无效 mask、padding、空 mask、Q/K 改动不影响 Uniform、V/O 数学参考、异常清理、统计分母与 CI、完整保存模型加载和重复运行。
- Shell 语法及参数透传检查通过；现有 P0 数据选集和基线只读核验通过。
- 尚未运行真实 1.7B GPU Uniform 条件，因此没有 Normal/Uniform/Off 的新实验结论。先由用户在空闲 GPU 上运行，再分析结果，不自动启动第二轮训练。
