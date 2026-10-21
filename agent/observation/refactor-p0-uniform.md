# CC-LLM：Uniform-Attention PGCA 实验分析

## 一、核心结论：Normal 明显优于 Uniform，且两者均优于 Off

这次实验得到了很有价值的结果。PGCA 的收益并非仅来自把音形 Value 作为额外残差注入；模型学到的 Attention 权重分配本身也具有明显的预测价值。

| 指标                 | Normal PGCA | Uniform PGCA | PGCA Off |
| ------------------ | ----------- | ------------ | -------- |
| CLM NLL ↓          | 1.97647     | 2.05444      | 2.23860  |
| 相对 Normal 的 NLL 恶化 | —           | +0.07796     | +0.26213 |
| 评估序列数              | 314         | 314          | 314      |
| 有效监督 Token         | 540,315     | 540,315      | 540,315  |

不同 PGCA Attention 策略的 CLM NLL

同一 checkpoint、相同 314 条序列；越低越好

00.61.21.82.4NormalUniformOff

### 配对统计结果

| 比较                | CLM NLL 差值 | 95% Bootstrap CI   | 有利于较优方案的序列 |
| ----------------- | ---------- | ------------------ | ---------- |
| Normal vs Uniform | 0.07796    | [0.06684, 0.09222] | 313 / 314  |
| Uniform vs Off    | 0.18416    | [0.17188, 0.19583] | 305 / 314  |
| Normal vs Off     | 0.26213    | [0.24631, 0.27837] | 313 / 314  |

差值均按“较差方案 NLL 减去较优方案 NLL”计算，正数表示后者更好。三组区间均不跨零，且结果覆盖了绝大多数样本。

与此前 P0 的 512 条总体评估不同，本次只筛选其中 314 条存在有效字音监督的序列，因此不能将这次的 NLL 直接与上轮 512 条样本的总体 NLL 比较。

目前可以确认三个层次的事实：

1. Normal 优于 Off：完整 PGCA 对这个已训练 checkpoint 有显著作用。
2. Uniform 优于 Off：即使不使用学习到的 Attention 分配，音形 Value 通道仍然有价值。
3. Normal 优于 Uniform：学习到的 Attention 分配带来了额外的预测收益。

第三项正是此前 P0 没有回答、这次实验成功补足的问题。

但它尚不能证明 PGCA 学会了准确的多音字语境消歧，也不能证明 PGCA 与无 PGCA 模型在相同训练预算下的最终优劣。

## 二、逐样本分析：结果并非少数异常样本驱动

我重新读取了全部 314 行 `uniform_samples.jsonl`，计算了 Token Accuracy、PPL、不同长度区间的 NLL 差异和异常样本。

### 1. 三种条件下的预测准确率

| 指标               | Normal  | Uniform | Off     |
| ---------------- | ------- | ------- | ------- |
| PPL ↓            | 7.217   | 7.802   | 9.380   |
| Token Accuracy ↑ | 58.09%  | 56.68%  | 53.04%  |
| 正确预测 Token 数     | 313,849 | 306,257 | 286,564 |

Normal 相比 Uniform：

- Token Accuracy 提升 1.41 个百分点；
- PPL 降低约 7.50%；
- 根据逐样本结果重新进行 2,000 次配对 Bootstrap，准确率提升的 95% 区间约为 [1.23, 1.61] 个百分点。

因此，Normal 的优势不仅体现在连续概率指标 NLL 上，也转化成了实际的下一 Token 预测准确率。

### 2. 三条件的逐样本排序

Normal 优于 Uniform 优于 Off

305 / 314

Normal 优于 Off 优于 Uniform

8 / 314

Off 优于 Uniform 优于 Normal

1 / 314

唯一一条 Normal 不及 Uniform 的记录为 `row_index=17`，但 NLL 仅相差约 0.00138。

再按样本有效 Token 长度分成四组：

| 有效 Token 长度 | Uniform − Normal NLL |
| ----------- | -------------------- |
| 最短 25%      | +0.09985             |
| 25%–50%     | +0.07446             |
| 50%–75%     | +0.06430             |
| 最长 25%      | +0.08340             |

四组均有正收益。即使排除对总体差值贡献最大的前 10 条序列，Normal 的优势仍为约 0.06377 NLL。

结果具有较强的一致性，不能简单解释为随机波动或少量短文本异常。

## 三、与最新代码交叉核对：Uniform 干预有效，比较基本公平

我核对了当前 GitHub `main` 的 commit `6ac3951`，重点是：

- [`scripts/diagnostics/uniform_pgca_probe.py`](https://github.com/Nephamaster/CC-LLM/blob/6ac39515bce1bafa894b60c2e3cbde9bc8ce1ac0/scripts/diagnostics/uniform_pgca_probe.py?utm_source=chatgpt.com)
- [`src/pgca/attention.py`](https://github.com/Nephamaster/CC-LLM/blob/6ac39515bce1bafa894b60c2e3cbde9bc8ce1ac0/src/pgca/attention.py?utm_source=chatgpt.com)
- [`src/modeling_qwen3_pgca.py`](https://github.com/Nephamaster/CC-LLM/blob/6ac39515bce1bafa894b60c2e3cbde9bc8ce1ac0/src/modeling_qwen3_pgca.py?utm_source=chatgpt.com)

实际干预位置正确。

原始 PGCA 的计算为：

\\[ A=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}+M\right) \\]

然后：

\\[ F=\operatorname{OProj}(AV) \\]

Uniform 将 \\(A\\) 替换为：

\\[ A^{U}\_{t,h,s}= \begin{cases} 1/N_t,&s\text{ 是有效 slot}\\\ 0,&\text{其他情况} \end{cases} \\]

其中 \\(N_t\\) 是该位置有效 slot 总数。

代码通过 `register_forward_hook` 替换 PGCA dropout 模块的输出，因此替换后的权重确实参与后续 `attention @ V` 运算。

它没有修改： Feature Embedding、四层 Causal Feature Encoder、K/V 的数值、O Projection、PGCA 层数以及残差拓扑。

另外，检查项均通过：

| 检查                         | 结果                |
| -------------------------- | ----------------- |
| 与原 P0 的 Normal/Off 首样本结果一致 | 通过                |
| Uniform 首样本重复计算一致          | 通过                |
| 移除 hook 后 Normal 恢复        | 通过                |
| 六个 PGCA 层均执行干预             | 通过                |
| 模型状态前后 SHA256              | 一致                |
| 314 条完整样本                  | `status=complete` |

每层的 hook 记录为 315 次，是因为 314 条正式前向之后额外进行了 1 次重复性检查，并非漏算或重复计入评测。

因此，没有发现会推翻这次比较的明显实现错误。但 Bootstrap 单位是编码后的序列，同一原始文档的多个分片可能相关；置信区间没有专门处理这种相关性。

## 四、机制上的关键发现：Attention 有用，不等于读音消歧有效

这是需要结合此前 P0 结果一起理解的地方。

### 1. 现在可以区分两种贡献

从 Off 到 Uniform，NLL 改善：

\\[ 2.23860-2.05444=0.18416 \\]

从 Uniform 到 Normal，额外改善：

\\[ 2.05444-1.97647=0.07796 \\]

沿这条干预顺序，Normal 相比 Off 的总 NLL 差距可以分为：

Uniform 分支保留的收益

70.3%

Normal 学习权重的额外收益

29.7%

按 Off → Uniform → Normal 的 NLL 改善量划分；不是模型真实信息贡献的唯一或可加性分解。

这说明，不能再将 PGCA 简单理解为“学到的 Attention 没有作用，只是音形信息在提供额外残差”。

Attention 权重的选择确实重要。

### 2. 但 Uniform 同时改变了三个东西

当前 Uniform 是在所有有效拼音和字形 slots 上统一平均。它同时消除了：

- 不同 Attention Head 各自学习到的 slot 偏好；
- 拼音模态和字形模态之间的权重差异；
- 同一汉字不同拼音候选之间的权重差异。

因此，Normal 比 Uniform 更好，并不能直接判断收益来自其中哪一项。

结合此前 P0：

- PGCA 六层的平均拼音模态权重约为 72%–84%；
- 拼音候选归一化 Attention 熵接近 1；
- 多音字正确读音的平均条件权重接近均匀基线；
- 配对语境的正确切换能力仍较弱。

现在最合理的假设是：

> PGCA 学到了有价值的模态分配和／或 Head 专业化，但尚不能确认它学到了有效的上下文拼音消歧。

这属于根据两次实验综合提出的机制假设，不是已经证明的唯一解释。

特别是，许多汉字只有一个拼音候选时，根本不存在候选内部选择问题，但仍可通过调节“拼音 vs 字形”的相对份额改善预测。

### 3. Context Encoder 的独立收益仍然未知

Uniform 保留了四层 Causal Feature Encoder，因此：

Uniform 优于 Off，并不能被解释成“静态音形特征有效、上下文编码器没有用”。

这个实验没有关闭上下文编码器，也没有比较静态 PGCA。它只验证了在已训练的上下文化 Feature Memory 上，学习到的 Attention 权重比均匀权重更有效。

## 五、对第二轮 P1/P2 的具体决策

基于当前结果，我会进一步收敛方案，而不是增加新的结构改造。

| 项目                         | 最新建议                      | 理由                                 |
| -------------------------- | ------------------------- | ---------------------------------- |
| PGCA Cross-Attention       | 保留                        | Normal 显著优于 Uniform                |
| 4 层 Causal Feature Encoder | 保留                        | 尚无其有害或冗余的直接证据                      |
| 无 gate 并行残差                | 保留                        | 目前没有理由重新引入门控                       |
| 候选数量校正                     | 第二轮默认不启用                  | Uniform 不足以证明模态配比需要改成更均衡           |
| 固定残差缩放                     | 默认不启用                     | PGCA 的现有贡献为正，不应无证据削弱               |
| 辅助损失                       | 保留原任务，采用此前暂定的 `0.03/0.03` | 这是单独的损失平衡选择，本次 Uniform 结果不能证明该权重最优 |
| 扩大音形编码器                    | 不做                        | 当前主要瓶颈不是已证实的模型容量不足                 |

我尤其不建议在第二轮同时调整 PGCA 的 Softmax、Attention Head 结构和残差缩放。这样即使最终 CLUE／CSC 提高，也会很难判断改善来自哪一项。

### 唯一值得考虑的补充诊断：保持模态权重，只平均拼音候选

考虑到你希望减少实验成本，我建议最多再补一次纯前向对照，不开展额外训练。

新增 `Modality-Preserving Uniform`：

\\[ A'\_{p}= \frac{\sum\_{k\in\mathcal P}A_k}{|\mathcal P|}, \qquad A'\_g=A_g \\]

它在每个位置、每个 Head 内保持原本拼音模态的总权重、字形 slot 的权重，只把有效拼音候选内部的权重改为均匀。

| 结果            | 可以支持的判断                            |
| ------------- | ---------------------------------- |
| 与 Normal 几乎一致 | 当前候选内部的精细 Attention 分配对 CLM 增益可能较小 |
| 明显差于 Normal   | 候选内部的学习权重具有独立价值                    |
| 明显优于 Normal   | 候选内部的选择策略值得改进                      |

它比当前 Uniform 更有针对性，因为不会把“模态偏好”和“候选选择”两种机制混在一起。需要继续注意：即使 Normal 优于该条件，也不自动证明被选中的拼音就是语境中的正确读音。

如果第二轮准备立即启动，这项测试可以不作为开训前的硬性阻塞条件。现有 P0＋Uniform 证据已经足够支持“保留现有 PGCA 架构”。

## 六、最终研究结论

两轮实验合并后，我们现在能够对 PGCA 提出比此前更强、也更准确的结论：

第一，PGCA 不是无效分支。 在同一训练 checkpoint 上，旁路 PGCA 会显著恶化 CLM NLL 和 Token Accuracy。

第二，学到的 PGCA Attention 分配不是可有可无的。 Normal 比所有有效 slot 均匀分配的 Uniform 额外降低约 0.078 NLL，准确率提高约 1.41 个百分点。

第三，仍不能把这种收益归因于精确的多音字消歧。 现有多音字诊断与熵统计并未体现稳定的正确读音选择。

第四，PGCA 已训练 checkpoint 的功能依赖，不等价于相同计算预算下相较 Char-only 的最终训练收益。 是否真正提高通用能力和纠错能力，仍要看第二轮同口径 CLUE／CSC 与基线比较。

因此，我建议第二轮架构先保持稳定：主要改变 Batch、warm-up、Weight Decay 和辅助损失平衡，不急于修改 PGCA 结构。 这次 Uniform 实验支持继续使用学习型 Cross-Attention，也帮助我们把下一阶段的研究重点收敛到“模态选择与候选选择分别贡献了多少”，而不是重新质疑 PGCA 是否参与计算。