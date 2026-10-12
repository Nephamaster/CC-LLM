**我的结论是：两阶段训练的总体思路合理，语义对齐确实起了作用，CPT 也有正常的学习迹象；但当前验证链路存在明确故障，因此还不能认定这轮训练已经充分验证有效，也不能据此判断 PGCA 的收益上限。**

目前最优先处理的是 **全部 `eval_loss = NaN`、PPL 的比较口径，以及 PGCA 是否实际产生有效贡献**。这些问题解决之后，才有依据决定是否增加上下文编码器或辅助训练目标。

我检查了两个压缩包中的 `args.json`、`logging.jsonl` 和截图，并审查了 [main 分支的 `ef68d4e` 快照](https://github.com/Nephamaster/CC-LLM/tree/ef68d4e53c82ea7908139fb9ec14c3ee0ef431eb)，同时对照了日志所用的 **ms-swift 4.4.2** 源码。以下会区分“日志已经证明的事实”和“仍需实际样本或 checkpoint 验证的推断”。

## 一、实际训练配置：阶段安排基本正确，CPT 尚未跑完

| 项目 | 语义对齐 ALIGN | 继续预训练 CPT |
|---|---|---|
| 起始模型 | `Qwen3-1.7B-Base-Char-PGCA` | ALIGN 的 `checkpoint-6289` |
| 参数训练范围 | 冻结全部参数，再开放 `model.embed_tokens`、`lm_head` | 配置为全参数训练 |
| 实际可训练参数记录 | **315.43M，占总参数的 17.03%** | 包中没有实际可训练参数统计 |
| 全局 batch | **8 卡 × 4 × 累积 4 = 128 条序列/步** | **8 卡 × 1 × 累积 8 = 64 条序列/步** |
| 峰值学习率 | **7e-5** | **1e-5** |
| 学习率计划 | Linear decay，3% warmup | Linear decay，3% warmup |
| 优化器及精度 | AdamW，β=(0.9, 0.95)，BF16 | 相同 |
| 序列处理 | `max_length=2048`、`split`、`packing=false` | 相同 |
| 日志中的完成进度 | **6,289 / 6,289 步，1 epoch** | **36,690 / 131,818 步，约 27.83%** |

这些设置来自两份日志中的实际运行参数，而不是此前讨论的计划配置。Qwen3-1.7B-Base-Char-PGCA-ALIGN Qwen3-1.7B-Base-Char-PGCA-ALIGN…

**ALIGN 的冻结安排符合“先适配新 tokenizer 与语义输入输出层”的目标。** 日志还记录了实际模型类 `Qwen3PGCAForCausalLM`，因此至少对齐阶段确实加载了你的自定义模型；不能因为参数里写着 `model_type=qwen3`，就断言 PGCA 模型没有加载。

学习率也没有明显离谱之处：对齐阶段只更新输入输出层，使用较大的学习率；CPT 解冻成熟主干后使用较小学习率，这个方向成立。**CPT 的 1e-5 比较保守，但是否偏低，当前没有可靠的验证 loss 支持判断。**

另外，有两个实际训练与名义计划之间的差别：

**第一，CPT 日志尚未覆盖完整训练。** 它只记录到计划步数的 27.8%，不能把它当成“完整 10B CPT 的最终结果”。日志没有记录实际累计输入 token 数，所以也不能直接说已经训练了多少 B tokens。

**第二，当前训练没有落实 Phase2 的 2K/4K/8K 混合上下文。** 仓库的 [phase2.yaml](https://github.com/Nephamaster/CC-LLM/blob/ef68d4e53c82ea7908139fb9ec14c3ee0ef431eb/scripts/data_factory/configs/phase2.yaml) 配置了 2K/4K/8K，但这次实际读取的是 `final/ms_swift` 下的 JSONL，进入 Trainer 后统一按最多 2048 tokens 切分。数据来源包含长文档，并不等于模型接受过 4K/8K 上下文训练。

## 二、最明确的问题：两阶段所有验证 loss 都是 NaN

这是本轮检查中最需要优先解决的异常。

| 阶段 | 验证次数 | 有限的 `eval_loss` | 验证 token accuracy |
|---|---:|---:|---|
| ALIGN | 13 次 | **0 次，全部 NaN** | 53.33% → 54.43% |
| CPT | 73 次 | **0 次，全部 NaN** | 56.94% → 58.76% |

ALIGN 最后的汇总还明确记录了：

```text
best_model_checkpoint = null
best_metric = Infinity
last_model_checkpoint = .../checkpoint-6289
```

因此，**对齐阶段“按验证 loss 选择最佳 checkpoint”的机制已经实际失效**。`checkpoint-6289` 是最后保存的模型，不是通过有效验证 loss 选出来的最佳模型。CPT 同样无法利用这 73 个 NaN 指标进行可靠选优。Qwen3-1.7B-Base-Char-PGCA-ALIGN Qwen3-1.7B-Base-Char-PGCA-ALIGN…

### 一个证据很强的原因：长度为 1 的分片没有有效预测目标

ALIGN 的最终统计中，训练集和验证集都出现了：

```text
min_length = 1
```

同时每卡验证 batch size 是 1。

我检查了 [ms-swift 4.4.2 的切分实现](https://github.com/modelscope/ms-swift/blob/v4.4.2/swift/template/base.py)：`truncation_strategy=split` 会保留最后的余数分片，而且把每个分片的第一个 label 设成 `-100`。直接抽取这段源码复现，可以得到：

| 编码后的序列长度 | 切分结果 | 各分片在 causal shift 后的有效 label 数 |
|---:|---|---|
| 2048 | 2048 | 2047 |
| **2049** | **2048 + 1** | **2047 + 0** |
| 2050 | 2048 + 2 | 2047 + 1 |
| **4097** | **2048 + 2048 + 1** | **2047 + 2047 + 0** |

也就是说，**原始文档并不需要很短，一篇正常长文档也可能被切出一个完全没有监督目标的尾段。**

你的 [模型 loss 实现](https://github.com/Nephamaster/CC-LLM/blob/ef68d4e53c82ea7908139fb9ec14c3ee0ef431eb/src/modeling_qwen3_pgca.py) 直接调用框架的 `loss_function`，没有自行处理零有效标签。标准交叉熵在没有有效目标时求均值，可以产生 NaN；一个这样的验证 batch 就足以污染整个验证 loss。

这也解释了为什么 **`eval_token_acc` 有正常值，而 `eval_loss` 始终 NaN**：ms-swift 的准确率按有效标签累加，空标签样本可以不贡献任何计数；loss 的均值聚合却可能被 NaN 污染。对应实现见 [accuracy 计算](https://github.com/modelscope/ms-swift/blob/v4.4.2/swift/metrics/acc.py) 和 [loss 计算](https://github.com/modelscope/ms-swift/blob/v4.4.2/swift/trainers/seq2seq_trainer.py)。

**这里可以确认的是：存在长度为 1 的实际样本，而且对应版本的切分逻辑确实会产生零有效标签分片。** 要把它最终确认为这次运行全部 NaN 的根因，还需要在真实验证集上捕获出错 batch，检查 `labels[..., 1:]` 的有效数量和 logits 是否有限。压缩包没有验证数据及模型权重，因此我没有完成这一步实际模型复现。

修复时的核心验收条件应是：

```python
有效监督数 = sum(label != -100 for label in labels[1:])
有效监督数 > 0
```

检查应放在**最终模板处理与切分之后**。只过滤原始空文本，无法处理长文档切出来的无效尾段。重新评估时，按全部有效目标累计 NLL 总和与目标数量，无效分片不进入分母。

## 三、训练曲线：有明确学习，但两个阶段都出现了平台

下面使用日志区间内的 loss 均值观察趋势，避免拿某一个 batch 的高低值判断收敛。这些是日志窗口均值，不是重新计算的整个语料 token 加权 NLL。

### ALIGN：前期恢复明显，后半程收益已经很小

| 步数区间 | 平均训练 loss |
|---|---:|
| 1–100 | **2.9217** |
| 201–500 | **2.2761** |
| 501–1,000 | **2.1653** |
| 1,001–2,000 | **2.1362** |
| 2,001–3,000 | **2.1248** |
| 3,001–6,280 | **约 2.123** |

验证 token accuracy 也呈现相同趋势：step 1000 为 **54.13%**，step 3000 为 **54.42%**，最后为 **54.43%**。Qwen3-1.7B-Base-Char-PGCA-ALIGN

这说明 **Embedding／LM Head 的适配确实在发生，主要收益集中在前 1000–2000 步，约 3000 步之后基本进入平台**。它不像训练没有生效，也不像优化过程持续发散。

但这个平台只能说明：**在当前数据、冻结范围和优化配置下，继续只训练这两个参数组的边际收益已经很小。** 它不能证明原模型能力已经完整恢复，也不能证明“1B 数据一定足够”。主干被冻结时，新 token 序列与原主干之间剩余的不匹配，本来就可能需要后续全参数训练处理。

### CPT：前期继续改善，20K 步后总体改善很慢

| 步数区间 | 平均训练 loss |
|---|---:|
| 1–1,000 | **2.0890** |
| 1,010–4,000 | **2.0206** |
| 4,010–10,000 | **1.9766** |
| 10,010–20,000 | **1.9649** |
| 20,010–30,000 | **1.9607** |
| 30,010–36,690 | **1.9609** |

CPT 的验证 token accuracy 从 step 500 的 **56.94%** 上升到 step 36500 的 **58.76%**，增加约 **1.83 个百分点**；后半段同样增长很小。这个数字是 next-token accuracy，不是下游 benchmark 分数。Qwen3-1.7B-Base-Char-PGCA-ALIGN…

**因此，CPT 有正常学习迹象，但已经出现明显的边际收益递减。** 现在无法判断继续跑剩余 72% 是否划算，因为最需要参考的验证 NLL 不可用。

还有两个解读边界：

- **ALIGN 的约 2.123 与 CPT 的约 1.961 不能直接作为两模型效果差。** 两阶段用了不同数据分布，需要把两个 checkpoint 放到同一个验证集上测。
- 可见训练 loss 和梯度没有持续发散，但两份配置都开启了 `logging_nan_inf_filter=true`。因此只能说“日志未显示持续数值崩溃”，不能保证每个 microbatch 从未产生异常。

## 四、PPL 截图：对齐效果明显，但不能直接衡量恢复了原模型多少能力

截图中的数据如下：

| 领域 | Base | Char | Char-Align | Char → Align 的 PPL 降幅 |
|---|---:|---:|---:|---:|
| Government | 3.1941 | 18.2181 | 5.9075 | **67.6%** |
| Movie | 11.2328 | 58.8474 | 16.9828 | **71.1%** |
| General | 4.5432 | 18.8631 | 7.2547 | **61.5%** |
| Game | 8.5195 | 40.2722 | 12.7068 | **68.4%** |
| Tech | 5.7692 | 25.7954 | 8.5490 | **66.9%** |
| Finance | 3.2086 | 11.8958 | 5.2470 | **55.9%** |
| 六领域算术平均 | **6.0779** | **28.9820** | **9.4413** | **67.4%** |

来源为你提供的截图。image

### 能得到的结论：Char → Align 的适配收益很明显

在 Char 与 Char-Align 使用同一个 tokenizer、同一评估设置的前提下，六个领域全部改善，PPL 降幅约 **56%–71%**。这与训练曲线中的快速下降相互支持。

所以，**当前证据支持“语义对齐有效”**。不能因为最终 benchmark 只提高 1–2 个百分点，就把语义对齐阶段也判断为没有作用；它首先承担的是恢复 tokenizer 改造造成的损失。

### 不能直接得到的结论：Align 仍比 Base 差多少

如果这张图计算的是通常的 **每 token PPL**，Base 与 Char 的分词单位不同，数值就不能直接横比。PPL 是平均 token 负对数似然的指数，tokenization 本身会影响它；Hugging Face 的文档也明确强调了这一点。[Hugging Face](https://huggingface.co/docs/transformers/perplexity?utm_source=chatgpt.com)

因此，不能直接用 `9.4413 / 6.0779` 计算“能力还差多少”，也不能由 `28.9820 → 9.4413` 推导“恢复了多少比例的原模型能力”。

跨 tokenizer 比较，更合适的是在相同原始文本与评估覆盖范围下，报告：

\[
\mathrm{BPC}
=
\frac{\text{总负对数似然}}
{\text{对应原始字符数}\cdot\ln 2}
\]

或者按 UTF-8 字节数归一的 BPB，并明确上下文窗口与截断策略。现有截图没有总 NLL、token 数及字符数，**无法事后从这几个 PPL 数字可靠换算。**

此外，截图中的 `Avg` 确实是六个领域 PPL 的**算术平均**，它可以作为汇总展示，但不等于把六个领域合并后计算的 corpus PPL。截图也没有 CPT 行，因此它还不能回答“当前 CPT 是否进一步恢复或超越 Base”。

## 五、PGCA 源码：主计算图基本自洽，但贡献尚未被证明

### 1. 当前没有发现明确的未来信息泄漏，零 gate 也不是已证实的问题

当前 [PGCA attention](https://github.com/Nephamaster/CC-LLM/blob/ef68d4e53c82ea7908139fb9ec14c3ee0ef431eb/src/pgca/attention.py) 只让每个位置关注**该位置自身的音形 slots**；特征由当前 `input_ids` 查表得到，没有从下一 token 的 labels 取特征。中间层的 Query 已经包含前面若干层积累的因果上下文，因此静态音形特征也可以被上下文化的 Query 选择。

门控实现是：

\[
y=x+gF_\theta(x,c)
\]

当 \(g=0\) 时，第一步内部参数 \(\theta\) 经这条路径得到零梯度，但 gate 自身通常仍可获得梯度；gate 更新后，内部参数就能开始学习。你的构建代码只把 gate 设为零，内部投影并非同时全部置零。对应实现见 [PGCA 初始化](https://github.com/Nephamaster/CC-LLM/blob/ef68d4e53c82ea7908139fb9ec14c3ee0ef431eb/src/pgca/build_pgca.py)。

**ALIGN 中 PGCA 被冻结，若初始 gate 为零，它整个对齐阶段保持零是符合阶段目标的。** 这不是 PGCA 训练失败。CPT 才是按配置开始训练 PGCA 的阶段。

目前日志没有逐层 gate、分支梯度及输出贡献，所以：

> **“PGCA 没学起来”“主干忽略了音形”“需要增加四层上下文编码器”，现在都还是假设，不能从 loss 平台直接推出。**

这里也需要更正我上一轮的一句话：**辅助头如果接在最终 hidden state 上，它回传到 PGCA 内部时仍然经过 gate，不会自动绕开零门控。** 是否需要独立分支监督，应由实际计算图决定。

### 2. 发现一个真实的拼音构造边界问题

在 [feature_vocab.py](https://github.com/Nephamaster/CC-LLM/blob/ef68d4e53c82ea7908139fb9ec14c3ee0ef431eb/src/vocab/feature_vocab.py) 中，`pinyin_candidates()` 调用 pypinyin 时没有设置 `errors`。官方默认行为是：遇到没有拼音的字符，原样保留该字符。[pypinyin 0.52.0 文档](https://pypinyin.readthedocs.io/zh-cn/v0.52.0/api.html?utm_source=chatgpt.com)

而你的后续代码会接受这个非空字符串：

```text
某个字查不到拼音
→ pypinyin 返回该字本身
→ normalize 后仍是这个汉字
→ 被接受为拼音 candidate
→ 提前返回，跳过 Unihan fallback
```

这样可能把“原汉字”错误加入拼音／韵母词表，同时没有计入拼音缺失数量。

**这是确定的源码边界缺陷，但本次 checkpoint 究竟影响了多少汉字，还需要检查实际生成的特征表。** 它不能直接解释全部 eval NaN，也不能直接解释 benchmark 增益小；但在你强调“新增汉字与音形覆盖”的项目里，应当优先修正和统计。

### 3. 做 PGCA 消融时，需要真正关闭已有模块

当前模型加载后，单独设置：

```python
model.config.use_pgca = False
```

**不会关闭已经构造好的 PGCA 分支**，因为 forward 检查的是实际模块和 feature memory 是否存在。对应逻辑在 [模型 forward](https://github.com/Nephamaster/CC-LLM/blob/ef68d4e53c82ea7908139fb9ec14c3ee0ef431eb/src/modeling_qwen3_pgca.py)。

当前实现下，快速测量模型对 PGCA 的依赖，可以保存各层 gate 后临时清零、评估相同样本，再恢复。这样得到的是**当前模型对分支的依赖程度**；要证明 PGCA 本身带来训练收益，仍需要同初始化、同数据和同 token 预算的无 PGCA 训练对照。

## 六、接下来应按什么顺序处理

| 优先级 | 具体工作 | 要解决的问题 |
|---|---|---|
| **首先** | 检查最终编码分片的有效 label 数，修复 NaN；在固定验证集上重测 ALIGN 和当前 CPT | 两个 checkpoint 的验证 NLL 是否真的改善，当前是否仍值得续训 |
| **其次** | 补齐 Base、Char、ALIGN、当前 CPT 的同口径评估；跨 tokenizer 用 BPC/BPB，同 tokenizer 保留 token PPL | tokenizer 改造损失、语义对齐恢复、CPT 增益各是多少 |
| **随后** | 记录逐层 gate、PGCA／feature embedding 梯度，以及 \(\|gF\|/\|\mathrm{SelfAttnOut}\|\)，做有效开关消融 | PGCA 是否被训练、是否被使用、是否改善结果 |
| **同步修正** | 排查非法拼音候选与真实特征覆盖；核对实际 token 预算及 4K/8K 训练是否落实 | 数据和实现是否与实验设定一致 |
| **最后再决定** | 根据上述结果选择分支 warm-up、参数组学习率、辅助目标或上下文编码器 | 对已定位的瓶颈进行改造 |

**当前 checkpoint 值得保留，没有足够证据要求从头重训。** 对齐阶段已经显示出明确收益，CPT 也呈现了正常的前期学习；目前真正阻碍判断的是验证 loss 全程不可用，以及缺少能够分离“普通 CPT 收益”和“PGCA 收益”的证据。先用修复后的验证集评估现有 checkpoint，再决定剩余训练预算和架构调整，能让下一步实验有清楚的依据。