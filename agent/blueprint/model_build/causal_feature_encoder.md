# CC-LLM 音形上下文建模方案

> 2026-10-18 实施校对：音形 schema v2 的候选/未知标签修复已落地，见 `checkpoint/26-10-18-feature-vocab-repair.md`。当前构建默认最多 16 个读音、17 个 slots，运行时必须读取 manifest，不能沿用旧版 8/9 的固定值。源码已有 `pgca_enabled` 运行时旁路；本次沿用它控制 Phase1/2，不删除分支权重。上下文与辅助模块由新构建参数启用，旧模型配置默认不创建它们。

本版固定采用：**保留单汉字 Tokenizer 和无 gate 的 PGCA 残差融合；在静态音形特征与 PGCA 之间加入共享的轻量因果 Transformer；Phase2 联合优化语言建模、下一 token 字音预测、下一 token 字形属性预测。**

这里的“下一 token”始终指紧邻的 \(x_{t+1}\)，仅当它是汉字且对应属性有效时计算辅助损失，**不跨过标点或英文寻找后面的汉字**。

这是一套待验证的研究方案，不预设它一定能修复当前 benchmark 退化。ChineseBERT 支持将音形信息用于中文表示学习，PLOME 支持引入发音预测目标；但二者不能直接证明下面这套 decoder-only 改造的收益，因此需要通过独立消融验证。[ACL Anthology](https://aclanthology.org/2021.acl-long.161/)

## 1. 总体结构与设计边界

整条计算路径分为两部分：

**输入侧：汉字 → 静态音形特征 → 轻量因果上下文编码器 → 上下文化音形 memory → PGCA → 主干隐藏状态。**

**输出侧：最终隐藏状态 → 原 LM Head、字音预测头、字形属性预测头。** 辅助头只参与训练与专项诊断，不向生成文本中添加拼音或字形标记。

当前 `feature_embedding.py` 为每个位置生成若干拼音候选 slot 和一个字形 slot；每个拼音 slot 已融合完整拼音、声母、韵母、声调，字形 slot 则融合笔画数、部首笔画组合、结构类别。新结构保留这一接口，不把候选先压缩成一个向量后直接交给 PGCA。

| 项目 | 本版确定的处理 |
|---|---|
| 主干 | Qwen3-1.7B，保持原 Self-Attention、MLP、LM Head |
| PGCA 融合 | 保留直接残差相加，不恢复可学习 gate |
| 音形上下文编码器 | **4 层、512 隐藏维度、8 个注意力头**，全模型共享一套 |
| PGCA memory | 保留“多个完整读音候选＋一个字形结构 slot”，维度与现有 PGCA 一致 |
| 字音辅助任务 | 预测下一汉字的**带调完整拼音**，多音字采用候选集合监督 |
| 字形辅助任务 | 预测下一汉字的笔画数、部首笔画组合、结构类别 |
| Phase1 | 主干适配新 Tokenizer，音形分支旁路并冻结，不计算辅助损失 |
| Phase2 | 开启音形分支，先短暂适配新增模块，再联合全参数 CPT |

**已修复的音形 schema v2 是构建前置条件。** 使用经过验收、带版本与哈希的音形词表和索引；不得把新表直接覆盖到旧 PGCA checkpoint。语义 tokenizer 和 token ID 不变，不需要重新构建语义词表。上下文网络不能选择一个不存在的候选，辅助目标也不能使用错误标签。

## 2. 音形上下文编码器：沿文本位置建模，保留候选级差异

### 2.1 输入输出与整体计算

设批大小为 \(B\)，文本长度为 \(T\)，每个位置最多有 \(P\) 个拼音候选，加上一个字形 slot，得到 \(S=P+1\) 个 slots。\(P\) 从 feature manifest 读取，当前新构建默认 \(P=16,S=17\)。主干隐藏维度记为 \(D\)，当前 1.7B 配置下为 2048；轻量编码器维度记为 \(d=512\)。

新模块的输入输出固定为：

| 张量 | 形状 | 含义 |
|---|---|---|
| `static_memory` | `[B, T, S, D]` | 当前音形 Embedding／Projector 生成的静态特征 |
| `feature_mask` | `[B, T, S]` | 每个 slot 是否包含有效特征 |
| `context_inputs` | `[B, T, d]` | 每个文本位置的音形摘要 |
| `context_states` | `[B, T, d]` | 经因果 Transformer 得到的上下文表示 |
| `contextual_memory` | `[B, T, S, D]` | 保留原候选结构的上下文化特征，传入 PGCA |

整个模块分为三步：**位置内汇总 → 位置间因果建模 → 候选级上下文融合**。

### 2.2 位置内汇总：只为上下文网络生成摘要，不丢弃原 slots

对位置 \(t\)，先对有效拼音候选做 masked mean，得到拼音摘要 \(\bar f_t^p\)，再与字形向量 \(f_t^g\) 拼接：

\[
z_t=W_{\mathrm{in}}[\bar f_t^p;f_t^g]+e_{\tau(t)}.
\]

其中，\(\bar f_t^p,f_t^g\in\mathbb R^D\)；\([\,;\,]\) 表示拼接；\(W_{\mathrm{in}}\in\mathbb R^{d\times2D}\) 将摘要投影到 512 维；\(e_{\tau(t)}\) 是位置类型 Embedding，用于区分汉字、拉丁字符、数字、标点、特殊 token 等。

这里的平均**只用于生成上下文网络输入**，原始候选向量全部保留，后面仍逐候选更新。

没有合法拼音时，拼音摘要置零并记录缺失状态，不能将 `<unk>` 当作真实读音平均进去。非汉字位置保留在序列中，使用类型表示保持位置与边界连续；其最终 PGCA slots 仍保持无效，不凭空生成音形信息。首版不额外输入主干语义 Embedding，以便明确检验“音形特征自身的上下文化”是否有价值。

### 2.3 位置间建模：一套共享的轻量因果 Transformer

令：

\[
c_{1:T}=\operatorname{CausalFeatureEncoder}(z_{1:T}).
\]

这里 \(c_t\in\mathbb R^d\) 只能依赖 \(z_1,\ldots,z_t\)，不能访问未来位置。

| 参数 | 首轮固定配置 |
|---|---|
| 层数 | 4 |
| Hidden size | 512 |
| Attention heads／Head dimension | 8／64 |
| FFN | SwiGLU，Intermediate size = 1536 |
| 归一化 | Pre-RMSNorm |
| 位置编码 | RoPE，使用与主干一致的 token 位置编号，按自身 head dimension 构造 |
| Attention | Causal，使用 SDPA |
| Dropout | 首版设为 0，避免同时引入额外干预 |
| 共享方式 | **每次模型 forward 计算一次，所有 PGCA 层复用结果** |

不能给六个 PGCA 层分别增加四层 Transformer，否则会变成六套上下文网络；也不把 `[T,S]` 展平成长度 \(TS\) 的序列。后者在普通全注意力下将注意力矩阵规模从 \(T^2\) 放大到 \(S^2T^2\)，没有必要。

按上述无偏置注意力投影和 SwiGLU 配置计算，四个 Transformer block 约 **13.6M 参数**；加上摘要投影和下面的候选融合模块，新增上下文模块约 **18M 参数**。这是结构计算值，最终以构建报告为准。

### 2.4 候选级融合：上下文必须能分别改变不同候选

不能直接将 \(c_t\) 当作唯一的 PGCA memory，也不建议只给所有候选加上完全相同的线性偏移。本版使用带非线性交互的共享残差适配器：

\[
\widetilde f_{t,s}
=
m_{t,s}\left[
f_{t,s}
+
W_{\mathrm{up}}
\operatorname{SiLU}
\left(
W_f\operatorname{RMSNorm}(f_{t,s})
+W_c c_t
+e_{\kappa(s)}
\right)
\right].
\]

各部分含义如下：\(f_{t,s}\in\mathbb R^D\) 是原始第 \(s\) 个 slot；\(m_{t,s}\in\{0,1\}\) 是有效性标记；\(W_f\in\mathbb R^{d\times D}\)、\(W_c\in\mathbb R^{d\times d}\)、\(W_{\mathrm{up}}\in\mathbb R^{D\times d}\) 是投影矩阵；\(e_{\kappa(s)}\in\mathbb R^d\) 区分拼音与字形两种 slot 类型。

这使同一上下文可以分别调整 `xing2`、`hang2` 和字形向量，同时保留它们各自的原始特征。所有拼音候选共享参数，**不为“第一个读音”“第二个读音”设置不同的语义参数**，避免学习候选排序偏置。

随后保持原有 PGCA 计算和残差形式：

\[
H'_\ell
=
H_\ell+
\operatorname{SelfAttn}_\ell(\operatorname{Norm}(H_\ell))
+
\operatorname{PGCA}_\ell(\operatorname{Norm}(H_\ell),\widetilde F).
\]

其中 \(\ell\) 为主干层，\(\widetilde F\) 是所有位置的上下文化 memory。没有任何有效 slot 的位置，PGCA 最终输出必须为零，包括输出投影存在 bias 时也要满足这一条件。

### 2.5 代码改动范围

当前实施接口如下，模型源码在 Phase1/2 保持一致：

| 文件／模块 | 修改内容 |
|---|---|
| `src/vocab/feature_vocab.py`、`feature_index.py` | 修复读音候选、规范化和未知标签；生成有效监督索引（若已完成则忽略） |
| **新增** `src/embedding/context_encoder.py` | 摘要投影、4 层因果 Transformer、候选级残差融合 |
| `src/embedding/feature_memory.py` | 在现有静态 memory 后接入上下文编码器，传递位置、padding mask 和 cache |
| `src/modeling_qwen3_pgca.py` | 调整 feature memory 构造时机，接入位置与 cache；增加辅助头及可选辅助 logits 输出 |
| `src/configuration_qwen3_pgca.py` | 保存编码器尺寸、辅助头开关、损失权重和特征版本 |
| `scripts/train/cpt/` | 本仓库管理的 Template、`BaseLoss`、参数组学习率、指标与训练脚本；使用 `scripts/train/env/` 普通安装的框架 |
| 构建／导出与测试 | 复制新增运行代码，检查权重完整性，覆盖旁路、因果、loss、cache 测试 |

新编码器放在 `model.feature_memory_builder.context_encoder` 下。旧脚本只冻结 `feature_memory_builder.feature_embedding`，不能覆盖新增编码器；新脚本改为冻结整个 `model.feature_memory_builder`，并冻结模型顶层的 `auxiliary_heads`。

## 3. 验收、消融与正式执行顺序

### 3.1 上大规模训练前必须通过的测试

| 测试 | 验收要求 |
|---|---|
| 候选与标签 | 常见多音字候选完整；未知读音不产生轻声真值；辅助类别与索引版本一致 |
| 因果性 | 修改后缀不能改变前缀的上下文特征、PGCA 权重及 logits |
| Slot 保留 | 拼音候选仍可分别访问；无效 slots 始终为零；重排候选不改变最终结果 |
| Phase1 旁路 | 音形编码器和 PGCA 调用次数为零；保存后权重键仍完整 |
| 初始化与梯度 | 首次开启不改变主干输出；输出投影更新后，内部 PGCA／上下文模块获得梯度 |
| 空监督与分布式 | 无合法音形标签时不产生 NaN；单卡、累积与 DDP 的归一化一致 |
| 推理缓存 | 完整前缀重算与 cache 增量解码结果在相应精度容差内一致 |
| 序列边界 | 若后续启用 packing，主干与音形编码器采用相同文档隔离边界 |

### 3.2 核心消融只改变一个因素

所有实验从**同一个旁路训练得到的 Phase1 checkpoint**出发，使用相同 Phase2 文本、训练预算和下游 SFT 配置：

| 实验 | 静态 PGCA | 上下文编码器 | 音形辅助目标 | 作用 |
|---|---:|---:|---:|---|
| A：Char-only | 无 | 无 | 无 | 字符级主干参照 |
| B：Static-PGCA | 有 | 无 | 无 | PGCA 基础贡献 |
| C：Context-PGCA | 有 | 有 | 无 | 单独检验上下文化 |
| D：Static-PGCA＋Aux | 有 | 无 | 有 | 单独检验辅助监督 |
| **E：完整方案** | 有 | 有 | 有 | 检验两项改造能否协同 |

先以相同的 **0.5B–1B Phase2 预算**完成筛选，再决定进入完整 10B 的实验。最终研究结论还应补充参数量匹配或固定打乱“汉字—音形映射”的对照，区分真实音形结构收益与单纯增加容量的收益。

结果分三层报告：**语言建模与通用任务是否保持；CSC 等专项任务是否改善；PGCA 是否真正依赖语境与正确音形内容。** Attention 诊断继续采用正常、均匀权重、旁路、分别屏蔽拼音／字形的配对实验，不能只凭热力图或辅助准确率证明机制有效。

**正式执行顺序为：修复并冻结音形资源版本 → 实现旁路与稳定初始化 → 实现上下文编码器及 cache → 接入辅助头和统一损失 → 通过一致性测试 → 小预算消融 → 完整 Phase2。**

这套方案的关键不是把音形分支做得更深，而是让它同时具备三种可验证的性质：**输入特征可靠、上下文计算严格因果、训练收益能够被独立归因。**

## 附录：音形编码模块

当前 CC-LLM 中，计算音形 Embedding 的模块是 `PhoneticGlyphFeatureEmbedding`。它位于 PGCA 交叉注意力之前，负责把每个汉字的离散音形属性转换为连续向量。

### 1. 模块是什么，位于哪里？

源码位置：

- [src/embedding/feature_embedding.py](../../../src/embedding/feature_embedding.py)：定义 `PhoneticGlyphFeatureEmbedding`。
- [src/embedding/feature_memory.py](../../../src/embedding/feature_memory.py)：通过 `FeatureMemoryBuilder` 查询汉字音形属性并调用 Embedding。
- [src/modeling_qwen3_pgca.py](../../../src/modeling_qwen3_pgca.py)：创建 FeatureMemoryBuilder，并将输出传给 PGCA。

目前的完整计算流程是：

```
                        input_ids
                            │
             ┌──────────────┴──────────────┐
             │                             │
      Semantic Embedding           FeatureMemoryBuilder
             │                             │
             │                     查询静态音形属性 ID
             │                             │
             │                  PhoneticGlyphFeatureEmbedding
             │                     ├─ 拼音 Embedding
             │                     ├─ 声母 Embedding
             │                     ├─ 韵母 Embedding
             │                     ├─ 声调 Embedding
             │                     ├─ 笔画数 Embedding
             │                     ├─ 部首笔画 Embedding
             │                     └─ 结构 Embedding
             │                             │
             │                     两个 MLP Projector
             │                             │
             ▼                             ▼
        Qwen 主干隐藏状态              Feature Memory
             │                             │
             └────────── PGCA ─────────────┘
                            │
                      残差直接相加
                            │
                       后续 Qwen 层
```

它由 7 个可训练的 `nn.Embedding` 表和 2 个 MLP Projector 组成。

当前默认配置为：

| 参数 | 数值 |
| ---- | ---- |
| 原始音形 Embedding 维度 | 256 |
| Projector 输出维度 | 2048 |
| 最大拼音候选数 | 从 manifest 读取，新构建默认 16 |
| 字形结构 Slot | 1 |
| PGCA Feature Memory | `[B, T, P+1, 2048]`，新构建默认 17 slots |

这里的字形是笔画、部首笔画、结构等属性表示，并不是汉字图像编码器。

该模块只在每次模型 forward 中构造一次 Feature Memory，供多个 PGCA 层复用，而不是每个 PGCA 层各自重新计算。

### 2. 它什么时候被训练？

首先区分初始化和训练：构建 PGCA 模型时，这些 Embedding 和 Projector 会进行随机初始化。它们没有经过独立预训练，也没有加载现成的音形编码模型。

| 阶段               | 音形 Embedding 的状态     | 是否更新参数 |
| ---------------- | -------------------- | ------ |
| Phase0：PGCA 模型构建 | 随机初始化                | 否      |
| Phase1：语义对齐      | 冻结；新方案中还会旁路整个 PGCA   | 否      |
| Phase2：全参数 CPT   | 与 PGCA 和 Qwen 主干联合训练 | 是      |

冻结参数本身不会关闭 PGCA。当前源码已有 `pgca_enabled` 开关，是否实际执行取决于运行配置；新脚本在 Phase1 显式设置 `PGCA_ENABLED=false`，Phase2 显式设为 true，不能仅根据冻结正则推断分支是否旁路。

### 3. 加入轻量 Transformer 后会怎样？

我们计划保留原有 `PhoneticGlyphFeatureEmbedding`，并在它与 PGCA 之间增加 `CausalFeatureEncoder`。

```
离散音形属性 ID
       │
       ▼
PhoneticGlyphFeatureEmbedding   ← 现有模块
       │
       ▼
Static Feature Memory
       │
       ▼
4-Layer Causal Feature Encoder  ← 新增模块
       │
       ▼
Contextual Feature Memory
       │
       ▼
PGCA Cross-Attention
       │
       ▼
Qwen Hidden States
```

训练安排为：

- Phase1： 音形 Embedding、上下文编码器、PGCA 全部旁路并冻结。
- Phase2-A： 冻结 Qwen 主干，训练音形 Embedding、上下文编码器、PGCA 和辅助预测头。
- Phase2-B： 全部解冻，联合优化三个任务。

因此，新增的轻量 Transformer 不是替代音形 Embedding，而是让已有的静态音形 Embedding 具备上下文感知能力。

这也是下一版架构的关键变化：从原来的“静态音形向量直接进入 PGCA”，变为“音形向量先进行因果上下文编码，再参与 PGCA 融合”。
