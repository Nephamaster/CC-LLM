# Character\-Level Chinese Large Language Model

## 动机

当前大语言模型通常将文本表示为离散 token 序列，并通过 token 之间的上下文关系学习语言分布。对于汉语场景而言，这种建模范式虽然能够获得较强的通用语义理解和生成能力，但其输入表示主要建立在 token ID 层面，通常没有显式刻画汉字所固有的音形结构信息。

汉语书写系统具有显著的形、音、义耦合特征。单个汉字不仅是文本中的基本书写单位，同时也承载了字形结构、部件组合、部首线索、笔画模式、拼音、声调等多层次语言信息。这些信息对于汉语理解、语义消歧、罕见字处理、同音字区分、形近字辨析以及中文拼写纠错等任务均具有重要价值。

已有中文表示学习和中文拼写纠错研究表明，拼音和字形并非外部冗余信息，而是能够有效增强中文语言表示的重要结构化信息来源。现有大语言模型虽然可以通过大规模语料学习汉语分布，但大多缺乏针对汉字音形结构的显式建模机制，尤其在细粒度汉字辨析、音近/形近错误处理、多音字消歧、罕见字理解、OCR/ASR 噪声恢复等任务中仍存在不足。

因此，本研究从汉语语言特性出发，探索如何将汉字的拼音、声调、字形、部首、笔画和结构等音形信息有效建模到 decoder\-only 大语言模型中，从而构建更符合汉语结构特征的通用汉语大模型。

本研究的核心问题可以概括为：

如何在保留大语言模型通用建模能力的前提下，将音形信息有效建模到LLM中，并增强对中文的理解和生成？

## 总体目标

本研究拟构建一个基于 decoder\-only LLM 的汉字原生大模型。整体改造包括三部分。

第一，输入特征改造。中文文本不使用 BPE/BBPE 的多汉字 token，而是强制每个汉字独立成 token，同时保留非汉字部分，以保证模型仍能处理真实场景中的中英混排、数学表达、代码和 URL。

第二，音形信息建模。声母、韵母、声调、部首、笔画、字形结构、五笔/仓颉/拆字等信息不作为普通文本 token 进入序列，而是作为 feature vocabulary 与每个汉字位置对齐，形成汉字音形特征库。

第三，模型结构改造。在 Transformer block 中引入 Phonetic\-Glyph Cross\-Attention Branch，简称 PGCA。原有 causal self\-attention 继续负责上下文语义建模；PGCA 分支负责让每个位置的隐藏状态查询当前汉字对应的音形特征。两路信息融合后进入 MLP，从而使模型在自回归建模过程中持续获得汉字音形结构信息。

## 方法论

### 1 词表设计

本研究中的词表分为两类：语义词表和音形特征词表。

语义词表记为 `V_sem`，它决定模型最终可以输入和输出哪些 token。音形词表记为 `V_feat`，它不参与普通文本解码，不进入 lm\_head，而是作为 PGCA 的结构化特征来源。

#### 1\.1 语义词表 `V_sem`

`V_sem` 包含两部分。

第一部分是汉语单字 token。所有中文汉字均以单字形式进入词表，不允许出现多汉字 token。初版覆盖以下范围：常用简体字；常用繁体字；《通用规范汉字表》；CJK Unified Ideographs 基本区；姓名、地名、古文和异体字中常见的高频罕见字。

第二部分是非汉字 token 和 特殊 token，即原底座 tokenizer 中不包含汉字的 token。

因此，新语义词表可形式化为：

`V_sem = V_hanzi_single ∪ V_nonhan_original`

其中，`V_hanzi_single` 只包含单个汉字，`V_nonhan_original` 只包含不含汉字的原 tokenizer token。

#### 1\.2 音形词表 `V_feat`

音形词表不参与文本生成，而是为每个汉字提供结构化特征。包括以下子词表：

`pinyin_id`：拼音序列

`shengmu_id`：声母。

`yunmu_id`：韵母。

`tone_id`：声调。

`stroke_seq_id`：笔顺或笔画序列。

`structure_id`：字形结构，例如左右结构、上下结构、半包围、全包围、独体字等。

`glyph_id / glyph_vec`：字形图像编码，可来自 CNN、ViT 或预先训练的 glyph encoder。

对于多音字，`pinyin_id`不应只保存一个静态读音，而应保存候选读音集合。例如“行”可以同时关联 `xing2` 和 `hang2`。PGCA 中的 query 来自上下文隐藏状态，理论上可以根据上下文动态选择更相关的读音特征。

### 2 Tokenizer 设计

本研究的 tokenizer 是一个“汉字单字化 \+ 非汉字保留原编码能力”的混合规则 tokenizer。它不训练 BPE merge，不允许中文汉字被合并成多字 token。

#### 2\.1 编码流程

输入文本首先经过轻量规范化。只做必要处理，例如统一换行、清理非法控制字符、修复无效 Unicode。

然后从左到右扫描文本。

如果当前位置匹配 special token，则输出对应 special token id。

如果当前位置是 CJK 汉字，则直接输出该汉字对应的单字 token id，并同时生成该位置的音形 ids。

如果当前位置是连续非汉字 span，则调用原底座 tokenizer 对该 span 编码。由于该 span 不含汉字，因此不会产生多汉字 token。这样可以保留英文、代码、数字、数学符号、URL 等文本的原有压缩能力。

如果遇到无法被原 tokenizer 或新词表覆盖的字符，则使用 byte fallback。

最终 tokenizer 输出两类结果：

`input_ids`：普通语义 token 序列，用于模型输入和 lm\_head 输出。

`feature_ids`：与 `input_ids` 对齐的音形特征序列。对于汉字 token，包含拼音、部首、字形等特征；对于非汉字 token，可以填充空特征、token type 特征或 zero feature。

#### 2\.2 解码流程

解码时只使用 `input_ids`。汉字 token 直接还原为对应汉字；非汉字 token 调用原 tokenizer 的 decode 逻辑；byte fallback 聚合后还原为原始 Unicode 字符。`feature_ids` 不参与解码。

### 3 Embedding 设计

模型输入 embedding 分为两部分：语义 token embedding 和音形 feature embedding。

#### 3\.1 语义 token embedding

语义 token embedding 对应 `V_sem`，记为：

`E_sem ∈ R^{|V_sem| × d_model}`

初始化策略如下。

对于保留的非汉字 token，直接复制原底座模型 embedding 和 lm\_head 权重。

对于单汉字 token，如果原 tokenizer 中已经存在完全相同的单汉字 token，则直接复制其 embedding 和 lm\_head 权重。

对于新加入的单汉字 token，如果原 tokenizer 会将其编码为多个旧 token，则用这些旧 token embedding 的平均值或加权平均值初始化。lm\_head 也采用同样的方式初始化。

对于 byte fallback token，如果原底座 tokenizer 已有对应 byte token，则复制；否则小尺度随机初始化。

这样可以最大限度继承原模型语义空间，避免新词表随机初始化导致训练初期不稳定。

#### 3\.2 音形 feature embedding

音形 feature embedding 不进入 lm\_head。它们只用于构造每个汉字位置的 feature memory。

拼音侧可以设计为：

`e_py = MLP([E_shengmu, E_yunmu, E_tone])`

字形侧可以设计为：

`e_glyph = MLP([E_stroke_count, E_structure])`

如果使用字形图片，可以额外引入 glyph encoder：

`e_img = GlyphEncoder(render(char))`

每个汉字 token 对应一个 feature memory：

`M_i = [e_char, e_pinyin_1, ..., e_pinyin_k, e_glyph, e_structure, ...]`

其中 `k` 是多音字候选读音数量。对于非多音字，`k=1`。对于非汉字 token，`M_i` 可以设为空、零向量。

### 4 Backbone：PGCA 音形交叉注意力

本研究在原 decoder block 中加入 PGCA 分支。

设第 `l` 层输入隐藏状态为：

`H^l ∈ R^{n × d}`

原模型 self\-attention 分支保持不变：

`S^l = SelfAttn(RMSNorm(H^l))`

该分支使用 causal mask，负责上下文语义建模。

PGCA 分支的输入同样来自当前层隐藏状态：

`Q^l = W_q^l RMSNorm(H^l)`

对于第 `i` 个位置，根据该位置 token 的 feature memory `M_i` 生成 key 和 value：

`K_i^l = W_k^l M_i`

`V_i^l = W_v^l M_i`

然后做 position\-wise cross\-attention：

`Z_i^l = softmax(Q_i^l K_i^{lT} / sqrt(d)) V_i^l`

这里的 cross\-attention 是局部的，即每个位置只查询自身 token 对应的音形特征，而不是查询整句所有 token 的音形特征。这样可以避免自回归训练中的未来信息泄漏。

融合方式建议采用 gated residual：

`H_mid^l = H^l + S^l + α_l W_o^l Z^l`

其中 `α_l` 是可学习 gate，初始化为 0 或 0\.01。这样模型初始行为接近原底座模型，训练时再逐渐学习使用音形分支。

然后进入原有 MLP：

`H^{l+1} = H_mid^l + MLP(RMSNorm(H_mid^l))`

这对应“语义 self\-attention 和音形 cross\-attention 两路并行，融合后进入 MLP”的结构。

#### 4\.1 PGCA 插入位置

轻量版：只在中间 1/3 层加入 PGCA，例如 1\.7B 模型中选择 4 到 8 个中间层。

间隔版：每 4 层加入一次 PGCA。

全层版：每层都加入 PGCA，但 gate 初始化为 0，并用较小学习率训练。

初版推荐“中间层插入”。底层过早注入音形信息可能扰动原始 token 表示，高层过晚注入又可能无法充分参与语义组合，中间层较稳妥。

#### 4\.2 双塔结构

采用“单主干 \+ PGCA 辅助分支”的弱双塔结构：

语义主干：原 LLM decoder block。

音形分支：轻量 PGCA，只提供 per\-position feature memory 查询。

这样既保留原模型能力，又能注入音形信息，工程成本也更可控。

### 5 训练策略

训练分为三个阶段。

#### Phase 0：Tokenizer 与权重迁移验证

目标是确保新 tokenizer 可用，新 embedding/lm\_head 初始化合理。

主要任务：用少量中文、英文、代码、中英混排文本做 forward 测试，观察 loss 是否异常。

必须完成以下指标：

- 中文文本必须完全单汉字化。

- 任意文本必须可逆编码。

- special token 和 chat template 行为正常。

- 新模型 forward 不报错，初始 loss 不应极端异常。

#### Phase 1：语义空间对齐

目标是让新 tokenizer 下的语义 embedding 和 lm\_head 初步适配原模型语义空间，让模型恢复原有能力。

训练参数：`E_sem`、`lm_head`；PGCA 暂不开启。

训练目标：causal language modeling loss，即预测下一个语义 token。

训练数据：高质量短文本为主，覆盖中文、英文、数字、符号、代码、中英混排。

1\.7B 试验模型先用 1B 到 5B 新 tokenizer token 做验证。如果资源允许，再扩大到 10B token。

#### Phase 2：继续预训练

目标是让模型在新 tokenizer 和 PGCA 结构下恢复并增强通用语言建模能力。

训练参数：全参数。

训练目标：causal language modeling。

训练数据：

- 中文通用语料：50%。

- 中文知识密集和长文档：15%。

- 英文、多语言和中英混排：10%。

- 数学、代码、科学文档：15%。

#### Phase 4：下游能力增强

目标是将预训练模型适配到可用的对话、纠错和中文专业任务。

任务方向包括：

- 中文对话 SFT。

- 中文纠错：拼写、语法

- 文言文理解与生成

- 汉语专业领域问答

这一阶段可以采用 instruction tuning 格式。

### 6 数据设计

本研究的数据分为五类。

第一类是中文通用语料。包括高质量中文网页、百科、新闻、论坛、书面语、现代汉语文本等。用于恢复和增强中文语言建模能力。

第二类是中文知识密集与长文档语料。包括百科、教材、论文、政策法规、技术文档、PDF/OCR 清洗文本、长篇文章等。由于单汉字 tokenizer 会拉长序列，长文档数据对模型适配长上下文很重要。

第三类是英文、多语言和中英混排语料。虽然研究目标是汉语模型，但真实中文场景中存在大量英文缩写、术语、代码片段、论文标题、产品名、URL 和中英混排内容，因此不能完全移除英文和多语言数据。

第四类是数学、代码和科学文档语料。用于保持通用推理、代码和结构化符号处理能力。

第五类是音形增强数据。包括汉字属性表、多音字上下文数据、拼音标注数据、形近字混淆集、音近字混淆集、OCR 噪声数据、ASR 噪声数据、CSC 数据等。

先做 1\.7B 模型的小规模验证。可以采用：

Phase 1：1B 到 5B token。

Phase 2：3B 到 10B token。

Phase 3：20B 到 100B token。

Phase 4：若干百万到数千万条 instruction / correction / QA 样本。

### 7 推理优化

单汉字 tokenizer 会显著增加中文序列长度，PGCA 也会带来额外计算。因此必须设计推理优化。

第一，PGCA 只在部分层插入，不做全层默认开启。

第二，PGCA 是局部 cross\-attention，每个位置只查询自身音形 feature memory。由于每个汉字的 feature 数量很小，PGCA 的复杂度接近 `O(n · m · d)`，其中 `m` 是每个 token 的音形特征数量，远小于 self\-attention 的序列长度维度。

第三，汉字 feature memory 可以预计算。对于每个汉字 token，可以提前构建音形 feature embedding，并在推理时按 token id 查表，减少重复计算。

第四，对非汉字 token 跳过 PGCA 或只使用 token type feature。

第五，结合常规 LLM 推理优化，包括 FlashAttention、Paged KV Cache、GQA/MQA、KV cache 量化、权重量化、chunked prefill、长上下文滑窗、speculative decoding 等。

第六，实验中必须单独报告效率指标，包括 prefill latency、decode latency、显存占用、tokens/s、同等上下文窗口可容纳的中文字符数，以及 PGCA 带来的额外开销。

## 4\. 实验设计

### 4\.1 核心研究问题

Q1：单汉字 tokenizer 是否能够提升模型的汉字级建模能力？

Q2：音形信息是否能够提升模型在中文理解、纠错、罕见字、多音字、形近字和音近字场景中的表现？

Q3：PGCA 是否优于简单的输入层 embedding 融合？

Q4：在引入单汉字 tokenizer 和 PGCA 后，模型是否仍能保持通用能力？

Q5：单汉字建模带来的序列变长是否可以通过结构设计和推理优化控制在可接受范围内？

### 4\.2 对比模型

Base：原始 Qwen3\-Base 或同规模 LLM。

Char：替换为单汉字 tokenizer，只做 embedding/lm\_head 适配，不加音形信息。

Char \+ PGCA：加入音形交叉注意力分支。

ChineseBERT、MacBERT、Chinese LLaMA、现有开源中文 LLM 作为任务级对照。

### 4\.3 消融实验

- Tokenizer 消融

- 音形特征消融

- PGCA 结构消融

- 初始化消融

### 4\.4 评估任务

- 通用中文能力：C\-Eval、CMMLU、中文阅读理解、中文摘要、中文开放问答、中文常识推理

- 汉字细粒度能力：SIGHAN、LEMON、ECSpell、CSCD\-NS

- 通用能力保持：MMLU、GSM8K、HumanEval、英文 perplexity、代码 perplexity、中英混排问答、数学符号和代码文本建模。

- 效率指标：中文字符/token 比例、同等上下文长度下可容纳中文字符数、prefill latency、decode latency、显存占用、KV cache 大小、长文本 perplexity、PGCA 额外开销

### 4\.5 预期结论

如果方法有效，应观察到：

单汉字 tokenizer 相比原 tokenizer 在汉字级定位、纠错和罕见字任务上更稳定。

PGCA 相比简单 embedding fusion 在多音字、形近字、音近字、CSC、OCR/ASR 噪声任务上更有效。

音形信息对通用中文任务有一定收益，但更主要的收益应体现在细粒度汉字任务和鲁棒性任务上。

相比原模型，通用能力可能略有下降，但通过非汉字 token 保留、混合数据继续预训练和 LoRA/全参适配，可以将下降控制在可接受范围内。

效率上，单汉字 tokenizer 会增加中文序列长度，但 PGCA 本身由于是局部特征查询，不应成为主要瓶颈。真正的效率瓶颈仍然来自 self\-attention 的序列长度增长。

## 5\. 最小可行版本

为了降低研究风险，先做最小可行版本。

模型：Qwen3\-1\.7B\-Base。

Tokenizer：汉字单字化，非汉字 token 保留原 tokenizer，byte fallback 兜底。

Feature：先使用拼音、声母、韵母、声调；笔画、结构类型，不使用图像 glyph encoder。

PGCA：只在中间若干层加入，gate 初始化为 0。

训练：先做 embedding/lm\_head 对齐，再做全参CPT。

数据：先用 20B token 以内验证，不直接追求大规模训练。

评估：优先看 loss 收敛、通用中文 benchmark 和效率指标。

