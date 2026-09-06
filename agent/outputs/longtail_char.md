下面给出一版可直接用于组会汇报的“新增汉字 Token 与音形覆盖增强数据”构建方案。目标、输入、步骤、公式、产物、验收标准全部固定。

## 新增汉字 Token 与音形覆盖增强数据构建方案

### 1. 目标

该部分数据占 Phase2 总训练数据的 **5%**。若 Phase2 为 20B tokens，则该部分目标为 **1B tokens**。

目的不是构造“罕见字语料”，而是补偿词表重构带来的新增单汉字 token：

> 对新词表中“原 Qwen3 无独立单字 token、当前新增为单汉字 token”的汉字进行强化训练，使其获得足够且多样的自然上下文，同时提高 PGCA 所需拼音、声韵调、部首、笔画和结构特征的训练覆盖。

---

## 2. 输入数据

### 2.1 新增汉字集合

直接从现有词表构建结果中确定：

\[
V_{new}
=
\{c\mid c\text{ 在新词表中为新增单汉字 token}\}
\]

数据来源：

- `new_token_init_token_ids.json`
- `char_feature_index.jsonl`
- `hanzi_set.txt`

只保留真正对应单个汉字的新增 token。

对每个汉字保存：

- `char`
- `token_id`
- 拼音候选
- 声母
- 韵母
- 声调
- 部首
- 笔画数
- 字形结构

最终形成 `new_hanzi_tokens.parquet`。

### 2.2 文档候选池

不单独抓取新的“罕见字数据集”，直接从 Phase2 已清洗中文语料中挖掘：

- CCI3.0-HQ
- FineWeb-Edu-Chinese
- WanJuan
- Chinese Cosmopedia
- Wikipedia
- FineWeb-zhtw
- Wikisource
- ect-krp

这一设计保证增强数据仍然以真实自然语言为主。

---

# 3. 第一步：统计新增汉字在全语料中的覆盖情况

对全部中文候选文档做一次字符级扫描。

对每个新增汉字 \(c\) 统计：

\[
TF(c)=\text{汉字 }c\text{ 在全部候选语料中的总出现次数}
\]

\[
DF(c)=\text{包含汉字 }c\text{ 的不同文档数量}
\]

其中：

- \(TF(c)\)：衡量字符出现总量；
- \(DF(c)\)：衡量字符拥有多少种不同训练上下文。

实际筛选以 **DF 为主，TF 为辅**。

同时统计：

- `source_df`：出现在多少不同数据源；
- `traditional/classical/modern` 分布；
- 拼音候选数；
- 是否多音字。

产物：

`new_hanzi_frequency.parquet`

---

# 4. 第二步：定义新增汉字覆盖等级

根据文档频率 \(DF(c)\) 将新增汉字分为四档：

| 等级 | 文档频率 |
|---|---:|
| Uncovered | \(DF=0\) |
| Low | \(1\le DF<100\) |
| Medium | \(100\le DF<1000\) |
| Well-covered | \(DF\ge1000\) |

该分档的作用不是决定是否保留汉字，而是决定后续采样优先级。

核心原则：

> 已经拥有大量自然上下文的新 token 不再重点过采样，有限的数据预算优先分配给 Low 和 Uncovered token。

---

# 5. 第三步：计算每篇文档的新增 Token 价值

对于文档 \(d\)，定义：

\[
S_{new}(d)
=
\sum_{c\in U(d)\cap V_{new}}
\frac{1}{\sqrt{DF(c)+1}}
\]

各项定义：

- \(d\)：当前文档；
- \(U(d)\)：文档中出现的不同汉字集合；
- \(V_{new}\)：新增单汉字 token 集合；
- \(DF(c)\)：包含汉字 \(c\) 的不同文档数；
- \(S_{new}(d)\)：文档对新增 token 训练的价值。

使用 `Unique Hanzi` 而不是字符出现次数，是为了防止某个汉字在同一文档重复很多次后获得异常高分。

权重：

\[
\frac{1}{\sqrt{DF(c)+1}}
\]

表示：

- 越缺少上下文的新增汉字，权重越高；
- 已经非常常见的新汉字，权重自动降低；
- 平方根用于避免低频字符权重过度放大；
- `+1` 防止零频字符出现除零问题。

同时记录：

- `new_char_unique_count`
- `new_char_occurrence_count`
- `new_char_score`

---

# 6. 第四步：计算音形特征覆盖价值

除新增 token 外，还需要避免 PGCA 某些特征长期训练不足。

统计全候选语料中的：

- pinyin frequency
- shengmu frequency
- yunmu frequency
- tone frequency
- radical frequency
- structure frequency
- polyphone document frequency

定义文档特征分数：

\[
S_{feat}(d)
=
\sum_{f\in F(d)}
\frac{1}{\sqrt{DF(f)+1}}
\]

其中：

- \(F(d)\)：该文档中新增汉字对应的不同音形特征集合；
- \(DF(f)\)：包含特征 \(f\) 的文档数量；
- \(S_{feat}(d)\)：文档对低覆盖音形特征的补充价值。

这里主要用于打破新增汉字分数相近时的排序，不作为主目标。

---

# 7. 第五步：计算最终文档优先级

最终固定采用：

\[
S(d)=S_{new}(d)+0.25S_{feat}(d)
\]

解释：

- \(S_{new}\)：主目标，占主导；
- \(S_{feat}\)：PGCA 音形覆盖辅助目标；
- `0.25`：控制音形特征不能反过来支配数据选择。

因此该阶段本质仍然是：

> 新增 token 训练增强，而不是人为构造拼音/字形训练集。

---

# 8. 第六步：第一轮——覆盖优先选择

先不考虑 1B token 总量，优先解决新增 token 覆盖不足问题。

选择规则：

1. 优先覆盖 `Uncovered`；
2. 然后覆盖 `Low`；
3. 同等情况下选择 \(S(d)\) 更高的文档；
4. 同一个新增汉字达到目标文档数后，不再因为该汉字获得额外优先权。

目标覆盖标准：

| 指标 | 目标 |
|---|---:|
| 新增汉字至少出现 1 个文档 | ≥99% |
| 至少出现 20 个不同文档 | ≥95% |
| 至少出现 100 个不同文档 | ≥90% |

若个别 Extension-B+ 极端低频字自然语料不足，不强行满足 100 文档要求。

---

# 9. 第七步：处理自然语料无法覆盖的新增汉字

对 `Uncovered` 或严重低覆盖字符，按固定顺序扩大检索：

1. Wikisource
2. ect-krp
3. FineWeb-zhtw
4. Wikipedia
5. CCI3-HQ / WanJuan
6. 姓名、地名类公开自然文本

仍无法找到时允许少量构造 coverage sample。

人工/规则生成数据必须满足：

- 仅用于解决完全无上下文字符；
- 每个汉字只生成少量样本；
- 总量不超过增强数据的 **1%**；
- 不重复模板批量制造大量相似句子。

因此 1B 中至少 **99% 仍为自然文本**。

---

# 10. 第八步：第二轮——按优先级补齐到目标 token 数

覆盖要求满足以后，对剩余高质量文档按 \(S(d)\) 加权采样。

目标：

\[
T_{enhance}=0.05T_{Phase2}
\]

若：

\[
T_{Phase2}=20B
\]

则：

\[
T_{enhance}=1B
\]

文档仍需满足正常 Phase2 质量标准：

- 质量过滤通过；
- exact dedup 通过；
- MinHash 去重通过；
- benchmark decontamination 通过。

不能因为文档包含新增汉字就放宽质量要求。

---

# 11. 第九步：限制来源和领域偏置

增强池很容易被古籍、繁体数据垄断，因此加入固定约束：

| 类型 | 约束 |
|---|---:|
| 单一数据源 | ≤30% |
| 古汉语 | ≤30% |
| 现代自然中文 | ≥40% |
| 繁体中文 | ≥15% |
| 人工 coverage 数据 | ≤1% |

这些约束针对增强池内部，而不是整个 Phase2。

目的是让新增 token 同时学习：

- 现代上下文；
- 古汉语上下文；
- 繁体上下文；
- 不同领域上下文。

---

# 12. 第十步：多音字专项处理

对于新增 token 中的多音字，额外统计其上下文多样性。

例如一个字符有：

\[
P(c)=\{p_1,p_2,\ldots,p_k\}
\]

不在数据中写入“正确拼音标签”。

只选择多样化自然上下文，例如：

- 行业
- 银行
- 行为
- 行走

模型仍然只看到原始文本。

目的：

> 让 hidden state 通过 PGCA 自主选择 pronunciation candidate，而不是由训练数据直接提供答案。

---

# 13. 第十一步：去重并从原数据桶中移除

如果某篇文档被选入 `new_char_enhancement`：

> 必须从原来的 `zh_general`、`zh_knowledge` 等 bucket 中删除。

同一文档只能出现在最终 Phase2 mixture 一次。

否则所谓 5% 实际变成重复训练，而不是独立的数据组成。

流程顺序：

**增强池选择 → 全局 doc_id 去重 → 原 bucket 删除 → 最终 mixture 合并。**

---

# 14. 第十二步：最终精确 Tokenization

前面所有筛选都基于 Unicode 字符和文档统计，不需要执行 tokenizer。

只有增强候选最终确定后：

1. 使用最终 Char tokenizer；
2. batch encode；
3. 获取 exact token count；
4. 精确截取到 1B tokens；
5. 若不足，从 reserve candidate 中按 \(S(d)\) 顺序补充。

因此整个增强数据构建只进行一次精确 tokenization。

---

# 15. 最终输出

数据：

`phase2/new_char_enhancement/*.parquet`

主要字段：

| 字段 | 含义 |
|---|---|
| `doc_id` | 文档 ID |
| `input_ids` | 最终训练 token |
| `source` | 数据来源 |
| `domain` | modern/classical/etc. |
| `new_char_unique` | 新增汉字种类数 |
| `new_char_score` | \(S_{new}\) |
| `feature_score` | \(S_{feat}\) |

不保存 `pinyin_ids`、`stroke_ids` 等。

PGCA 在训练时通过模型内部 `feature_index` 直接从 `input_ids` 查询。

---

# 16. 最终验收标准

该阶段完成后必须满足：

| 指标 | 验收要求 |
|---|---:|
| 总量 | 1B tokens ±1% |
| 自然文本比例 | ≥99% |
| 新增汉字有自然上下文覆盖 | ≥99% |
| 新增汉字 ≥20 个文档覆盖 | ≥95% |
| 新增汉字 ≥100 个文档覆盖 | ≥90% |
| 单一数据源 | ≤30% |
| 现代中文 | ≥40% |
| 人工补充 | ≤1% |
| Exact duplicate | 0 |
| Benchmark contamination | 0 |
| feature index missing | 0 |

同时输出四份统计：

- `new_hanzi_frequency.parquet`
- `new_hanzi_selected_frequency.parquet`
- `new_hanzi_coverage_report.json`
- `feature_coverage_report.json`

---

## 最终流程

**确定新增单汉字 token → 全中文语料统计 TF/DF → 计算新增 Token 分数 → 统计 PGCA 音形覆盖 → 覆盖优先选样 → 补充极低覆盖字符 → 加权选样至 1B → 来源/领域约束 → 全局去重与去污染 → 从原 bucket 移除重复文档 → 一次精确 Tokenization → Coverage 验收。**

这一部分在论文中建议正式命名为：

**New Character Token and Phonetic-Glyph Coverage Enhancement**

其作用可以明确表述为：

> 针对词表重构中新引入的单汉字 token，通过高覆盖、多上下文的自然语料重采样，提高其语义表示学习充分度，并同步改善显式音形特征的训练覆盖。