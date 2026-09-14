# CC-LLM 统一预训练数据构建方案

## 1. 总体目标与设计原则

CC-LLM 的 Phase1 语义对齐与 Phase2 全参数继续预训练采用**同一套数据构建 Pipeline**。两阶段不维护独立脚本，只通过配置文件调整数据规模、来源配比、质量阈值、增强比例和序列长度。

统一技术栈如下：

| 功能 | 工具 |
|---|---|
| 数据读取、流水线编排、并行、采样、MinHash、统计 | **DataTrove** |
| 中间数据格式 | **PyArrow / Parquet** |
| 最终精确 Tokenization | **HuggingFace tokenizers** |
| 语言识别 | **DataTrove LanguageFilter / FastText** |
| 哈希与精确去重 | **xxhash** |
| 高速 JSON | **orjson** |
| 编码异常修复 | **ftfy**，仅处理明确乱码 |
| 集群执行 | DataTrove `SlurmPipelineExecutor`；本地调试使用 `LocalPipelineExecutor` |

整个 Pipeline 固定为：

> **Raw Corpus → Canonical Parquet Cache → 基础清洗与分类 → Token Calibration → Candidate Sampling → 候选质量筛选 → Exact/MinHash 去重 → Benchmark 去污染 → 新增汉字 Token 与音形增强 → Final Mixture → Exact Tokenization → Packing → QA**

核心原则有三点：

1. **原始数据只解析一次。** 所有数据源第一次转换成统一 Parquet Cache，之后 Phase1、Phase2 以及消融实验全部复用。
2. **昂贵操作后置。** 全量语料只执行廉价的清洗、统计和分类；MinHash、语义质量模型、去污染和精确 Tokenization 仅对目标规模附近的候选池执行。
3. **最终文本只精确 Tokenize 一次。** PGCA 的拼音、声韵调、部首、笔画等特征不写入训练数据，而由模型内部 feature index 根据 `input_ids` 在线查询。

## 2. 统一数据构建 Pipeline

Phase1 和 Phase2 共用完全相同的数据处理代码与 Corpus Cache，仅通过配置文件调整目标 token 数、数据配比、质量阈值、增强比例和序列长度。

整体流程固定为：

**Raw → Canonical Cache → Cheap Clean & Tag → Token Calibration → Candidate Sampling → Candidate Quality Filtering → Exact Dedup → MinHash → Decontamination → New-Token Enhancement → Mixture Selection → Exact Tokenization → Packing → QA**

| 阶段 | 实际操作 | 执行范围 | 主要工具 |
|---|---|---|---|
| ① Source Cache | 原始 JSONL/Parquet/tar 等统一字段、基础解析并转换为 Parquet Shards | **全量，仅首次执行** | DataTrove + PyArrow |
| ② Cheap Clean & Tag | Unicode/乱码清理、Web 噪声过滤；语言、领域、中英混排、繁体、古文、长文档等标签生成 | **全量** | DataTrove + 自定义 Filter / LanguageFilter |
| ③ Token Calibration | 每个 `source × category` 抽约 50K 文档，用最终 Char Tokenizer 估算 token 密度与长度分布 | **小样本** | HF `tokenizers` |
| ④ Candidate Sampling | 根据当前 Phase 的 source/category 配额和 estimated tokens 确定性过采样 | **全量扫描，但仅廉价操作** | DataTrove `SamplerFilter` |
| ⑤ Quality Filtering | 对 WanJuan、普通 Web、中英混排等宽泛来源候选做进一步语义质量评分；高质量 curated 数据跳过重模型评分 | **仅 Candidate Pool** | FastText / 小型 Quality Classifier + 来源先验 |
| ⑥ Exact Dedup | 对 `normalized_text` 计算哈希，删除完全重复文档；跨源重复保留高质量版本 | **仅 Candidate Pool** | `xxhash` |
| ⑦ MinHash Dedup | 删除转载、镜像、轻微格式修改等近似重复文本 | **仅 Candidate Pool** | DataTrove MinHash |
| ⑧ Decontamination | 与 MMLU、C-Eval、CMMLU、CSC、CGEC、C³Bench、Fùxì 等评测集进行 Exact / n-gram 去污染 | **仅 Candidate Pool** | DataTrove / 自定义 n-gram |
| ⑨ New-Token Enhancement | 统计新增单汉字 Token 的文档覆盖率，结合音形 Feature Coverage 进行覆盖优先与加权重采样 | **中文 Candidate Pool** | 自定义 DataTrove Step |
| ⑩ Mixture Selection | 根据 Phase 配置精确满足 source/category/token quota，并执行 source/domain 横向约束 | **去重后数据** | 自定义 Mixture Policy |
| ⑪ Exact Tokenization | 使用最终 HF-compatible Char Tokenizer 对选中文档执行**唯一一次正式 Tokenization** | **最终约 1B / 10B 文本** | HF `tokenizers` |
| ⑫ Packing | Token-level 截断、EOS 隔离、连续文档切分与 1K/2K/4K/8K Packing | **Token Level** | DataTrove / 自定义 Packing |
| ⑬ QA | 统计来源、类别、质量、重复率、污染率、新增汉字覆盖、音形 Feature Coverage、长度分布等 | **最终数据** | DataTrove Stats + 自定义统计 |

其中最关键的效率原则是：

> **全量 Corpus 只执行一次 Cache、廉价清洗和统计；语义质量模型、MinHash、去污染和精确 Tokenization 等高成本步骤全部后置到 Candidate Pool。Phase1 约只处理 1.2B 候选 tokens，Phase2 约只处理 11.5B 候选 tokens，而不是对整个原始数据湖执行昂贵操作。**

### 2.1 Corpus Cache、清洗与分类

所有源数据第一次进入系统时，通过 DataTrove Reader 转换成统一 Document，再写入 Parquet Cache。推荐保存：

```text
doc_id
text
source
subset
license
url
language
domain
char_count
hanzi_count
latin_count
digit_count
traditional
classical
long_doc
quality_prior
normalized_hash
```

Raw 数据仅在这一阶段读取一次。CCI JSONL、WanJuan 压缩包、FineWeb Parquet、The Stack 等源数据全部转换为中等大小的 Parquet shards，之后不再反复解压和解析。

文本规范化采用**保守策略**：统一换行、删除 NUL 和非法控制字符、清除明确 HTML/导航噪声和乱码；禁止繁转简、古文现代化、全量 NFKC 和异体字统一，因为这些差异本身属于 CC-LLM 希望建模的汉字信息。

文档随后完成语言与类型标注。`zh`、`en`、`zh_en_mixed`、`code`、`math`、`scientific` 作为主要 domain；`traditional`、`classical`、`long_doc`、`polyphone_dense` 等作为可叠加标签。中英混排通过汉字比例、Latin 比例和技术信号联合判断，不再依赖单独大规模抓取 GitHub 网页。

质量控制分为两层。第一层是廉价规则过滤，包括文档长度、重复行、重复 n-gram、字符/符号比例、HTML/SEO 噪声、语言置信度和文本完整性。现代 Web、古文、代码和科学文档分别使用独立的 quality profile，避免用现代中文的标点率或平均句长误删古籍。

第二层是语义质量评分，但**不对全部数据执行**。CCI3.0-HQ、FineWeb-Edu、Wikipedia、Cosmopedia、ect-krp 等已有较强上游质量控制的数据只记录 `quality_prior`；WanJuan、普通 FineWeb 和中英混排等宽泛来源，在进入 Candidate Pool 后才运行额外质量评分。这样可以避免把大量算力浪费在最终不会被采样的数据上。

### 2.2 Token Calibration 与 Candidate Sampling

不能先精确 Tokenize 整个数据湖。对每个 `source × domain` 随机抽取约 50K 文档，使用最终 Char Tokenizer batch encode，统计平均 tokens/document、tokens/character 以及 P50/P90/P99 长度。

若数据源 \(s\) 包含 \(N_s\) 个候选文档，小样本平均 token 数为 \(\bar T_s\)，则其总 token 规模估计为：

\[
\hat T_s=N_s\bar T_s
\]

其中：

- \(N_s\)：数据源可用文档数；
- \(\bar T_s\)：Calibration 样本的平均 token 数；
- \(\hat T_s\)：该数据源估计可用 token 总量。

若最终希望从该来源获得 \(T_s\) tokens，则采样概率为：

\[
p_s=
\min\left(
1,\frac{T_s}{\hat T_s}r
\right)
\]

其中：

- \(T_s\)：最终来源配额；
- \(\hat T_s\)：估计可用 token 数；
- \(p_s\)：候选文档采样概率；
- \(r\)：过采样系数，用于补偿后续质量过滤、去重和去污染造成的损失。

固定采用：

- Phase1：`r = 1.20`
- Phase2：`r = 1.15`

因此 1B Phase1 只有约 **1.2B estimated tokens** 进入昂贵处理，10B Phase2 只有约 **11.5B estimated tokens** 进入昂贵处理，而不是对整个几十甚至上百 B 的 Corpus 做 MinHash 和 Tokenization。

这是整个 Pipeline 的核心效率设计。

### 2.3 去重、去污染与新增汉字增强

候选池首先执行 Exact Dedup。对规范化文本计算 `xxhash`，完全相同的文档只保留一个版本。跨来源重复时按照“权威/人工整理源 > 高质量 curated corpus > 普通 Web > synthetic”的优先级保留，例如 ect-krp 优先于网页转载，Wikipedia 优先于其镜像页面。

随后执行 DataTrove MinHash Near-Dedup，处理转载、镜像和轻微格式修改文本。MinHash 只处理约 1.2B/11.5B candidate，而不是全量 Raw Corpus。

去重后执行 Benchmark Decontamination。建立统一污染库，包括 MMLU、C-Eval、CMMLU、ARC、GSM8K、HumanEval/MBPP，以及后续 CSC、CGEC、C³Bench、Fùxì 等 dev/test 集。第一版采用 normalized exact match 与 13-gram overlap 即可。古汉语必须重点检查，因为 Wikisource、ect-krp 与 C³Bench/Fùxì 可能共享经典原文。

最后执行 CC-LLM 特有的**新增汉字 Token 与音形覆盖增强**。这里不再使用“Rare Hanzi”作为核心概念，而直接针对词表重构中新增加的单汉字 token。

定义：

\[
V_{\text{new}}
=
\{c\mid c\text{ 是新词表中新增的单汉字 Token}\}
\]

对于每个新增汉字 \(c\)，统计包含它的不同文档数量 \(DF(c)\)。对于文档 \(d\)，定义新增 Token 覆盖价值：

\[
S_{\text{new}}(d)=
\sum_{c\in U(d)\cap V_{\text{new}}}
\frac{1}{\sqrt{DF(c)+1}}
\]

其中：

- \(d\)：当前文档；
- \(U(d)\)：文档中出现的不同汉字集合；
- \(V_{\text{new}}\)：新增单汉字 Token 对应的汉字集合；
- \(DF(c)\)：包含汉字 \(c\) 的不同候选文档数；
- \(S_{\text{new}}(d)\)：该文档为新增 Token 提供有效训练上下文的价值。

文档频率较低的新 Token 权重更高。拼音候选、多音字、部首、笔画、结构等 feature coverage 作为次级排序信号。

筛选分两轮：先进行 coverage-first selection，优先覆盖没有或只有少量自然上下文的新 Token；达到最低覆盖后，再按 `S_new + feature_score` 加权采样至目标配额。Phase1 选 20M，Phase2 选 500M。

增强数据必须以自然文本为主体；无法获得自然上下文的极少数字符才允许生成少量 coverage sample。某文档一旦被选入增强池，就从原 `zh_general` 或 `zh_knowledge` 中移除，避免同一文本重复训练。

### 2.4 Final Mixture、Tokenization 与 Packing

完成去重、去污染和增强之后，根据 Phase Profile 精确选择最终 mixture。此时再使用最终 HF-compatible Char Tokenizer 对选中文档进行**唯一一次正式 Tokenization**。

最终训练数据不保存：

```text
pinyin_ids
shengmu_ids
yunmu_ids
tone_ids
radical_ids
stroke_ids
```

因为 PGCA checkpoint 已经内置 feature index，训练时直接执行：

> `input_ids → feature lookup → feature memory`

Tokenization 后所有长度截断和 Packing 都直接操作 token IDs，不再重新调用 tokenizer。

Phase1 主要采用 1K/2K sequence；Phase2 采用 2K/4K/8K 混合 sequence。对于长文档优先保持同一文档连续 chunk，不应先打碎后随机重新拼接。不同文档之间必须插入 EOS。

---

## 4. 两阶段统一配置

工程上只维护一个：

```text
scripts/data_factory/
├── common/
├── pipeline/
├── configs/
│   ├── phase1.yaml
│   └── phase2.yaml
└── run.py
```

Phase1/Phase2 的区别全部集中在配置：

| 配置项 | Phase1 | Phase2 |
|---|---:|---:|
| `target_tokens` | 1B | 10B |
| `oversample_ratio` | 1.20 | 1.15 |
| `mixture` | Phase1 配比 | Phase2 配比 |
| `semantic_quality` | 较轻 | 较严格 |
| `exact_dedup` | 开启 | 开启 |
| `minhash_dedup` | 开启 | 开启 |
| `decontamination` | 开启 | 开启 |
| `new_char_enhancement` | 20M / 2% | 500M / 5% |
| `long_doc_sampling` | 弱化 | 开启 |
| `traditional/classical constraint` | 仅统计 | 参与 Mixing |
| `sequence_length` | 1K/2K | 2K/4K/8K |

执行方式完全一致：

```bash
python -m scripts.data_factory.run \
  --config scripts/data_factory/configs/phase1.yaml
```

或：

```bash
python -m scripts.data_factory.run \
  --config scripts/data_factory/configs/phase2.yaml
```

Phase1 与 Phase2 共用 `raw/`、`cache/`、数据源统计以及 Calibration 结果；只有 candidate、selected、tokenized 和 packed 产物按不同 run 隔离。

---

## 5. 数据目录、产物与验收

推荐最终目录：

```text
data/
├── raw/                  # 原始数据，只读
├── cache/                # 全阶段共享 Parquet Corpus
├── stats/                # source/token/汉字统计
├── candidates/
│   ├── phase1/
│   └── phase2/
├── dedup/
├── selected/
├── tokenized/
└── packed/
    ├── phase1/
    └── phase2/
```

每次构建必须生成独立 `dataset_report.json`，至少覆盖以下维度：

| 维度 | 验收内容 |
|---|---|
| Token | 总 token、各类别和各 source 实际占比 |
| Quality | 各阶段过滤数量、质量分数分布 |
| Dedup | Exact 删除量、MinHash cluster 和去重率 |
| Contamination | 各 benchmark 命中/删除数量 |
| Language | zh/en/multilingual/mixed 分布 |
| Chinese | 新增 Token 覆盖、繁体/古文比例、PGCA feature 缺失率 |
| Length | 文档长度 P50/P90/P99、最终序列长度分布 |
| Source | 单一来源比例、许可和数据可追溯性 |

新增汉字增强额外输出：

```text
new_hanzi_frequency.parquet
new_hanzi_selected_frequency.parquet
new_hanzi_coverage_report.json
feature_coverage_report.json
```

建议基本验收要求为：

- 最终 token 规模误差 ≤1%；
- Exact duplicate = 0；
- 已知 benchmark contamination = 0；
- 新增单汉字 Token 至少拥有一个自然上下文的覆盖率 ≥99%；
- PGCA feature index 对训练中显式单汉字 Token 的缺失率为 0；
- Phase2 单一来源不应异常支配总体 mixture。

---

## 6. 数据构建效率设计

之前构建 1B 数据需要数天，核心问题并不是 1B token 本身，而是大量昂贵工作发生在最终不会进入训练集的文档上。最终 Pipeline 必须避免以下行为：

> 全量 JSONL → 全量 SQLite → 全量 MinHash → 全量精确 Tokenize → 最后选 1B。

正式流程应变为：

> **全量数据只做一次 Parquet Cache + 廉价 metadata → 小样本 Token Calibration → 先按目标比例抽到 1.15–1.20 倍候选 → 仅候选执行质量模型、MinHash、去污染 → 最终文档只 Tokenize 一次。**

对 Phase2 10B 来说，大致数据规模变化应是：

```text
几十/上百 B Raw Corpus
          ↓ cheap processing
约 11.5B Candidate
          ↓ quality / dedup / decontamination
约 10～11B Valid Corpus
          ↓ mixture
10B Selected
          ↓ exact tokenizer
10B Training Tokens
```

在工程层面，SQLite 不再保存整个文档库，只可用于 run 状态或少量索引；真正的大规模文档数据全部使用 Parquet shard。DataTrove 负责 task 级并行和断点恢复，服务器支持 Slurm 时直接使用 `SlurmPipelineExecutor`，各数据源/shard 独立运行。Tokenization 使用 Rust `tokenizers.encode_batch()`，避免 Python 逐文档调用 `AutoTokenizer`。

最终这套架构实现了三个目标：

> **一次 Cache、多阶段复用；先采样再做昂贵处理；一次 Tokenization 直接产出训练数据。**

因此 Phase1、Phase2 以及之后的数据规模消融都只需要调整配置，而不需要重新设计或重新跑完整的数据基础设施。