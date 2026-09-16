# CC-LLM Data Factory V2 设计与实施方案

## 1. 目标与固定口径

Data Factory V2 为 CC-LLM 的语义对齐和全参数继续预训练提供统一的数据基础设施。Phase 1 与 Phase 2 共享来源注册、Parquet Cache、质量标签、Token Calibration、候选抽样、去重去污染、Tokenizer 和质量验收代码，只通过配置切换规模、数据配比、质量阈值和序列长度。

以下口径已经固定，后续实现和文档不得再使用其他版本：

- Phase 1：1B token，用于单汉字 tokenizer 与新 embedding/lm_head 的语义对齐。
- Phase 2：10B token，用于 PGCA 条件下的全参数继续预训练。
- 不再设计或统计“被裁多字 Token 桥接”数据，不保留独立桥接 bucket、覆盖指标或采样逻辑。
- Phase 1 和 Phase 2 的正式构建均执行候选级 SHA-256 精确去重、Benchmark 去污染和 DataTrove MinHash；另提供关闭 MinHash 的 `exact_only` 调试 Profile。
- 新增汉字增强的主实验只使用自然文本。人工 coverage 文本默认关闭，仅允许在独立消融配置中启用，且不超过增强数据的 1%。
- DataTrove 是数据读取、分片执行、断点恢复、过滤、统计、Parquet I/O 和 MinHash 的稳定底座；中文特化规则、阶段配额和最终训练格式由项目实现。

## 2. 对现有实现的判断

### 2.1 可以复用的能力

当前仓库已经实现了多项可直接迁移的业务逻辑：

| 现有模块 | 可复用内容 | V2 中的位置 |
| --- | --- | --- |
| `sources.py` | CCI3-HQ、WanJuan、FineWeb、外部 JSONL 字段适配 | DataTrove 自定义 Reader/Adapter |
| `text.py` | NFC、字符统计、混排识别、格式检查、PII/密钥规则 | Cheap Clean、Tag 和 Quality Filter |
| `source_cache.py` | 统一 Parquet Schema、任务级产物思想 | Canonical Cache，执行器替换为 DataTrove |
| `token_calibration.py` | 有界抽样、Rust Tokenizer 批量编码、Token 估算 | Calibration Pipeline Step |
| `sampling_plan.py` | 稳定文件排序、来源上限、候选安全余量 | Run Planner |
| `windowing.py` | 自然边界切分、代码围栏/公式/表格保护 | Final Windowing |
| `selection.py` | 父文档互斥、验证集预留、配额选择 | Mixture Policy |
| `fast_common.py` | 稳定 Hash、Parquet 辅助、报告结构 | V2 Common Utilities |

现有 `normalized/*.jsonl`、`cache_parquet/*.parquet` 和已采集外部 JSONL 均可作为迁移输入，不要求重新读取原始语料。复用前必须登记 Schema 版本、Tokenizer 版本、清洗配置和来源指纹；无法证明兼容时只复用其上游文本，不复用旧统计或去重状态。

### 2.2 必须替换的逻辑

以下实现不进入 V2 正式路径：

- `prepare.py -> dedup.py -> prescan.py -> sample.py` 的旧式全量 JSONL/SQLite 流程。
- 将全部文档偏移、Token 估计和选择状态写入单个 SQLite 数据库。
- 为抽取 1B/10B 数据而预扫描全部 Corpus，或为补采预先索引全部未选文档。
- 对同一文本重复执行 AutoTokenizer 或 Python 单条 Tokenize。
- 在训练数据中保存拼音、声母、韵母、声调、部首、笔画等 PGCA Feature IDs。
- `candidate_features.py` 中所有多字 Token 桥接匹配、计数和优先级逻辑。
- 把网页发现、GitHub 搜索、Stack Exchange 下载作为 Phase 1/2 的必经步骤。

旧路径在 V2 完成 10M、100M 和 Phase 1 全量验收前保持只读，不继续增加功能。验收通过后删除或移入 `legacy/`，避免两套正式入口长期并存。

## 3. 总体架构

```text
Source Registry
    -> File Manifest
    -> Canonical Parquet Cache
    -> Cheap Clean and Tag
    -> Token Calibration and Source Inventory
    -> Immutable Run Plan
    -> Candidate Materialization
    -> Candidate Quality Filtering
    -> Exact Dedup
    -> MinHash Dedup
    -> Benchmark Decontamination
    -> New-Character Enhancement Selection
    -> Final Mixture Selection
    -> Exact Tokenization
    -> Packing
    -> QA and Dataset Report
```

目录统一为：

```text
data/corpus/
  registry/
    sources.yaml
    manifests/<source>.parquet
  cache/<source>/*.parquet
  metadata/
    cache_stats/<source>.json
    calibration/<tokenizer_hash>/<source>.json
    dedup_registry.parquet
  runs/
    phase1/<run_id>/
      config.snapshot.yaml
      plans/
      candidates/
      quality_filtered/
      deduplicated/
      decontaminated/
      selected/
      tokenized/
      packed/
      validation/
      reports/
      logs/
    phase2/<run_id>/
      ...
```

`raw` 不复制到项目目录。注册表保存服务器原始绝对路径、数据版本、许可证和 Adapter。Cache 是跨阶段共享资产；从 Candidate 开始的产物全部按 Phase 和 `run_id` 隔离。

## 4. 数据契约

### 4.1 Canonical Document

DataTrove 内部统一使用 `Document(text, id, metadata)`。写入 Parquet 时展开为：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `doc_id` | string | 全局稳定 ID，由来源、子集、文件和原始 ID/行号组成 |
| `parent_doc_id` | string | 原始父文档 ID，窗口和 Chunk 继承该值 |
| `text` | large_string | 保留结构的规范化正文 |
| `source` | string | 稳定来源名 |
| `subset` | string/null | 数据集子集或站点 |
| `source_path` | string | 原始相对路径或可追踪路径 |
| `revision` | string/null | 数据版本、Commit 或快照日期 |
| `license` | string | 可核验许可证 |
| `url` | string/null | 原始 URL |
| `language` | string | `zh`、`en`、`multilingual` 等 |
| `domain` | string | `general`、`knowledge`、`code`、`math`、`scientific`、`structured` |
| `char_count` | int64 | Unicode 字符数 |
| `hanzi_count` | int64 | 汉字数 |
| `latin_count` | int64 | Latin 字符数 |
| `digit_count` | int64 | 数字数 |
| `quality_prior` | float/null | 上游质量分或来源先验 |
| `tags` | list[string] | `mixed`、`traditional`、`classical`、`long_doc` 等横向属性 |

候选级产物增加 `content_sha256`、`estimated_tokens`、`candidate_bucket` 和稳定采样键；Tokenize 后增加 `input_ids`、`token_count` 和 `tokenizer_hash`。PGCA Feature IDs 不进入数据 Schema，由模型内部 persistent feature index 根据 `input_ids` 查询。

### 4.2 文本规范化

训练文本和哈希文本分开生成。训练文本执行 UTF-8、Unicode NFC、统一换行、NUL/非法控制字符删除和明确模板噪声清理，保留繁简、异体、大小写、标点、Markdown、代码缩进、LaTeX 和表格结构。禁止 NFKC、繁转简和古文现代化。

哈希文本在训练文本基础上统一行尾空白和连续空行。任何会改变训练语义的清洗规则必须进入配置 Hash；规则变化只使对应 Cache Task 和下游产物失效。

### 4.3 数据可追踪性

每个输出 Shard 必须能够追踪到 Source Manifest。来源许可为 `unknown`、`NOASSERTION` 或存在冲突的文档只能进入隔离区，不得进入 Candidate。所有最终 Shard 记录 SHA-256、Token 数、文档数、来源和类别分布。

## 5. 数据配比

### 5.1 Phase 1：1B Token

| 一级类别 | 比例 | Token | 默认来源策略 |
| --- | ---: | ---: | --- |
| 中文通用 | 30% | 300M | CCI3-HQ 70%、WanJuan 30% |
| 中文高质量知识 | 20% | 200M | FineWeb-Edu-Chinese 为主 |
| 英文能力保持 | 15% | 150M | FineWeb-Edu-English |
| 中英混排/技术中文 | 10% | 100M | CCI3-HQ 40%、FineWeb-Edu-Chinese 40%、The Stack V3 README/docs 20% |
| 代码/数学/结构化 | 10% | 100M | The Stack V3 70%、OpenWebMath 30% |
| 新增汉字 Token/边界覆盖 | 15% | 150M | 从全部中文自然候选中覆盖优先选择 |

代码/数学/结构化内部默认按 40%/40%/20% 管理，正式配置允许调整，但不得在构建过程中静默回填。数据不足时 Run Report 标记 Shortfall，由新的来源配置或增量 Plan 解决。

Phase 1 以高质量短文本和稳定分布为主。自然文本窗口控制在 512-2048 token；长文、古文和繁体只统计，不设置硬占比。一个 `parent_doc_id` 只能属于一个一级类别，新增汉字增强文档入选后必须从原中文 Bucket 移除。

### 5.2 Phase 2：10B Token

| 一级类别 | 比例 | Token | 默认来源策略 |
| --- | ---: | ---: | --- |
| 中文高质量通用 | 30% | 3.0B | CCI3-HQ 50%、FineWeb-Edu-Chinese 35%、WanJuan 15% |
| 中文知识密集 | 25% | 2.5B | Chinese Cosmopedia 40%、中文 Wikipedia 25%、FineWeb-zhtw 20%、Wikisource 10%、ect-krp 不超过 5% |
| 英文/多语/中英混排 | 20% | 2.0B | FineWeb-Edu-English 55%、FineWeb2 多语 15%、中英技术混排 30% |
| 数学/代码/科学 | 15% | 1.5B | The Stack V3 45%、OpenWebMath 30%、peS2o 25% |
| 新增汉字 Token 与音形覆盖 | 10% | 1.0B | 从全部中文自然候选中覆盖优先选择 |

长文本与古汉语是横向属性，不单独占一级 Bucket：

- 长文本不少于总 Token 的 10%，即至少 1B token。
- 古汉语/文言目标为总 Token 的 8%-12%，即 0.8B-1.2B token。
- 长文本、古文、繁体、知识和新增汉字标签允许重叠，但每篇文档只计入一个一级类别和一次总 Token。

Phase 2 单一来源不得异常支配整体分布。一级类别内采用显式 Source Weight 和上限；来源不足时不自动放宽许可证、质量或跨类别约束。

## 6. Pipeline 分阶段设计

### Stage 0：Source Registry 与 File Manifest

`sources.yaml` 定义来源路径、格式、Adapter、许可证、Revision、允许 Phase、上游质量等级和默认 Domain。构建 File Manifest 时只读取文件系统元数据和 Parquet Footer，不扫描 JSONL 正文；记录绝对路径、大小、mtime、行数/Row Group、Archive Member 和可选文件 SHA-256。

DataTrove 负责文件 Sharding 和 Task 分发。项目为 WanJuan tar、CCI JSONL 等来源实现 `BaseDiskReader` 或 Adapter。单条 UTF-8/JSON 异常写入 Reject Report 后继续，单文件失败只使该 Task 失败。

### Stage 1：Canonical Parquet Cache

Cache 只执行字段提取、保守规范化、许可证校验、极端低质量过滤和廉价字符统计，不执行 Tokenize、MinHash、语义质量模型、序列切分或新增汉字全局统计。

DataTrove `LocalPipelineExecutor` 用于单机验证，远程正式任务使用 `SlurmPipelineExecutor`；Reader 输出标准 Document，经自定义 Formatter/Filter 后由 `ParquetWriter` 写入中等大小 Shard。每个输入文件对应稳定 Task，成功后由 completion marker 记录。重新运行只处理新增、变化或失败 Task。

Cache 采用增量策略：长期目标是完整缓存所有注册来源，但 Phase Run 不必等待整个数据湖缓存完成。只要当前已完成 Cache 的估计容量能满足配额和安全余量，即可生成 Run Plan；其余来源继续独立补齐。

### Stage 2：Cheap Clean、Tag 与 Source Inventory

全量 Cache 允许执行的操作仅限线性、无模型或轻模型的低成本处理：

- 长度、可打印字符、HTML/SEO、重复行、符号比例和文本完整性规则。
- LanguageFilter/FastText 语言识别。
- 中文、英文、中英混排、代码、数学、科学、结构化 Domain。
- 繁体、古文、长文档、多音字密集等横向标签。
- `quality_prior` 和来源统计。

不同 Domain 使用独立 Quality Profile。现代中文标点率和句长规则不得直接用于古籍；自然语言规则不得用于代码、JSON 或 LaTeX。CCI3-HQ、FineWeb-Edu、Wikipedia 等已有上游质量控制的来源只记录 Prior；WanJuan、普通 Web 和混排候选在 Candidate 阶段再执行较昂贵质量评分。

### Stage 3：Token Calibration

每个 `source x domain` 默认确定性抽取 50K 文档，可配置为 20K-100K。达到目标后必须停止读取该来源，不继续扫描剩余 Cache。使用最终 `tokenizer.json` 和 Rust `tokenizers.encode_batch()` 统计：

- Tokens/Document、Tokens/Character。
- 长度 P50/P90/P99。
- 质量过滤通过率。
- 各一级类别和横向标签的 Token 产率。
- 新增汉字文档命中率。

Calibration 结果按 `tokenizer_hash + cache_schema_hash + quality_profile_hash` 缓存。Tokenizer、清洗或分类规则不变时，Phase 1/2 和规模消融共用结果。

### Stage 4：File-level Run Plan 与 Candidate Sampling

Controller 根据目标配额、Source Weight、Calibration 产率、质量损耗估计和 Oversample Ratio 生成不可变 `plan-round-000.jsonl`。先用稳定 Hash 排序并选择足量 Cache Shard，最后一个过大 Shard 才使用 Document Hash Threshold。DataTrove 只读取 Plan 中列出的文件，不能把 `SamplerFilter` 直接应用到全部 Cache 后再扫描数据湖。

Phase 1 普通 Bucket 默认过采样 1.20 倍，新增汉字 Bucket 默认 1.35-1.50 倍；预计候选总量约 1.2B-1.25B token。Phase 2 普通 Bucket默认 1.15 倍，新增汉字 Bucket默认约 1.30 倍；预计候选总量约 11.5B-11.7B token。

Candidate Materialization 在单遍数据流中完成基础分类、横向标签和稳定采样。普通类别与新增汉字增强资格分别判定，Calibration 分别统计产率，Candidate 对两种抽样结果取并集且每篇文档只保存一次。仅最终入选增强池的父文档从普通桶排除；Phase 2 中 FineWeb 中文的 knowledge 标签允许用于配置指定的中文通用桶，不修改共享 Cache。

Plan 同时预留验证集容量和普通桶被增强选择占用的余量。Mixture/Finalize 输出按 Bucket × Source 的缺额后，可执行 `plan --round 1` 生成 `plan-round-001.json`；仅抽取前轮未使用的 Cache 文件。其后的 candidate、exact_dedup、minhash、decontaminate、mixture、tokenize、finalize 均传相同的 `--round`。去重合并各轮候选重新执行，避免增量重复；候选生成不重扫旧 Cache。无可用新分片时报告容量不足，不改变来源配比。

### Stage 5：Candidate Quality Filtering

Candidate 先经过来源对应的正式 Quality Profile。规则过滤始终启用；模型或 FastText 质量评分只应用于宽泛来源。高质量 Curated Source 可以跳过重模型评分，但仍执行许可证、乱码、重复模式、隐私和密钥检查。

Quality Step 输出保留和拒绝样本分片，并按来源统计输入 Token 估计、保留率和拒绝原因。任何类别保留率显著偏离 Calibration 时停止后续昂贵步骤，先生成补采 Plan。

### Stage 6：SHA-256 Exact Dedup

对哈希规范化文本计算 SHA-256。完全相同内容只保留一个代表文档；代表选择顺序为许可证明确、来源权威/Curated、质量分高、正文完整，最后使用稳定 Doc ID 打破平局。

DataTrove 负责分片读取、并行 Hash Step、统计、重复 ID 写出和回读过滤；项目负责规范化、SHA-256、跨来源代表策略和全局 Registry。可使用 xxhash 作为分区键或内存预筛，但最终必须核对 SHA-256，不能仅以短 Hash 作为可审计去重依据。

`dedup_registry.parquet` 只记录最终保留文档的 SHA-256、Doc ID、Source、Phase 和 Run ID。Phase 2 必须先查询 Phase 1 Registry，防止跨阶段重复训练。

### Stage 7：DataTrove MinHash Near-Dedup

正式 Pipeline 使用 DataTrove 四阶段 MinHash：

1. `MinhashDedupSignature` 生成签名。
2. `MinhashDedupBuckets` 按 LSH Bucket 聚合候选。
3. `MinhashDedupCluster` 生成近重复簇和删除 ID。
4. `MinhashDedupFilter` 回读完全相同的 Candidate 输入并删除重复文档。

自然中文使用 Unicode 字符 5-gram，英文自然文本使用词级 5-gram，代码使用词法 Token 5-gram。配置 threshold 分别为 0.80、0.80 和 0.85，表示 LSH 概率曲线的近似拐点，不是逐对验证的严格 Jaccard 截断。固定 num_buckets 后，由 `(1 / num_buckets) ** (1 / hashes_per_bucket)` 反推整数 hashes_per_bucket，并报告实际拐点。参数通过 Pilot 抽查调整。MinHash 只处理候选池。

DataTrove 默认解决召回、Bucket、聚类和过滤；项目仍负责语言对应 Shingle、阈值和质量感知代表选择。MinHash Filter 必须使用与 Signature 阶段完全一致的输入文件和 Task 数，恢复时不得改变 Task Sharding。

`exact_only` Profile 只允许用于代码调试和小规模性能定位，不能产出正式 Phase 1/2 数据集。

### Stage 8：Benchmark Decontamination

污染库至少包括 MMLU、C-Eval、CMMLU、ARC、GSM8K、HumanEval、MBPP，以及后续 CSC、CGEC、C3Bench、Fuxi 等 Dev/Test 数据。污染库冻结版本、来源和 Hash，并与 Run Config 一起归档。

去污染分两层：完整样本规范化 SHA-256 精确匹配；正文片段精确 n-gram/substring 匹配。英文和空格丰富文本采用词级 13-gram；中文采用更长的字符窗口和最小覆盖阈值，避免 13 个单汉字造成大量误杀。具体窗口和阈值必须在 Pilot 中抽查后固化，不允许运行时临时改变。

古文来源重点检查经典原文重叠。污染命中样本写入独立 Audit 文件，不进入后续选择。

### Stage 9：新增汉字 Token 与音形覆盖增强

增强目标集合只来自新词表中新增的单汉字 Token，基于 `new_hanzi_token_ids.json` 与 Feature Index 构建 `new_hanzi_tokens.parquet`。不再读取或使用被裁多字 Token 清单。

在去重、去污染后的中文 Candidate Pool 中统计每个新增汉字的 TF、DF、Source DF、现代/古文/繁体分布、多音字属性和音形 Feature 覆盖。文档主分数为：

\[
S_{new}(d)=\sum_{c\in U(d)\cap V_{new}}\frac{1}{\sqrt{DF(c)+1}}
\]

其中 `U(d)` 使用不同汉字集合，防止单字在同一文档重复出现导致分数虚高。音形 Feature 的逆文档频率只作为次级排序信号，默认权重 0.25，不能支配新增 Token 目标。

选择分两轮：先使用 Coverage-first Greedy 选择补齐 Uncovered 和 Low 字符，再按综合分数和 Source/Domain 约束补齐 Token 配额。Phase 1 目标 150M，Phase 2 目标 1B。入选增强池的 `parent_doc_id` 从其他一级 Bucket 移除。

主实验不生成自然语料无法覆盖的字符。报告同时给出全体新增汉字覆盖率和自然候选可观测集合覆盖率，不能通过缩小分母掩盖缺口。人工 Coverage 仅存在于单独消融 Profile，比例不超过增强池 1%，并单独标记 Source 和 Synthetic Tag。

增强池内部约束：以既定 source_weights 为准，Phase 1 单一来源上限为 40%，Phase 2 为 30%；配置加载时拒绝权重超过上限。现代自然中文不少于 40%，Phase 2 繁体中文不少于 15%，古汉语不超过增强池 30%。覆盖目标 99%/95%/90% 仅作诊断，不影响 passed，也不会单独触发无限补采。报告保留全体新增汉字分母及候选/选中覆盖。多音字不写入拼音标签。

### Stage 10：Final Mixture 与验证集

Mixture Policy 使用真实或校准 Token 数、稳定采样键和父文档互斥约束完成选择。验证集必须在训练集选择前按 `parent_doc_id` 稳定切分，并与训练集执行相同的质量、去重和去污染流程。

最终选择满足一级 Bucket、Source Sub-quota 和横向属性三层约束。横向属性只统计，不重复增加总 Token。配额不足时输出 Shortfall 并生成增量 Plan，不复制样本、不降低许可证要求、不从错误类别静默回填。

Phase 1 验证集同时包含自然分布集和新增汉字诊断集；Phase 2 验证集按一级类别、长度档位、古文和繁体分层，并单独保留长文本诊断集。

### Stage 11：Exact Tokenization

最终选中文档只执行一次正式 Tokenization。使用模型目录中的 `tokenizer.json` 和 Rust `tokenizers.encode_batch()`，不在热路径逐条调用 AutoTokenizer。输出至少包含 `doc_id`、`parent_doc_id`、`input_ids`、`token_count`、`source`、`bucket` 和 `tokenizer_hash`。

Canonical Text Shard 与 Tokenized Parquet 同时保留：文本产物用于审计、复现和框架兼容，Tokenized 产物用于避免训练时重复编码。在确认 ms-swift 可以稳定跳过 Preprocessor 并直接消费 `input_ids` 前，不能只保留 Tokenized 数据。

### Stage 12：Packing

Packing 直接操作 Token IDs。不同父文档之间插入 EOS；同一长文优先保持连续 Chunk 顺序；代码块、公式和结构化对象在 Windowing 阶段保护。Phase 1 使用 1K/2K 长度，Phase 2 使用 2K/4K/8K Profile，并保证长文档配额与实际训练 `max_length` 一致。

Packed Shard 不替代未 Packed Tokenized 数据。训练 Sequence 配置变化时只重跑 Packing，不重跑清洗、去重或 Tokenization。

### Stage 13：QA 与 Dashboard

每次 Run 输出统一 `dataset_report.json` 和可视化 Dashboard 数据，至少覆盖：

- 目标/实际 Token、一级类别、Source 和横向属性比例。
- 各阶段输入、保留、拒绝数量及原因。
- SHA-256 重复数、MinHash Cluster、删除率和来源影响。
- 各 Benchmark 污染命中和删除数量。
- 语言、领域、繁体、古文和长度分布。
- 新增汉字 TF/DF、选中前后覆盖、自然上下文缺口和 Feature 缺失率。
- 文档长度与 Packed Sequence 长度 P50/P90/P99。
- 各 Stage Documents/s、MB/s、Tokens/s、峰值内存和预计剩余时间。

Dashboard 只读取报告产物，不直接扫描训练数据。报告缺失或任一硬门禁失败时，Run 状态为 `passed=false`。

## 7. DataTrove 与项目代码边界

| 能力 | DataTrove | CC-LLM 自定义代码 |
| --- | --- | --- |
| 执行器 | Local、Slurm、Task Sharding、Completion、Stats | Run Controller、配置依赖图 |
| 输入输出 | JSONL/Parquet Reader、Parquet/JSONL Writer、fsspec | WanJuan tar 和来源字段 Adapter |
| 清洗过滤 | BaseFilter、Formatter、LanguageFilter | 中文、古文、混排、代码和格式规则 |
| 抽样 | SamplerFilter、Shard 执行 | File-level Plan、来源/类别配额 |
| 精确去重 | Pipeline 执行和过滤 | SHA-256、Registry、代表选择 |
| 近似去重 | MinHash Signature/Bucket/Cluster/Filter | Shingle、阈值、质量感知代表策略 |
| 去污染 | Exact Substring/Decont 执行能力 | Benchmark Registry 和中英文匹配规则 |
| Token/Stats | Token Block、Stats 框架 | Char Tokenizer Calibration、覆盖指标 |
| 增强与混合 | 自定义 Step 的执行框架 | 新增汉字评分、Greedy、Mixture Policy |
| Packing | 数据流和 Writer | EOS、长文连续性、ms-swift Contract |

DataTrove 固定使用 `datatrove[io,processing]>=0.10,<0.11`，Python 3.10+。版本升级必须先通过 10M Integration Run，不直接在正式 Run 中浮动依赖版本。

## 8. 配置与执行门面

代码只保留一个入口：

```bash
python -m scripts.data_factory.run \
  --config scripts/data_factory/configs/phase1.yaml \
  --stage all \
  --executor slurm \
  --resume
```

支持的 Stage 为 `manifest`、`cache`、`tag`、`calibrate`、`plan`、`candidate`、`quality`、`exact_dedup`、`minhash`、`decontaminate`、`enhance`、`select`、`tokenize`、`pack`、`qa` 和 `all`。Phase 2 只替换配置文件。

配置拆为：

```text
scripts/data_factory/
  common/
  readers/
  filters/
  dedup/
  selection/
  tokenization/
  reports/
  configs/
    sources.yaml
    phase1.yaml
    phase1_exact_only.yaml
    phase2.yaml
    phase2_exact_only.yaml
  run.py
```

每个 Run 保存解析后的完整配置快照。Run ID 由 Phase、目标规模、配置 Hash、Tokenizer Hash 和 Source Manifest Hash 组成。配置内容变化创建新 Run，不覆盖旧 Run。

## 9. 并行、恢复与状态失效

一个中等大小输入文件或 Row Group 对应一个 Task。DataTrove `logging_dir` 为每个 Stage 独立保存 Executor 配置、Task Log、Completion 和 Stats。重新启动只运行未完成 Task；同一 MinHash Job 恢复时 Task 总数保持不变。

状态失效规则：

- Worker 数、日志级别和批大小变化不使数据语义产物失效。
- 输入文件内容、Adapter、规范化和质量规则变化使对应 Cache Task 及下游失效。
- Tokenizer 变化从 Calibration 开始失效。
- 配额和 Source Weight 变化从 Plan 开始失效。
- SHA/MinHash/污染库变化从对应 Dedup/Decontamination Stage 开始失效。
- Sequence Length 变化只使 Packing 失效。

`--overwrite` 必须指定 Stage 和 Run，不能默认删除整个 Corpus。临时文件写入 Task 私有目录，成功后原子提交。坏记录写 Reject Shard；单文件失败不能回滚其他 Task。

## 10. 性能门禁

任何正式构建前依次执行 10M 和 100M Token Pilot。报告每个 Stage 的 MB/s、Documents/s、Tokens/s、CPU、内存和存储吞吐，并根据 Plan 字节数估计全量时长。实测吞吐与估算偏差超过 30% 时停止放大，先修正 Source Inventory 或实现。

必须满足：

- Calibration 对每个 `source x domain` 的扫描量有硬上限，达到样本目标后立即停止。
- Candidate 只读取 Plan 文件，不扫描完整 Cache。
- MinHash、去污染和正式 Tokenize 只处理 Candidate/Selected 数据。
- 最终文档只精确 Tokenize 一次。
- Pipeline 不创建文档级全局 SQLite。
- 中断恢复只重跑未完成 Task。

## 11. 验收条件

### 11.1 通用门禁

- 总 Token 和一级类别误差不超过 1%。
- 未知/冲突许可证为 0。
- SHA-256 Exact Duplicate 为 0。
- 已知 Benchmark Contamination 为 0。
- 训练/验证 `parent_doc_id` 和 `content_sha256` 交集为 0。
- PGCA Feature Index 对训练中目标汉字 Token 的缺失率为 0。
- Manifest、配置、Tokenizer、输出 Shard Hash 和完整报告齐全。

### 11.2 新增汉字诊断

- 新增汉字至少具有一个自然上下文的目标覆盖率为 99%。
- 至少 20 个不同文档覆盖的目标比例为 95%。
- 至少 100 个不同文档覆盖的目标比例为 90%。
- 上述三项仅作诊断参考，不参与 passed；未达到时记录自然候选与选中频率，不允许主实验自动生成文本补足。
- 人工消融数据比例不超过增强池 1%，并与自然数据分开统计。

### 11.3 Phase 2 附加门禁

- 长文本至少占 10%。
- 古汉语/文言占 8%-12%。
- 新增汉字增强池单一来源不超过 30%，现代中文不少于 40%，繁体中文不少于 15%。
- MinHash 完整执行并报告 Cluster 与删除比例。

Mixture 和 Finalize 分别按估算/真实 Token 检查来源配比（允许一个百分点误差）、专项子类及全局横向属性。Finalize 在验证集隔离和裁剪后重新统计训练汉字覆盖。失败报告保留在磁盘，CLI 非零退出，tokenize 拒绝消费失败或其他 Plan 的 Mixture。Schema 从输入继承，tokenize 显式追加 Token 字段。

本次修复升级 Pipeline identity，旧 Run 留作审计。Stack V3 Cache 的子文件清洗错误曾提前终止整个仓库展开，现改为只拒绝单文件，需单独重建 Stack V3 Cache（adapter_revision=2）；其余来源契约不变时复用。两阶段从 calibrate 开始重跑。ect-krp 的既定配额仍需足量真实数据，约 5M 的现有容量不足以满足 175M 配额，不能以软化覆盖门槛绕过容量约束。

## 12. 实施顺序

### Milestone 1：冻结口径与公共配置

新增统一 YAML Schema、Source Registry、Run ID 和依赖 Hash；把 Phase 1 配置改为新 30/20/15/10/10/15 配比，删除桥接字段和代码。为 Phase 2 建立固定 10B 配置。

### Milestone 2：DataTrove Cache 与 Tag

实现来源 Reader/Adapter、Canonical Formatter、Cheap Filter/Tag 和 Parquet Writer；兼容导入现有 Cache。用 Local Executor 完成小数据测试，再切换 Slurm。

### Milestone 3：Calibration、Plan 与 Candidate

迁移现有有界 Calibration 和 File-level Sampling Plan；补充 `source x domain` 产率、增量 Plan 和 Candidate 单遍处理。确认不再扫描 Plan 外 Cache。

### Milestone 4：Dedup 与 Decontamination

实现 SHA-256 Registry 和质量感知代表选择；接入 DataTrove MinHash 四阶段；建立 Benchmark Registry 和中英文精确片段去污染。

### Milestone 5：新增汉字增强与 Mixture

实现新增汉字/Feature 频率统计、Coverage-first Greedy、加权补齐、来源/领域约束和父文档互斥。删除全部桥接逻辑和报告字段。

### Milestone 6：Tokenize、Packing 与 ms-swift

并行执行唯一一次 Exact Tokenization，输出 Text 和 Tokenized 双产物；实现 1K/2K 与 2K/4K/8K Packing，并验证 ms-swift 的 Dataset/Preprocessor 接口。

### Milestone 7：放大与淘汰旧路径

依次通过 Unit Test、10M、100M、Phase 1 1B，再运行 Phase 2 10B。Phase 1 全量通过后淘汰旧 SQLite/Prescan/Sample 正式入口。

## 13. 参考实现与依据

- [DataTrove](https://github.com/huggingface/datatrove)：Document Pipeline、Reader/Filter/Writer、Local/Slurm Executor、Completion 和 Stats。
- [DataTrove MinHash 示例](https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py)：Signature、Bucket、Cluster、Filter 四阶段近似去重。
- [DataTrove Token Estimation](https://github.com/huggingface/datatrove/blob/main/examples/estimate_tokens.py)：小样本估算 Token 密度和子集采样率。
- [FineWeb Pipeline](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py)：过滤、MinHash 和分阶段执行的公开实现。
- [DataComp-LM](https://github.com/mlfoundations/dclm)：大规模数据过滤、混合、Rust 去重和质量评估。
- [Dolma](https://github.com/allenai/dolma)：十亿文档并行处理、过滤、去重与 Mixer。
- [The Stack](https://arxiv.org/abs/2211.15533)：代码许可证、隐私处理和近重复去重。
