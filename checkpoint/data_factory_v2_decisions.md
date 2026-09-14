# Data Factory V2 关键决策

## 已确认口径

- Phase 1 固定为 1B token，配比为：中文通用 30%、中文高质量知识 20%、英文 15%、中英混排/技术中文 10%、代码/数学/结构化 10%、新增汉字 Token/边界覆盖 15%。
- Phase 2 固定为 10B token，配比为：中文高质量通用 30%、中文知识密集 25%、英文/多语/中英混排 20%、数学/代码/科学 15%、新增汉字与音形覆盖 10%。
- 完全删除“多字 Token 桥接”数据概念，不保留独立 Bucket、覆盖指标或采样代码。
- 长文本与古汉语是 Phase 2 横向属性，分别约束为至少 10% 和 8%-12%，不重复计入一级类别。
- Phase 1/2 正式构建均执行候选级 SHA-256、Benchmark 精确片段去污染和 DataTrove MinHash；`exact_only` 只用于调试。
- 新增汉字主实验只用自然文本。人工 Coverage 默认关闭，仅用于独立消融且不超过增强池 1%。

## 工程决策

- DataTrove 作为 Local/Slurm 执行、分片、完成标记、统计、Parquet I/O 和 MinHash 的稳定底座。
- 项目代码负责来源 Adapter、中文/古文/混排规则、新增汉字评分、阶段配额、去污染规则和 ms-swift 输出。
- Cache 跨阶段共享并增量构建；Candidate 必须由 File-level Plan 限定，不能直接扫描完整 Cache。
- MinHash、质量模型、去污染和正式 Tokenize 仅作用于 Candidate/Selected 数据。
- 最终文档只正式 Tokenize 一次；同时保留 Canonical Text 和 Tokenized Parquet，PGCA Feature IDs 不写入数据。
- 旧 JSONL/SQLite/Prescan/Sample 流程在 V2 完成 Phase 1 验收后淘汰。

完整方案见 `agent/blueprint/data_build/data_factory.md`。
## Milestone 1 状态（2026-09-06）

已完成 V2 配置底座：

- 新增 `scripts/data_factory/v2/config.py`，统一加载 Phase、Source Registry、Bucket、去重、增强和序列配置。
- 新增 `scripts/data_factory/configs/sources.yaml`、`phase1.yaml` 和 `phase2.yaml`。
- 强制 Phase 1=1B、Phase 2=10B，并校验 Bucket 与 Source 权重总和。
- 配置层拒绝任何 Bridge Bucket；主实验固定自然新增汉字增强，人工数据最多 1% 且只能由消融配置启用。
- `default` Profile 启用 MinHash，`exact_only` Profile 只关闭 MinHash且保留 SHA-256。
- Run ID 纳入 Phase 配置、Source Registry、Tokenizer 和 Source Manifest Hash；支持写出完整配置快照。
- Python 语法和静态配置契约检查通过。本机缺少 PyYAML，配置单元测试留待远程依赖环境执行。

旧 `scripts/data_factory/config.py` 和旧 Pipeline 仍保留为 Legacy，Milestone 2 起不再扩展；待 V2 完成 Phase 1 验收后统一淘汰。
## Milestone 2 状态（2026-09-06）

已完成全新的 DataTrove Cache 与 Cheap Tag 基础链路，不调用 V1：

- Source Registry 明确区分物理 `reader` 与逻辑 `adapter`。
- 新增 JSONL、Parquet、tar 内 JSONL Reader，以及 CCI3-HQ、WanJuan、FineWeb、The Stack V2、OpenWebMath、S2ORC 和通用文本 Adapter。
- 新增 `inspect` 门禁，输出真实字段、样本、拒绝原因和来源契约 Hash；契约变化后旧 Inspection 自动失效。
- The Stack V2 官方数据可能仅含 SWHID；若本地数据不存在 `content/text`，Inspection 直接失败，不能把 ID 当训练文本。
- 新增 File Manifest，记录路径、大小、mtime、Parquet 行数/Row Group、来源契约和稳定指纹。
- 新增 DataTrove Canonical Reader、廉价清洗/标签和 ParquetWriter，支持 Local/Slurm Executor 与 completion 恢复。
- Cache ID 同时绑定来源契约和该来源 Manifest Hash；失败 Task 重跑前清理本 Rank 的残留 Parquet。
- Canonical Cache 只执行字段提取、NFC、控制字符清理、许可证检查、极端质量过滤和廉价标签，不执行 Tokenize、MinHash 或 PGCA Feature 生成。
- 新增统一入口 `python -m scripts.data_factory.v2.run {inspect,manifest,cache}`。

验证：`work` 环境运行配置和 Adapter 共 10 项测试全部通过；Python 语法和 CLI 检查通过。DataTrove 未安装在本地 `work` 环境，因此 Cache Executor 实际运行留待远程安装项目依赖后验证。
## Milestone 3 状态（2026-09-06）

已完成 Token Calibration、File-level Plan 与 Candidate Materialization：

- 配置新增有界 Calibration 参数、显式 Seed、Bucket 选择优先级和新增汉字 Token 映射路径。
- Run ID 与 Calibration 缓存同时绑定 Tokenizer、新增汉字映射和 Source Manifest Hash，依赖变化不会复用旧统计。
- Calibration 每来源最多选择 32 个稳定 Cache Shard、扫描 200K 文档并抽取 50K；使用 Rust Char Tokenizer 精确编码样本。
- Calibration 和 Candidate 使用同一个互斥 Bucket 分配函数，按配置优先级先处理新增汉字、专项和混排，防止重叠类别抢占导致 Plan 估算失真。
- Plan 根据 Bucket Token、Source Weight、互斥产率与 Oversample Ratio 选择最少的 Cache 文件；同一来源多个 Bucket 的扫描需求取最大值而不是相加。
- Candidate 仅读取 Plan 的 `selected_files`，通过 DataTrove 单遍分类和稳定 Hash 抽样，不扫描完整 Cache。
- Candidate 输出目录与 Plan Hash 绑定，失败 Task 复用 Runtime Output Guard 清理残留 Parquet。
- 增量 Plan Round 暂不暴露；必须等后续 Selection 产生真实 Shortfall 后实现，避免重复抽取完整目标量。
- CLI 新增 `calibrate`、`plan`、`candidate`。

验证：`work` 环境共 14 项 V2 测试全部通过，其中包含真实项目 Char Tokenizer 对临时 Canonical Parquet 的有界 Calibration；全模块语法和 CLI 检查通过。DataTrove Candidate Executor 的实际多进程/Slurm 运行留待远程安装依赖后验证。
## Milestone 4 状态（2026-09-06）

已完成 SHA-256 Exact Dedup、DataTrove MinHash 编排和 Benchmark Decontamination：

- Exact Signature 按 SHA-256 前两位拆分为 256 个 Parquet 分区，不使用文档级 SQLite。
- Exact Cluster 每次只加载一个 Hash Prefix，按来源质量等级、上游质量分、正文长度和稳定 Doc ID 选择代表。
- Exact Filter 回读 Candidate，写入 `zh/en/code` 三个 MinHash Profile，并保留 `content_sha256`。
- MinHash 使用 DataTrove 官方 Signature、Bucket、Cluster、Filter 四阶段；Signature 与 Filter 保持相同输入和 Task 数。
- 中文使用自定义字符 Tokenizer，代码使用词法 Tokenizer，避免 DataTrove 默认中文 Jieba 分词偏离字符 5-gram 设计。
- `default` Profile 执行 MinHash；`exact_only` 仅用于调试并跳过该阶段。
- 新增 `benchmarks.yaml`，覆盖通用、中英文、代码、CSC/CGEC 和古文评测；必需路径未设置时去污染直接失败。
- 去污染执行完整样本 SHA-256、中文 32 字符精确窗口（至少两个命中）和英文/代码 13-word 精确 n-gram。
- Runtime 支持 Slurm 阶段依赖，并递归清理 Hash/Profile 子目录内失败 Task 的残留 Parquet。
- CLI 新增 `exact_dedup`、`minhash`、`decontaminate`。

验证：前序 14 项 V2 测试继续全部通过；去重模块语法和 CLI 检查通过。本地未安装 DataTrove，因此 MinHash 四阶段和 Slurm 依赖链必须在远程安装依赖后先用 10M Pilot 验证。
## Milestone 5 状态（2026-09-06）

已完成新增汉字覆盖增强与预 Tokenization Final Mixture：

- 从 `new_hanzi_token_ids.json` 获取新增单字集合，仅从 `char_feature_index.jsonl` 加载对应音形 Feature，不读取或生成 PGCA Feature Tensor。
- 第一遍流式统计新增汉字 TF/DF、Source DF、Feature DF 和各 Bucket/Source 可用 Token。
- 第二遍计算 `S_new + 0.25 * S_feature`，为每个字符维护有界 Top-K，并执行 Coverage-first Greedy，避免加载全部候选正文。
- 覆盖目标固定为至少 1/20/100 个文档对应 99%/95%/90%；未自然覆盖字符保留为真实缺口，不生成主实验人工文本。
- 增强池执行单来源、现代中文、繁体和古文比例约束，所有约束和覆盖结果进入总 `passed`。
- 入选增强池的 Doc ID 不再进入其他 Bucket；其余 Bucket 按 Source Weight 和稳定 Hash 选择。
- 第三遍流式写出 Sharded Parquet，并生成新增汉字选中前后频率、Feature Coverage、Bucket/Source Token 和估算缺额报告。
- CLI 新增 `mixture`。

当前 Mixture 使用 Calibration 估算 Token，只是 Exact Tokenization 前的预选结果。最终 ±1% 配额和增量补采必须由 Milestone 6 的真实 Token 数决定。

验证：14 项 V2 回归测试、Mixture 语法和 CLI 检查通过；项目内无 `__pycache__`。完整大规模增强选择需在远程 10M Pilot 验证吞吐和内存。
## Milestone 6 状态（2026-09-06）

已完成 Exact Tokenization、真实配额、验证集、Packing 和 ms-swift 输出：

- Phase 1/2 分别配置独立 1M/10M 验证 Token，Mixture 会额外预留该容量，不从 1B/10B 训练目标中扣除。
- `tokenize` 对 Selected 文本执行唯一一次 Rust `encode_batch()`，保存 `input_ids`、真实 `token_count` 和 Tokenizer Hash。
- `finalize` 基于真实 Token 数按 Bucket 稳定预留验证集，并保证训练/验证 Doc ID 无交集。
- Bucket 最后一个文档超过剩余配额时，只截取已保存 Token IDs并 Decode 对应文本，不重新 Encode，从而精确达到目标。
- 同一批 Token IDs 生成配置指定的 1K/2K 或 2K/4K/8K Packed Parquet，文档间插入 EOS。
- 输出 ms-swift assistant-only `messages` JSONL，同时保留 Tokenized 与 Packed Parquet 用于审计和其他训练后端。
- 不自行伪造 ms-swift 私有缓存格式；新增 `export_ms_swift.sh`，调用官方 `swift export --to_cached_dataset true --truncation_strategy split`。
- 导出与训练必须使用相同 `max_length` 和 `truncation_strategy=split`。
- 最终报告包含精确 Bucket Token、验证 Token、文档交集、Packing Token 和 `exact_shortfalls`。
- CLI 新增 `tokenize`、`finalize`。

验证：V2 共 16 项测试全部通过，覆盖标准 ms-swift Messages JSONL、EOS Packing、真实 Char Tokenizer Calibration 和配置契约；全模块语法检查通过，项目内无 `__pycache__`。
## Dashboard与服务器来源审计（2026-09-14）

根据最新版 `agent/blueprint/data_build/dashboard.md` 和 `scripts/data_factory/configs/sources.yaml` 调整V2：

- Phase 1中文通用改为CCI3-HQ 70% + WanJuan 30%；混排改为CCI3-HQ 40% + FineWeb-Edu-Chinese 40% + The Stack V2 20%；专项改为The Stack V2 70% + OpenWebMath 30%。
- Phase 1新增汉字主来源为CCI3-HQ/FineWeb-Edu-Chinese/WanJuan，并给FineWeb-zhtw和Wikisource各2.5%小额补充权重。
- OpenWebMath、FineWeb-zhtw和Wikisource允许进入Phase 1；Phase 2科学语料由旧`s2orc_open_access`名称改为`peS2o`。
- ect-krp实际为`.txt`，使用整文件Text Reader和`plain_text` Adapter；仓库许可证为CC-BY-SA-4.0。
- peS2o实际为zstd压缩JSONL，新增流式`zstd_jsonl` Reader和`pes2o` Adapter，只保留`source=s2orc`全文论文。Inspection可扩大扫描以越过被过滤的`s2ag`记录。
- 所有dashboard数据集网址进入Source Registry的`homepage`，并纳入来源契约Hash与File Manifest。
- The Stack V2当前路径指向`the-stack-v2-train-full-ids`。该数据通常只有SWHID而没有源码正文；Adapter继续要求`content/text`，Inspection失败时必须先完成Software Heritage代码内容物化，不能绕过。
- 新增运行依赖`zstandard`。本地work环境未安装该包，peS2o Reader需在远程安装更新后的requirements后验证。

V2不依赖根目录旧Python模块。归档V1时仅保留`scripts/data_factory/__init__.py`和`v2/`、`configs/`；其余根目录Python、旧Shell、旧JSON配置和旧README可移入`v1/`。旧单元测试也需同步归档或修正Import。
## Stack V3与最终来源配置审计（2026-09-14）

本节覆盖此前关于The Stack V2的判断：

- 代码来源切换为`HuggingFaceCode/stack-v3-train`，Phase 1/2配置统一使用`the_stack_v3`。
- Stack V3 Train每行是仓库，源码内嵌于`files[].content`；V2的SWHID物化问题不再适用。
- Adapter将仓库行展开为文件级Canonical文档，只保留`is_vendor=false`、`license_type=permissive`且`detected_licenses`非空的文件，保留repo/commit/path/language/content_id来源信息。
- 本地只下载全量Parquet的10%；Pipeline不再二次乘0.1，而是以下载后的真实Manifest和Calibration容量为准。应随机选择Shard，不应只下载连续前10%。
- Phase 1权重按最新dashboard同步，Phase 2科学来源改为peS2o，所有Source Homepage进入来源契约和Manifest。
- ect-krp使用整文件Text Reader和CC-BY-SA-4.0；peS2o使用zstd JSONL Reader并只保留`source=s2orc`。

验证：更新后18项V2测试全部通过，包含Stack V3仓库展开、许可/vendor过滤、peS2o过滤和ect-krp文本读取测试。本地仍缺少`zstandard`，需在远程安装requirements后验证peS2o实际压缩文件。
补充实现结论：Phase 1互斥Bucket优先级调整为`new_char_enhancement -> mixed_zh_en -> specialized -> zh_knowledge -> zh_general -> english`，确保Stack V3中含中文的README/docs进入混排桶，其余代码/结构化文件进入专项桶。V1已由用户归档至`scripts/data_factory/v1/`与`tests/v1/`，根目录重新建立V2专用最小`__init__.py`。
