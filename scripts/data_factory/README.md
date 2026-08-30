# Data Factory

Phase 1 语义对齐数据构建工具。默认配置为 `scripts/data_factory/phase1_config.json`，所有命令均在项目根目录执行。


## 推荐：Fast Phase 1 流水线

对于 1B token 语义对齐数据，推荐使用新的 `cache -> calibrate -> fast_sample` 管线。它避免旧版 `document_index.sqlite/candidate_index.sqlite` 的逐文档索引，并保证全量语料扫描阶段不调用 tokenizer。

```bash
# 一次执行（首次会构建可复用 Parquet cache）
bash scripts/data_factory/run_phase1_fast.sh

# 或分阶段
python -m scripts.data_factory.build_phase1 cache --workers 8
python -m scripts.data_factory.build_phase1 calibrate --workers 8
python -m scripts.data_factory.build_phase1 fast_sample --workers 8
```

数据路径：

```text
data/semantic_alignment/
├── cache_parquet/        # 原始数据只清洗/解压一次，可被 Phase1/Phase2 重用
├── fast_candidates/      # 约 1.2x token budget；尚未精确 tokenize
├── fast_tokenized/       # candidate-only，一次 exact tokenization 后的 input_ids
├── fast_final/
│   ├── train/            # 与旧流程兼容的 text JSONL
│   ├── validation/
│   └── packed/           # 推荐训练直接使用的 pre-tokenized Parquet
└── reports/
    ├── phase1_fast_cache_report.json
    ├── phase1_token_calibration.json
    ├── phase1_fast_prescan_report.json
    ├── phase1_fast_tokenization_report.json
    └── phase1_fast_final_report.json
```

Fast pipeline 的执行顺序：

1. `cache`：WanJuan `.tar.gz`、CCI JSONL、FineWeb Parquet 等统一清洗并缓存成中等大小 Parquet shard；之后不再反复解压/JSON parse。
2. `calibrate`：每个 source 仅确定性抽样约 50K 文档，使用底层 Rust `tokenizers` 做 batch tokenize，拟合 `hanzi/latin/digit/other -> token_count` estimator。
3. `fast_sample` 第一阶段：全量 cache 只做 cheap classifier、Aho-Corasick bridge、新汉字统计和 estimated-token sampling；按 source 配额过采样 20%，feature pool 过采样 50%。
4. candidate pool 先做 exact hash dedup 和 exact benchmark decontamination。
5. 只对 candidate pool 执行一次 `encode_batch()`；`input_ids` 写入 Parquet。
6. validation、1B quota、rare-Hanzi/bridge coverage 和最终 packing 全部基于已经保存的 token ids，不再次 tokenize。

`phase1_config.json` 的 `fast_pipeline` 控制缓存分片、校准样本数、oversampling、Rayon 线程数和 packing 长度。第一次 `cache` 成本较高，但 Phase1 重建、配额消融以及后续 Phase2 都可以直接复用。

旧的 `prepare/sample` SQLite 管线暂时保留用于结果对照和回退，不建议再作为 1B 主构建路径。

## 旧版流水线：快速使用

```bash
# 完整执行：清洗 -> 1.1B 候选预选 -> 精确过滤 -> 缺额补采 -> 分片
bash scripts/data_factory/run_phase1.sh all

# 分阶段执行
bash scripts/data_factory/run_phase1.sh prepare
bash scripts/data_factory/run_phase1.sh sample --candidate-tokens 1100000000

# 可选：旧实验所需的全量精确去重，不属于默认 all 流程
bash scripts/data_factory/run_phase1.sh dedup
# 仅从已有选择重新导出两套验证集
bash scripts/data_factory/run_phase1.sh validation --overwrite
```

可通过环境变量替换配置文件：

```bash
PHASE1_CONFIG=/path/to/config.json bash scripts/data_factory/run_phase1.sh all
```

## 可执行脚本

### `build_phase1.py`

Phase 1 主执行门面。

```bash
python -m scripts.data_factory.build_phase1 ACTION \
  --config scripts/data_factory/phase1_config.json \
  [--overwrite | --resume] [--workers N] [--candidate-tokens N]
```

- `ACTION`：旧版为 `prepare`、`dedup`、`sample`、`validation`、`all`；Fast pipeline 为 `cache`、`calibrate`、`fast_sample`、`fast_all`。
- `--config`：配置文件路径，默认 `scripts/data_factory/phase1_config.json`。
- `--overwrite`：删除对应阶段的已有产物后重新构建。
- `--resume`：用于 `prepare`、`dedup`、`sample`，从各阶段已提交的文件或分片继续。
- `--workers N`：覆盖对应阶段的进程数；默认分别读取 `prepare_workers`、`dedup_workers`、sample_workers。
- `--candidate-tokens N`：仅用于 `sample`/`all`，从 `normalized` 预选的估算 token 预算，默认 `1100000000`。
- `--source NAME`：用于 `prepare` 或 `cache`，只处理指定来源；可重复传入。Fast cache 支持分来源逐步构建并保留其他已完成来源。

仅处理单个来源并保留其他已生成数据：

```bash
bash scripts/data_factory/run_phase1.sh prepare --source cci3_hq
bash scripts/data_factory/run_phase1.sh prepare --source wanjuan
```

重新生成同名来源时追加 `--overwrite`。CLUE 默认禁用，不进入 Phase 1 训练数据。

### `run_phase1.sh`

对 `build_phase1.py` 的 Bash 封装。第一个参数是 `ACTION`，其余参数原样传递给 Python 门面。

### `collect_external.py`

根据 JSONL manifest 采集 GitHub、Hugging Face 和许可明确的白名单网页，输出统一 JSONL。

```bash
python -m scripts.data_factory.collect_external \
  --manifest resources/raw/manifests/sources.jsonl \
  --output resources/raw/phase1/collected/external.jsonl
```

- `--manifest`：采集清单，每行一个来源对象。
- `--output`：标准 JSONL 输出路径。
- `--github-token-env`：GitHub Token 环境变量名，默认 `GITHUB_TOKEN`。
- `--user-agent`：网页请求 User-Agent。
- `--delay`：相邻请求间隔秒数，默认 `0.2`。
- `--max-file-bytes`：单文件大小上限，默认 `1000000`。

Manifest 示例：

```jsonl
{"provider":"github","repo":"owner/repo","license":"MIT","mode":"mixed"}
{"provider":"huggingface","repo":"org/model","repo_type":"model","license":"Apache-2.0","mode":"mixed"}
{"provider":"web","url":"https://example.org/doc","license":"CC-BY-4.0","revision":"2026-07-15","mode":"mixed"}
```

`mode` 支持 `mixed`、`code`、`structured`、`math`、`hanzi`、`chat`。GitHub/Hugging Face 可额外指定 `revision`、`max_files`、`max_chars`；网页来源必须提供 `revision`。

### `extract_dumps.py`

将公开 dump 转换为统一 JSONL。

```bash
# Stack Exchange
python -m scripts.data_factory.extract_dumps stackexchange \
  --input Posts.xml \
  --output resources/raw/phase1/collected/stack_math.jsonl \
  --site math.stackexchange.com \
  --revision 2026-07-01 \
  --quota-group math

# Wikimedia
python -m scripts.data_factory.extract_dumps wikimedia \
  --input zhwiki-pages-articles.xml.bz2 \
  --output resources/raw/phase1/collected/zhwiki.jsonl \
  --revision 2026-07-01 \
  --category mixed_zh_en \
  --quota-group wikimedia_openalex

# OpenAlex
python -m scripts.data_factory.extract_dumps openalex \
  --input works.jsonl.gz \
  --output resources/raw/phase1/collected/openalex.jsonl \
  --revision 2026-07-01
```

Stack Exchange 参数：`--site` 指站点域名；`--quota-group` 为 `code`、`structured` 或 `math`；`--min-question-score`、`--min-answer-score` 默认分别为 `0`、`5`；`--max-chars` 默认 `16000`。

Wikimedia 参数：`--project` 默认 `zh.wikipedia.org`；`--category` 为 `mixed_zh_en` 或 `supplemental`；`--quota-group` 为 `wikimedia_openalex`、`math` 或 `hanzi`；`--max-chars` 默认 `16000`。

### `generate_supplemental.py`

生成 chat template 数据或缺失汉字覆盖数据。

```bash
# 将含 messages 字段的 JSONL 渲染为 Qwen chat template
python -m scripts.data_factory.generate_supplemental chat \
  --inputs conversations.jsonl \
  --output resources/raw/phase1/collected/chat.jsonl \
  [--model-path models/Qwen3-1.7B-Base-Char]

# 为未覆盖汉字生成补充样本
python -m scripts.data_factory.generate_supplemental hanzi \
  --observed 'data/semantic_alignment/normalized/*.jsonl' \
  --output resources/raw/phase1/collected/hanzi.jsonl \
  --license CC0-1.0 \
  --revision 2026-07-15
```

汉字模式可用 `--hanzi-set` 和 `--feature-index` 覆盖默认资源路径；`--observed` 接收零个或多个 JSONL glob；`--output`、`--license`、`--revision` 必填。

## 内部模块

| 文件 | 功能 |
| --- | --- |
| `config.py` | 加载并校验 Phase 1 总配额、专项子配额、窗口与验证配置。 |
| `sources.py` | 流式读取 CCI3-HQ、WanJuan、Chinese FineWeb、FineWeb-Edu 和外部 JSONL；CLUE 仅保留诊断适配器。 |
| `prepare.py` | 统一字段、文本清洗、许可与质量过滤。 |
| `text.py` | NFC 规范化、PII/密钥检测、混排判定、格式校验及 shingle 生成。 |
| `windowing.py` | 按句子和自然段构造 token 窗口，保护代码围栏、展示公式和 Markdown 表格。 |
| `candidate_features.py` | 分类基础候选池，并抽取被裁多字 token 与新增汉字命中窗口。 |
| `dedup.py` | SQLite 精确哈希去重、精确评测去污染、可选 MinHash 近似去重及 registry 导出。 |
| `prescan.py` | 全量轻量扫描、全局桥接词统计、来源平衡预选和分片级恢复。 |
| `sample.py` | 从 `normalized` 预选候选，执行候选级精确去重/去污染、缺额补采、验证预留和 1B 分片。 |
| `selection.py` | 执行父文档互斥分组、新增汉字覆盖及被裁多字 token 桥接采样。 |
| `io_utils.py` | JSON/JSONL、路径展开、哈希和分片写入工具。 |
| `source_cache.py` | 将不同原始来源一次性标准化为可复用 Parquet cache，按原始文件并行并支持 artifact resume。 |
| `token_calibration.py` | 每来源小样本 exact tokenization，拟合 token estimator，避免全量精确计数。 |
| `fast_common.py` | Fast pipeline 的 deterministic sampling、Parquet writer 和 token estimator。 |
| `fast_sample.py` | Cheap prescan、source-aware oversampling、candidate-only exact tokenization、quota finalize 与 token-level packing。 |
| `phase1_config.json` | 默认数据路径、清洗阈值、去重参数和来源配置。 |

默认产物位于 `data/semantic_alignment/` 下的 `raw_manifest/`、`normalized/`、`final/` 和 `reports/`；`deduplicated/` 仅在显式运行旧式全量 `dedup` 时生成。

## 来源发现

discover_sources.py 生成可供 collect_external.py 使用的候选 manifest。候选项仍需人工抽查许可证和内容质量，确认后再采集。

### 仓库发现

```bash
python -m scripts.data_factory.discover_sources repositories   --output resources/raw/manifests/repositories.candidates.jsonl
```

常用参数：

- --provider github huggingface：搜索平台，默认两者。
- --search TEXT：搜索式，可重复传入；未指定时使用内置中文技术主题。
- --mode mixed|code|structured：采集内容类型，默认 mixed。
- --limit-per-query N：每个平台、每个搜索式最多返回数，默认 100。
- --min-stars N：GitHub 最低 star 数，默认 10。
- --licenses ...：允许的 SPDX 许可证。
- --hf-repo-types model dataset：Hugging Face 仓库类型。
- --github-token-env、--hf-token-env：Token 环境变量名。

建议设置 GITHUB_TOKEN，否则 GitHub 搜索 API 的频率限制较低。候选清单抽查后可直接作为采集 manifest：

```bash
python -m scripts.data_factory.collect_external   --manifest resources/raw/manifests/repositories.candidates.jsonl   --output resources/raw/phase1/collected/external.jsonl
```

### Sitemap 展开

先写一个站点级清单，每行只描述一个已确认许可证的站点：

```jsonl
{"sitemap":"https://example.org/sitemap.xml","license":"CC-BY-4.0","revision":"2026-07-15","mode":"mixed","include":"/docs/","exclude":"/archive/","max_urls":5000}
```

然后展开为逐页 manifest：

```bash
python -m scripts.data_factory.discover_sources sitemap   --sites resources/raw/manifests/sites.jsonl   --output resources/raw/manifests/web.candidates.jsonl
```

参数 --max-urls-per-site 和 --max-sitemaps 控制单站规模，--delay 控制 sitemap 请求间隔。采集网页时仍会检查 robots.txt。

GitHub 采集现在按仓库 tarball 下载，再在本地筛选文件，避免逐文件调用 API。collect_external.py 的 --max-archive-bytes 控制单仓库压缩包上限，默认 500000000；manifest 中也可为单个仓库设置 max_archive_bytes。

## 中英混排/代码/数学/格式初始数据获取流程

以下流程只负责获得“中英混排”和“代码/数学/格式”原始统一 JSONL，不执行后续清洗、去重和采样。

**0. 准备**
```bash
mkdir -p resources/raw/manifests
mkdir -p resources/raw/phase1/{archives,collected}

export GITHUB_TOKEN=你的_token
export HF_TOKEN=你的_token   # 公共仓库可不设置，但建议设置
```

**1. 中英混排：GitHub + Hugging Face**
```bash
python -m scripts.data_factory.discover_sources repositories \
  --provider github huggingface \
  --mode mixed \
  --output resources/raw/manifests/mixed_repositories.candidates.jsonl
```

人工检查候选文件，删除主题无关、质量明显较差的仓库，不要手工修改许可证：

```bash
cp resources/raw/manifests/mixed_repositories.candidates.jsonl \
   resources/raw/manifests/mixed_repositories.approved.jsonl
vim resources/raw/manifests/mixed_repositories.approved.jsonl
```

采集：

```bash
python -m scripts.data_factory.collect_external \
  --manifest resources/raw/manifests/mixed_repositories.approved.jsonl \
  --output resources/raw/phase1/collected/mixed_repositories.jsonl
```

**2. 中英混排：网页补充，可选**

你只需要填写少量“许可证明确的网站”，不需要逐页填写：

```jsonl
{"sitemap":"https://example.org/sitemap.xml","license":"CC-BY-4.0","revision":"2026-07-15","mode":"mixed","include":"/docs/","max_urls":5000}
```

保存为 `resources/raw/manifests/mixed_sites.jsonl`，然后执行：

```bash
python -m scripts.data_factory.discover_sources sitemap \
  --sites resources/raw/manifests/mixed_sites.jsonl \
  --output resources/raw/manifests/mixed_web.jsonl

python -m scripts.data_factory.collect_external \
  --manifest resources/raw/manifests/mixed_web.jsonl \
  --output resources/raw/phase1/collected/mixed_web.jsonl
```

没有足够的合规网站可以先跳过，后续用更多合规 GitHub/Hugging Face 来源补足。

**3. 代码数据**
```bash
python -m scripts.data_factory.discover_sources repositories \
  --provider github \
  --mode code \
  --search "language:Python" \
  --search "language:JavaScript" \
  --search "language:TypeScript" \
  --search "language:Java" \
  --search "language:C++" \
  --search "language:Go" \
  --search "language:Rust" \
  --search "language:Shell" \
  --licenses Apache-2.0 MIT BSD-2-Clause BSD-3-Clause ISC MPL-2.0 \
  --output resources/raw/manifests/code.candidates.jsonl
```

检查后采集：

```bash
cp resources/raw/manifests/code.candidates.jsonl resources/raw/manifests/code.approved.jsonl

python -m scripts.data_factory.collect_external \
  --manifest resources/raw/manifests/code.approved.jsonl \
  --output resources/raw/phase1/collected/code_github.jsonl
```

**4. Markdown/JSON/YAML/TOML/XML 格式数据**
```bash
python -m scripts.data_factory.discover_sources repositories \
  --provider github \
  --mode structured \
  --search "API documentation JSON" \
  --search "YAML configuration" \
  --search "TOML configuration" \
  --search "XML configuration" \
  --search "Markdown documentation" \
  --output resources/raw/manifests/structured.candidates.jsonl

cp resources/raw/manifests/structured.candidates.jsonl \
   resources/raw/manifests/structured.approved.jsonl

python -m scripts.data_factory.collect_external \
  --manifest resources/raw/manifests/structured.approved.jsonl \
  --output resources/raw/phase1/collected/structured_github.jsonl
```

**5. 数学数据**

从 [Stack Exchange 官方数据归档](https://archive.org/download/stackexchange/) 下载这三个文件：

```bash
wget -c https://archive.org/download/stackexchange/math.stackexchange.com.7z
wget -c https://archive.org/download/stackexchange/stats.stackexchange.com.7z
wget -c https://archive.org/download/stackexchange/tex.stackexchange.com.7z

7z x math.stackexchange.com.7z -o resources/raw/phase1/archives/math
7z x stats.stackexchange.com.7z -o resources/raw/phase1/archives/stats
7z x tex.stackexchange.com.7z -o resources/raw/phase1/archives/tex
```

将 `DUMP_DATE` 设置成实际下载的 dump 日期：

```bash
DUMP_DATE=2026-07-15

python -m scripts.data_factory.extract_dumps stackexchange \
  --input resources/raw/phase1/archives/math/Posts.xml \
  --output resources/raw/phase1/collected/math_stackexchange.jsonl \
  --site math.stackexchange.com --revision "$DUMP_DATE" --quota-group math

python -m scripts.data_factory.extract_dumps stackexchange \
  --input resources/raw/phase1/archives/stats/Posts.xml \
  --output resources/raw/phase1/collected/stats_stackexchange.jsonl \
  --site stats.stackexchange.com --revision "$DUMP_DATE" --quota-group math

python -m scripts.data_factory.extract_dumps stackexchange \
  --input resources/raw/phase1/archives/tex/Posts.xml \
  --output resources/raw/phase1/collected/tex_stackexchange.jsonl \
  --site tex.stackexchange.com --revision "$DUMP_DATE" --quota-group math
```

至此，所需初始数据都位于：

```text
resources/raw/phase1/collected/
```

无需合并这些 JSONL，现有配置会读取该目录下所有文件。若数量不足，使用新的搜索词重复发现流程，但每批使用不同输出文件名，避免覆盖已有结果。

## Prepare 与可选全量去重性能

`prepare` 按原始文件拆分任务并行完成清洗。每个任务先写入 `normalized/.prepare/staging/`，完成后原子提交；状态保存在 `normalized/.prepare/state*.json`。单个任务失败不会阻止其他成功任务提交，失败清单写入状态的 `failed_tasks`，命令最终仍以非零状态退出。显式 `dedup` 仅用于旧实验的全量精确去重，不是当前 `sample` 的依赖；MinHash/LSH 默认关闭。

如需复现旧式全量去重：

```bash
bash scripts/data_factory/run_phase1.sh prepare --overwrite --workers 16
bash scripts/data_factory/run_phase1.sh dedup --overwrite --workers 16
```

中断后继续：

```bash
bash scripts/data_factory/run_phase1.sh prepare --resume --workers 16
bash scripts/data_factory/run_phase1.sh dedup --resume --workers 16
```

配置项：

- `prepare_workers`、`dedup_workers`：默认进程数，均为 `8`。
- `dedup.near_duplicate_enabled`：是否执行 MinHash 近似去重；当前为 `false`。
- `dedup.batch_size`、`dedup.batch_chars`：单个哈希计算批次上限，精确模式默认为 `2048` 条、`4000000` 字符。
- `dedup.commit_interval`：SQLite 提交间隔，精确模式默认为 `20000` 条。
- `dedup.sqlite_cache_mb`：全局 SQLite 索引缓存上限。
- `dedup.export_registry_parquet`：完成后是否导出跨阶段 registry，默认开启。

调整 worker、批大小或 SQLite 缓存不会使恢复状态失效。`prepare --resume` 会按文件指纹仅重跑新增、重新下载或配置受影响的任务；只有 prepare 状态结构版本不兼容时才需要 `--overwrite`。标准化输入或去重语义参数发生变化后，`dedup` 仍须使用 `--overwrite` 重建全局索引。进程数过高会争用共享存储和内存带宽，建议从 `8` 开始，根据 CPU、内存和磁盘利用率增加。

## Sampling performance and resume

`sample` 依次执行：

1. 轻量扫描全部 `normalized` 分片，只记录偏移、估算 token、类别和对齐特征。
2. 按训练/验证配额，从 `normalized` 预选 `candidate_tokens` 候选，默认 1.1B 估算 token。
3. 仅对候选执行 SHA-256 精确去重和评测集精确去污染，再运行 Fast Tokenizer 和窗口构建。
4. 按实际 token 选择训练集与验证集；若配额不足，从未选文档中按缺额乘 1.25 自动补采，最多 8 轮。

主要配置：

- `candidate_tokens`：初始候选预算，默认 `1100000000`，可由 `--candidate-tokens` 覆盖。
- `sample_batch_size`：精确 tokenizer 单批最大记录数，默认 `512`。
- `sample_batch_chars`：精确 tokenizer 单批最大字符数，默认 `1000000`。
- `sample_workers`：预扫描进程和 tokenizer 并发数，默认 `8`。
- `preselection_buffer_ratio`：稀有汉字和桥接上下文的覆盖安全系数，默认 `1.25`。

首次使用新流程必须重建 sample 状态：

```bash
bash scripts/data_factory/run_phase1.sh sample \
  --overwrite \
  --workers 16 \
  --candidate-tokens 1100000000
```

中断后继续：

```bash
bash scripts/data_factory/run_phase1.sh sample --resume --workers 16
```

`--resume` 会复用已完成的轻量扫描、已过滤父文档和已 tokenizer 化候选。规范化输入、窗口配置、词表对齐元数据、去污染文件或候选预算变化后，应使用 `sample --overwrite`。
## Phase 1 validation set

`sample` 会在训练采样前按父文档预留验证数据，并同时输出：

- `data/semantic_alignment/validation/validation_natural.jsonl`：约 2.5M token。
- `data/semantic_alignment/validation/validation_alignment.jsonl`：约 1M token。
- `data/semantic_alignment/reports/phase1_validation_report.json`：配额和训练重叠检查。

如文件被删除，可从已有 `candidate_index.sqlite` 重新导出，不会重新选择或扫描原始语料：

```bash
bash scripts/data_factory/run_phase1.sh validation --overwrite
```

`sample --resume` reuses completed lightweight-scan shards and exact-tokenized documents. Changes to windowing, vocabulary-alignment metadata, priority-Hanzi resources, or preselection settings require `sample --overwrite`.