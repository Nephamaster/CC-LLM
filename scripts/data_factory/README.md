# Data Factory

Phase 1 语义对齐数据构建工具。默认配置为 `scripts/data_factory/phase1_config.json`，所有命令均在项目根目录执行。

## 快速使用

```bash
# 完整执行：清洗 -> 去重 -> 候选索引 -> 验证预留 -> 配额采样与分片
bash scripts/data_factory/run_phase1.sh all

# 分阶段执行
bash scripts/data_factory/run_phase1.sh prepare
bash scripts/data_factory/run_phase1.sh dedup
bash scripts/data_factory/run_phase1.sh sample
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
  [--overwrite | --resume] [--workers N]
```

- `ACTION`：`prepare`、`dedup`、`sample`、`validation` 或 `all`。
- `--config`：配置文件路径，默认 `scripts/data_factory/phase1_config.json`。
- `--overwrite`：删除对应阶段的已有产物后重新构建。
- `--resume`：用于 `prepare`、`dedup`、`sample`，从各阶段已提交的文件或分片继续。
- `--workers N`：覆盖对应阶段的进程数；默认分别读取 `prepare_workers`、`dedup_workers`、`sample_workers`。
- `--source NAME`：仅用于 `prepare`，只处理指定来源；可重复传入。与 `--overwrite` 合用时只覆盖该来源的标准化分片。

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
  --observed 'data/semantic_alignment/deduplicated/*.jsonl' \
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
| `dedup.py` | SQLite 精确去重、MinHash 近似去重、评测去污染及 registry 导出。 |
| `prescan.py` | 全量轻量扫描、全局桥接词统计、来源平衡预选和分片级恢复。 |
| `sample.py` | 构建候选索引，先预留验证父文档，再按 1B 配额采样、打乱和分片。 |
| `selection.py` | 执行父文档互斥分组、新增汉字覆盖及被裁多字 token 桥接采样。 |
| `io_utils.py` | JSON/JSONL、路径展开、哈希和分片写入工具。 |
| `phase1_config.json` | 默认数据路径、清洗阈值、去重参数和来源配置。 |

默认产物位于 `data/semantic_alignment/` 下的 `raw_manifest/`、`normalized/`、`deduplicated/`、`final/` 和 `reports/`。

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

## Prepare 与去重性能

`prepare` 按原始文件拆分任务并行完成清洗。每个任务先写入 `normalized/.prepare/staging/`，完成后原子提交；状态保存在 `normalized/.prepare/state*.json`。`dedup` 在进程池中并行计算 SHA-256 和 MinHash，主进程仍按固定输入顺序更新全局 SQLite LSH，因此保留跨分片去重和确定性。每完成一个标准化分片即更新 `deduplicated/dedup_state.json`。

首次使用新实现重建：

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
- `dedup.batch_size`、`dedup.batch_chars`：单个签名计算批次的记录数和字符数上限。
- `dedup.sqlite_cache_mb`：全局 SQLite 索引缓存上限。
- `dedup.export_registry_parquet`：完成后是否导出跨阶段 registry，默认开启。

调整 worker、批大小或 SQLite 缓存不会使恢复状态失效。`prepare --resume` 会按文件指纹仅重跑新增、重新下载或配置受影响的任务；只有 prepare 状态结构版本不兼容时才需要 `--overwrite`。标准化输入或去重语义参数发生变化后，`dedup` 仍须使用 `--overwrite` 重建全局索引。进程数过高会争用共享存储和内存带宽，建议从 `8` 开始，根据 CPU、内存和磁盘利用率增加。

## Sampling performance and resume

`sample` now runs three stages:

1. Scan every deduplicated shard without tokenization, recording compact byte offsets, estimated window tokens, new-Hanzi hits, and global removed-token frequencies in `document_index.sqlite`.
2. Rescan only for the global Top-5000 bridge terms and preselect a buffered, source-balanced candidate set.
3. Run the Fast Tokenizer and exact windowing only on preselected documents, storing exact candidates in `candidate_index.sqlite` before validation reservation and quota selection.

Tune these fields in `phase1_config.json`:

- `sample_batch_size`: maximum records per exact-tokenizer batch; default `512`.
- `sample_batch_chars`: maximum characters per exact-tokenizer batch; default `1000000`.
- `sample_workers`: prescan processes and concurrent exact-tokenizer batches; default `8`.
- `preselection_buffer_ratio`: estimated-token safety margin before exact tokenization; default `1.25`.

Start with `--workers 8` or `--workers 16`. More workers increase Aho-Corasick copies and in-flight tokenizer memory, so stop increasing the value after CPU, memory, or storage is saturated.

Resume an interrupted run with both SQLite indexes intact:

```bash
bash scripts/data_factory/run_phase1.sh sample --resume --workers 16
```

The first run after this pipeline upgrade must use `sample --overwrite` because the old full-tokenization candidate index is incompatible. Afterwards, use `--resume`; `--overwrite` deletes both indexes and the per-shard prescan cache.

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