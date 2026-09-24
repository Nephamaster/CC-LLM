# Data Factory V2

Milestone 2 提供 `inspect -> manifest -> cache` 三个阶段。所有命令在项目根目录执行。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

DataTrove 固定为 `datatrove[io,processing]>=0.10,<0.11`。

## 2. 配置来源路径

已知服务器路径已写入 `scripts/data_factory/configs/sources.yaml`。尚未确定的来源通过环境变量提供，例如：

```bash
export FINEWEB_CHINESE_PATH='/path/to/fineweb_chinese/**/*.parquet'
export CHINESE_COSMOPEDIA_PATH='/path/to/chinese_cosmopedia/**/*.parquet'
```

Phase 2 其他变量见 `sources.yaml`。环境变量未设置时，程序会直接报告 unresolved path。

Stack V3 Train 每行是一个仓库，代码位于 `files[].content`。Adapter 会展开非 vendor、许可明确的文件；已下载的 10% Parquet 分片直接作为可用池，由 Calibration 和 Plan 按实际容量抽样。

## 3. 检查真实 Schema

```bash
python -m scripts.data_factory.v2.run inspect \
  --config scripts/data_factory/configs/phase1.yaml \
  --max-files 3 \
  --max-rows 20
```

只检查指定来源：

```bash
python -m scripts.data_factory.v2.run inspect \
  --config scripts/data_factory/configs/phase1.yaml \
  --source cci3_hq
```

报告写入 `data/corpus/metadata/inspection/<source>.json`。Cache 要求 Reader、Adapter 和来源契约完全匹配且 Inspection `passed=true`。

## 4. 冻结 File Manifest

```bash
python -m scripts.data_factory.v2.run manifest \
  --config scripts/data_factory/configs/phase1.yaml
```

Manifest 位于 `data/corpus/registry/manifests/`，记录文件路径、大小、mtime、Parquet 行数、Row Group 和稳定指纹。

## 5. 构建 Canonical Cache

单机多进程：

```bash
python -m scripts.data_factory.v2.run cache \
  --config scripts/data_factory/configs/phase1.yaml \
  --executor local \
  --workers 16
```

Slurm：

```bash
python -m scripts.data_factory.v2.run cache \
  --config scripts/data_factory/configs/phase1.yaml \
  --executor slurm \
  --workers 64 \
  --slurm-partition cpu \
  --slurm-time 12:00:00 \
  --cpus-per-task 1 \
  --mem-per-cpu-gb 4 \
  --venv-path /path/to/venv
```

输出位于 `data/corpus/cache/<source>/<cache_id>/`，DataTrove 日志和 completion marker 位于 `data/corpus/metadata/cache_logs/`。重新执行相同配置时只运行未完成 Task。

## 6. Calibration、Plan 与 Candidate

完成所有当前 Phase 来源的 Cache 后，重新生成完整 Manifest，再执行：

```bash
python -m scripts.data_factory.v2.run calibrate \
  --config scripts/data_factory/configs/phase1.yaml \
  --overwrite

python -m scripts.data_factory.v2.run plan \
  --config scripts/data_factory/configs/phase1.yaml \
  --overwrite

python -m scripts.data_factory.v2.run candidate \
  --config scripts/data_factory/configs/phase1.yaml \
  --executor local \
  --workers 16
```

Calibration 对每个来源最多读取配置的 32 个 Cache Shard，并最多扫描 `50000 * 4` 条记录；普通类别与增强资格分别统计，二者可重叠。

Plan 按 Bucket Token、Source Weight、产率和安全余量选择 Cache 文件。`passed=false` 时 Candidate 拒绝启动。Plan 是不可变产物；修改配置、Tokenizer、新增汉字清单或 Source Manifest 会生成新的 Run ID。

Candidate 只读取 Plan 的 `selected_files`，使用与 Calibration 相同的 Bucket 优先级单遍分类和采样，再由 DataTrove 写入：

```text
data/corpus/runs/<phase>/<run_id>/candidates/<plan_hash>/
```

使用 Slurm 时，将 Candidate 命令的执行器参数替换为 Cache 阶段相同的 Slurm 参数。

## 7. Exact、MinHash 与去污染

Candidate 完成后依次执行：

```bash
python -m scripts.data_factory.v2.run exact_dedup \
  --config scripts/data_factory/configs/phase1.yaml \
  --executor local --workers 16

python -m scripts.data_factory.v2.run minhash \
  --config scripts/data_factory/configs/phase1.yaml \
  --executor local --workers 16

python -m scripts.data_factory.v2.run decontaminate \
  --config scripts/data_factory/configs/phase1.yaml \
  --executor local --workers 16
```

Exact 阶段以 SHA-256 前两位分成 256 个分区，按来源质量、上游质量分、正文长度和稳定 Doc ID 选择代表，不建立文档级 SQLite。Exact 输出按 `zh/en/code` 分区，供 MinHash 分别处理。

MinHash 使用 DataTrove 官方 Signature、Bucket、Cluster、Filter 四阶段。中文使用字符 5-gram，英文使用词级 5-gram，代码使用词法 Token 5-gram。`exact_only` Profile 会跳过 MinHash，但不能作为正式训练数据产物。

去污染前必须设置 `scripts/data_factory/configs/benchmarks.yaml` 中所有必需环境变量。完整样本使用 SHA-256，中文使用 32 字符精确窗口且至少命中两个窗口，英文和代码使用 13-word 精确 n-gram。任何必需 Benchmark 路径为空时阶段直接失败。

## 8. 新增汉字增强与 Mixture

完成去污染后执行：

```bash
python -m scripts.data_factory.v2.run mixture \
  --config scripts/data_factory/configs/phase1.yaml \
  --overwrite
```

该阶段扫描去污染后的 Candidate，输出新增汉字 TF/DF、Feature DF、选中前后覆盖和来源约束报告；采用 Coverage-first Greedy 后按 `S_new + 0.25 * S_feature` 补齐增强配额。入选增强池的文档不会再进入原 Bucket。

其他 Bucket 按来源配额从剩余候选确定性填充，输出到 `selected/<plan_hash>/`。99%/95%/90%覆盖只作诊断；来源、数量及横向约束仍参与 passed。失败阶段非零退出，tokenize 只接受当前 Plan 的成功 Mixture。

若 Mixture 或 Finalize 有Token缺额，执行 `plan --round 1`，再依次执行 `candidate/exact_dedup/minhash/decontaminate/mixture/tokenize/finalize --round 1`（配置参数照旧）。Candidate 只读未用过的Cache分片，去重对全部轮次候选重新执行。再次补采使用递增round，报告必须来自上一轮；没有Token缺额时拒绝补采。只有横向/来源配比失败时应调整选择或数据配置，不能靠无限补采。

本次更新生成新的Run ID。先用 `cache --source the_stack_v3` 重建修复后的Stack V3缓存，再用不带source的manifest冻结完整来源清单，其他Cache可复用。两阶段从calibrate开始执行。首次运行无需overwrite；在同一轮重建Mixture、Tokenize或Finalize才使用overwrite。不要删除Cache来清理失败Run。增量Tokenize按ID、文本Hash和Tokenizer Hash复用旧轮Token IDs，未变化文本不重复编码。

## 9. Exact Tokenization、Finalization 与 ms-swift

```bash
python -m scripts.data_factory.v2.run tokenize \
  --config scripts/data_factory/configs/phase1.yaml \
  --overwrite

python -m scripts.data_factory.v2.run finalize \
  --config scripts/data_factory/configs/phase1.yaml \
  --overwrite
```

`tokenize` 使用最终 `tokenizer.json` 对预选文档执行一次批量编码并保存 Tokenized Parquet。`finalize` 只读取 `input_ids`，按真实 Token 数预留验证集、精确裁剪 Bucket 配额并生成多长度 Packed Parquet；跨越配额边界的文档只截取已有 Token IDs，不重新编码。

ms-swift 使用 assistant-only `messages` JSONL。按照官方接口生成 cached dataset：

```bash
MODEL_PATH=models/Qwen3-1.7B-Base-Char-PGCA \
FINAL_DIR=data/corpus/runs/phase1/<run_id>/final \
MAX_LENGTH=2048 \
bash scripts/data_factory/v2/export_ms_swift.sh
```

训练时使用脚本输出的 `--cached_dataset` 和 `--cached_val_dataset`。导出与训练必须保持相同的 `max_length` 和 `truncation_strategy=split`。

## 10. 当前边界

### 接受已有Mixture继续处理

如果明确决定停止补采，接受已有数据的Token数量、来源及专项子类比例偏差，可执行：

```bash
python -m scripts.data_factory.v2.run tokenize \
  --config scripts/data_factory/configs/phase1.yaml \
  --round 1 --accept-existing-mixture

python -m scripts.data_factory.v2.run finalize \
  --config scripts/data_factory/configs/phase1.yaml \
  --round 1 --accept-existing-mixture
```

该运行参数不改变配置Hash或Run ID，不需重跑上游。只有Tokenize/Finalize接受此参数；默认严格流程不变。必须匹配当前Run和Plan，增强约束及全局横向约束失败不能通过此开关绕过。

Finalize按稳定父文档划分预留验证集，保留完整文档，因此验证Token数可能略超过目标；其余文档全部进入训练集，不再按原Bucket/Source配额裁剪。真实Token数以finalization_report.json的actual_train_tokens、actual_validation_tokens为准。报告passed仍表示原实验配额是否达标；accepted_for_use表示显式接受且训练/验证非空、无父文档交集；execution_passed决定CLI退出状态。不可把accepted_for_use解释为原1B配比已通过。原mixture_report.json不修改。

首次执行无需overwrite；已有该轮部分Tokenize/Finalize输出时才加overwrite重建对应阶段。成功输出仍位于当前Run的final/，可继续按上一节导出ms-swift缓存。

### Cache职责

- Cache 只执行字段提取、NFC、明确控制字符清理、许可证检查、极端质量过滤和廉价标签。
- 不执行 Tokenize、MinHash、语义质量模型、新增汉字全局统计或 Packing。
- PGCA Feature IDs 不写入数据。
- 当前 V2 不调用任何旧版 `prepare/prescan/sample` 模块。
