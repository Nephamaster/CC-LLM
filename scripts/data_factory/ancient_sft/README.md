# 古汉语 SFT 数据构建

实现依据：[ancient_sft.md](../../../agent/blueprint/data_build/ancient_sft.md)。与 CPT Data Factory 隔离，不修改 V1，不生成教师模型答案，不启动训练。

## 输入与依赖

从项目根目录执行。默认输入根为 `data/downstream/ancient`，本地/服务器保持相同相对路径。配置见 `config.yaml`。

- Gujilab：`raw/chinese-classical-corpus/{corpus,translate,punctuate}.jsonl`。
- HistoryTrans：`raw/HistoryTrans/HistoryTrans-{train,validation,test}.jsonl`。仅 train 入候选；validation/test 作为所有来源的排除集，不混作本实验 Dev。
- C³Bench：`benchmark/c3bench/C3_bench.json`，读取 **classical_Chinese/modern_Chinese**，不是通用 question/answer 字段。
- Fùxì：`benchmark/FuxiBench/test_data/*.json`，并排除 `dev_fewshot.json` 的示例。
- 不含 EvaHan。标点初始目标 5,000 条全部来自 Gujilab，并控制实际监督 token 占比，报告明确标识专家数据未纳入。

安装轻量构建依赖：

```bash
python -m pip install -r scripts/data_factory/ancient_sft/requirements.txt
```

Python 的 SQLite 需支持 FTS5 trigram（用于原文子串检索，检索后仍验证完整包含）；不需要额外模型服务。

`finalize` 使用当前环境已安装的 ms-swift，仅记录软件及依赖版本，不校验 Git commit、工作区或源码指纹。旧 manifest 中的 swift_commit 已废弃，不参与数据配置比较。不会因为普通安装或升级依赖阻断流程；接口不兼容或实际编码异常仍会正常报错。不下载模型权重，仅加载 tokenizer/processor。

编码契约：现有 Char tokenizer、`template=qwen3`、`enable_thinking=false`、`template_backend=swift`、`loss_scale=default`、`use_chat_template=true`、最大长度 2048。**不是**无空 think 前缀的 `qwen3_nothinking` 模板，也不是 CPT 的 `loss_scale=all`。按官方模板生成完整 `input_ids/labels` 后统计，assistant loss 含模板实际参与 loss 的结束符等 token。超长完整样本剔除，不切断问答。

## 执行

```bash
python -m scripts.data_factory.ancient_sft.run inspect
python -m scripts.data_factory.ancient_sft.run prepare
python -m scripts.data_factory.ancient_sft.run finalize
```

`inspect` 输出路径、哈希、首条记录 Schema 和缺失项。缺少 HistoryTrans 保留分区、任一 Benchmark 或 tokenizer 时，正式 `prepare` 拒绝运行。

仅验证少量真实数据可用：

```bash
python -m scripts.data_factory.ancient_sft.run prepare \
  --diagnostic --limit 300 \
  --output-dir data/downstream/ancient/train/smoke
```

`--limit` 是每个训练来源的原始记录窗口，不缩小 Benchmark 排除集。诊断模式只允许缺少 HistoryTrans 保留分区，**无法 finalize**；不要将诊断候选用于训练。

输出阶段目录已有内容时直接拒绝，不自动删除、不复用部分任务。失败、修改配置/源文件/审核文件/实现后，换新 `--output-dir` 从 prepare 开始；同一轮 finalize 使用相同目录。不存在隐含 overwrite。

目录迁移不改变数据版本。当前统一布局是 `raw/` 原始数据、`benchmark/` 测试数据、`train/prepared/` 候选和 `train/final/` 最终数据。本地可以只保留部分文件，服务器 prepared 无须为目录调整而重建。

已完成 v2 prepare 的数据可以继续 finalize：核验 v2 篇章划分标记、数据配置语义、输入文件 SHA256/大小和数据库哈希；识别固定旧→新路径映射，不修改旧 manifest 或历史 metadata ID。不再以整个 Python 模块的实现指纹判断是否必须重跑 prepare。新 final manifest 额外记录 effective_config/effective_inventory。已有 final 不会被覆盖。

v1→v2 的分组变化仍需 repartition；如需重建划分，必须指定不同的空输出目录，不能读写同一个 prepared：

```bash
python -m scripts.data_factory.ancient_sft.run repartition \
  --from-prepared data/downstream/ancient/train/prepared \
  --output-dir data/downstream/ancient/train/rebuild
python -m scripts.data_factory.ancient_sft.run finalize \
  --output-dir data/downstream/ancient/train/rebuild
```

默认输出为 `data/downstream/ancient/train`，上述重划分示例另外写入其 `rebuild/`，原始产物不覆盖、不自动替换。必须同步整个模块和新版 config；只有 final metadata、没有 prepared/candidates.sqlite 时不能执行此复用，需要完整 prepare。复用前核验已知旧实现、过滤规则、OpenCC 版本、所有输入/排除集与数据库哈希；除本次明确的路径迁移外，仅允许修改划分/采样配置。只读取旧 eligible 表，不从 raw candidates 或审核隔离区恢复记录。历史已删记录不恢复，旧 manifest 保持不变。

## 分组、去重与审核

1. 训练正文仅 NFC，保留简繁、异体和扩展区字；判重文本用 OpenCC t2s、去标点空白，不回写训练文本。
2. 保留 source_id、原始记录 ID、pair_group、document_group/document_groups。在 corpus 原文中进行精确子串定位，优先以卷/章或可追溯原记录作为边界；再使用明确的来源篇章标签，无可靠信息才回退整书。《说文》以字头条目为组。跨书引用不再把整书传递合并。
3. 古文 5-gram MinHash（64 permutations，16×4 LSH）召回候选，实际 Jaccard 达到阈值才合并；这是近似召回，不承诺穷尽语义相似或所有包含关系。对同一任务的近重复组及同一规范化输入各保留一条。Gujilab 优先，HistoryTrans 仅作补充。
4. HistoryTrans 从 corpus 恢复作品/篇章；无法恢复或作品归属有歧义时隔离。先按任务、书籍覆盖确定完整 Dev 候选组，再排除 Train 中同篇章、多出处交集、同近重复组、同译文、完整包含或共享至少 24 字连续片段的记录。无法对齐章名的同书边界从严排除，不随机拆句，不合并整部书。不足部分由 Gujilab 补采或报告缺额。
5. Benchmark 规范化整段精确匹配直接排除，不能人工放行。通过 24 字采样锚点召回并验证的共享片段进入待审核；这是近似召回，短错位片段和语义改写可能漏检，不等于穷尽去污染。短答案本身相同（例如作者名）仅统计，不自动删除整部作品。短问答完整重合仍排除。
6. 所有可选字书/出处候选必须经过源数据质量审核。代码只摘录原始字段和原释，绝不自动生成现代释义；默认不将未审核样本导出。

首次 prepare 的审核记录在 `prepared/audit.jsonl`；新划分的边界排除在 `prepared/partition_audit.jsonl`。repartition 不复制旧审核文件，旧目录仍保留。可将以下结构写入独立 JSONL，再在配置的 `reviews` 中指定路径，并完整 prepare 新版本（审核变化不允许复用）：

```json
{"id":"审核清单中的完整内容ID","decision":"allow","reviewer":"审核人","reason":"核对来源后的具体依据"}
```

`decision` 也可为 `reject`。ID 绑定原候选内容，审核文件本身进入输入哈希。允许疑似重合需要解释为何不构成测试泄漏；精确匹配的硬排除不能被此机制绕过。

## 配额与产物

Train 初始目标：c2m 38K、m2c 10K、标点 5K，核心 53K；字书、出处各最多 4K，仍需审核。HistoryTrans 最多 13K 新增 c2m，不足由 Gujilab 补充。普通任务单书上限 20%；字书专门来自《说文》，不应用该上限。按作品、篇章、长度档和简繁类型交错抽样，避免只取短文本。

Dev 独立目标为 c2m/m2c/punctuate 各 600 条，合计 1,800；每任务至少 5 部作品、10 个独立组，单书至多 20%。先预留约 1.5 倍候选容量，留出的是完整组而非随机句子，实际编码后再选样。若安全边界下无法满足条件，应报告失败，不放宽隔离或自动更换随机种子。

标点在其他任务选完后编码入选，累计 assistant loss token 不允许超过全部监督 token 的 40%，完整样本超过剩余预算就跳过，绝不截断答案。达到初始 5K 后若占比仍低于 30%，可继续补采至最多 6K（`punctuation_max_samples`）；仍不足则验收失败。最终占比须在 30%–40%，条数缺额也明确报告，不冒充达标。该范围是实验起点，不等同于真实梯度权重。

默认最终目录 `data/downstream/ancient/train/final/`：

- `train.jsonl`、`dev.jsonl`：仅标准 messages。
- `dev_by_task/{c2m,m2c,punctuate}.jsonl`：用于分任务评估。训练端可计算预先约定的 `(loss_c2m + loss_m2c + loss_punctuate) / 3`，不要仅按合并 eval_loss 选择 checkpoint；数据管线不自动修改 Trainer 的评估逻辑。
- `train.metadata.jsonl`、`dev.metadata.jsonl`：按行一一对应，包含稳定 ID、row_number、messages 哈希、来源/分组、审核、实际长度和 assistant loss token 数。
- `report.json`：合格候选数、实际数量/缺额、来源/作品/时代/简繁启发式分布、分组数量、token 长度均值/P95/P99及 loss token 比例。
- `manifest.json`：完整配置、所有原始文件 SHA256、来源版本/许可证（配置 provenance 提供，未提供明确标 unknown）、当前 ms-swift/依赖版本和输出哈希。

`execution_passed` 只代表流程完成，`quotas_passed/core_quotas_passed` 分别记录完整/核心配额。`ready_for_training` 还要求核心 Train 配额、三项 Dev 的数量/作品/独立组/集中度验收，以及标点 token 占比均通过；`dev_checks`、`dev_coverage_passed`、`punctuation_budget` 给出明细。没有满足条件的输出不能作为正式验收通过的数据集，程序退出成功也不等于验收成功。

训练需继续使用上述模板契约，以 `swift sft` 消费 train/dev，并显式指定验证集。不要用 CPT 导出脚本处理这份 SFT 数据；该脚本的非聊天全文 loss 契约不同。

当前 SFT 检查不能消除或证明不存在 Phase2 CPT 污染；manifest 始终保留 `cpt_contamination_audited=false`。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_ancient_sft.py
```

普通测试以临时文件与可控编码器验证数据逻辑，不冒充真实 ms-swift 编码验收；真实 finalize 使用当前框架自身模板，检查实际 labels 和长度，不做软件指纹准入。

在已安装 ms-swift 的环境中，启用实际 Char tokenizer / 模板及临时全链路集成测试：

```bash
ANCIENT_SWIFT_INTEGRATION=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q tests/test_ancient_sft.py
```
