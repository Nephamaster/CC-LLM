# 古汉语 SFT 数据构建

实现依据：[ancient_sft.md](../../agent/blueprint/data_build/ancient_sft.md)。与 CPT Data Factory 隔离，不修改 V1，不生成教师模型答案，不启动训练。

## 输入与依赖

从项目根目录执行。默认输入根为 `data/downstream/ancient`，本地/服务器保持相同相对路径。配置见 `config.yaml`。

- Gujilab：`chinese-classical-corpus/{corpus,translate,punctuate}.jsonl`。
- HistoryTrans：`HistoryTrans/HistoryTrans-{train,validation,test}.jsonl`。仅 train 入候选；validation/test 作为所有来源的排除集，不混作本实验 Dev。
- C³Bench：`benchmark/c3bench/C3_bench.json`，读取 **classical_Chinese/modern_Chinese**，不是通用 question/answer 字段。
- Fùxì：`benchmark/FuxiBench/test_data/*.json`，并排除 `dev_fewshot.json` 的示例。
- 不含 EvaHan。标点目标 24,000 条全部来自 Gujilab，报告明确标识专家数据未纳入。

安装轻量构建依赖：

```bash
pip install -r scripts/ancient_sft/requirements.txt
```

`finalize` 必须在已安装用户 ms-swift fork 的环境执行；当前锁定提交 `1516d78d627381c2072be80037a7a64c3811480d`。要求可定位到该 Git checkout，且 `swift/` 无未提交变动；其他目录的训练脚本变动不影响检查。不下载模型权重，仅加载 tokenizer/processor。

编码契约：现有 Char tokenizer、`template=qwen3`、`enable_thinking=false`、`template_backend=swift`、`loss_scale=default`、`use_chat_template=true`、最大长度 2048。**不是**无空 think 前缀的 `qwen3_nothinking` 模板，也不是 CPT 的 `loss_scale=all`。按官方模板生成完整 `input_ids/labels` 后统计，assistant loss 含模板实际参与 loss 的结束符等 token。超长完整样本剔除，不切断问答。

## 执行

```bash
python -m scripts.ancient_sft.run inspect
python -m scripts.ancient_sft.run prepare
python -m scripts.ancient_sft.run finalize
```

`inspect` 输出路径、哈希、首条记录 Schema 和缺失项。缺少 HistoryTrans 保留分区、任一 Benchmark 或 tokenizer 时，正式 `prepare` 拒绝运行。

仅验证少量真实数据可用：

```bash
python -m scripts.ancient_sft.run prepare \
  --diagnostic --limit 300 \
  --output-dir data/downstream/ancient_sft_smoke
```

`--limit` 是每个训练来源的原始记录窗口，不缩小 Benchmark 排除集。诊断模式只允许缺少 HistoryTrans 保留分区，**无法 finalize**；不要将诊断候选用于训练。

输出阶段目录已有内容时直接拒绝，不自动删除、不复用部分任务。失败、修改配置/源文件/审核文件/实现后，换新 `--output-dir` 从 prepare 开始；同一轮 finalize 使用相同目录。不存在隐含 overwrite。

## 分组、去重与审核

1. 训练正文仅 NFC，保留简繁、异体和扩展区字；判重文本用 OpenCC t2s、去标点空白，不回写训练文本。
2. 保留 source_id、原始记录 ID、pair_group、document_group。缺少统一章卷标识时保守使用整部作品为分组；《说文》以字头条目为组。共享古文、译文和近重复记录连接分组后再确定 Train/Dev。
3. 古文 5-gram MinHash（64 permutations，16×4 LSH）召回候选，实际 Jaccard 达到阈值才合并；这是近似召回，不承诺穷尽语义相似或所有包含关系。对同一任务的近重复组及同一规范化输入各保留一条。Gujilab 优先，HistoryTrans 仅作补充。
4. HistoryTrans 先通过精确/近重复平行对恢复分组，再尝试定位到 corpus 原文（24 字锚点召回，完整包含验证）；多作品命中会连接分组。仍无法恢复来源时隔离，尤其不猜测短句归属。不得随机分配这些未知来源句子来宣称篇章隔离；不足部分从 Gujilab 补采，仍不足则报告缺额。
5. Benchmark 规范化整段精确匹配直接排除，不能人工放行。通过 24 字采样锚点召回并验证的共享片段进入待审核；这是近似召回，短错位片段和语义改写可能漏检，不等于穷尽去污染。短答案本身相同（例如作者名）仅统计，不自动删除整部作品。短问答完整重合仍排除。
6. 所有可选字书/出处候选必须经过源数据质量审核。代码只摘录原始字段和原释，绝不自动生成现代释义；默认不将未审核样本导出。

审核记录在 `prepared/audit.jsonl`。可将以下结构写入独立 JSONL，再在配置的 `reviews` 中指定路径，并构建新版本：

```json
{"id":"审核清单中的完整内容ID","decision":"allow","reviewer":"审核人","reason":"核对来源后的具体依据"}
```

`decision` 也可为 `reject`。ID 绑定原候选内容，审核文件本身进入输入哈希。允许疑似重合需要解释为何不构成测试泄漏；精确匹配的硬排除不能被此机制绕过。

## 配额与产物

Train 目标：c2m 38K、m2c 10K、标点 24K、字书最多 4K、出处最多 4K。HistoryTrans 优先预留最多 13K 新增 c2m，其余由 Gujilab 补充。普通任务单部作品最多占该任务目标的 20%；字书任务专门来自《说文》，不应用该上限。Dev 采用稳定分组哈希约 2%，另行采样，不从 Train 的 80K 扣除；整书分组会使实际比例偏离目标，必须检查报告。

最终目录 `data/downstream/ancient_sft_v1/final/`：

- `train.jsonl`、`dev.jsonl`：仅标准 messages。
- `train.metadata.jsonl`、`dev.metadata.jsonl`：按行一一对应，包含稳定 ID、row_number、messages 哈希、来源/分组、审核、实际长度和 assistant loss token 数。
- `report.json`：合格候选数、实际数量/缺额、来源/作品/时代/简繁启发式分布、分组数量、token 长度均值/P95/P99及 loss token 比例。
- `manifest.json`：完整配置、所有原始文件 SHA256、来源版本/许可证（配置 provenance 提供，未提供明确标 unknown）、ms-swift commit、依赖版本和输出哈希。

`execution_passed` 只代表流程完成，`quotas_passed/core_quotas_passed` 分别记录完整/核心配额。`ready_for_training` 要求完成严格预检、真实编码、分组隔离且 Train/Dev 均非空；它不代表配额已满足或人工审核覆盖所有原始译文。启动训练前核对实际数量和任务缺额，不只检查程序退出码。

训练需继续使用上述模板契约，以 `swift sft` 消费 train/dev，并显式指定验证集。不要用 CPT 导出脚本处理这份 SFT 数据；该脚本的非聊天全文 loss 契约不同。

当前 SFT 检查不能消除或证明不存在 Phase2 CPT 污染；manifest 始终保留 `cpt_contamination_audited=false`。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_ancient_sft.py
```

普通测试以临时文件与可控编码器验证数据逻辑，不冒充真实 ms-swift 编码验收；真实 finalize 使用框架自身模板，且严格校验版本。

在已安装锁定 ms-swift 的环境中，启用实际 Char tokenizer / 模板及临时全链路集成测试：

```bash
ANCIENT_SWIFT_INTEGRATION=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q tests/test_ancient_sft.py
```
