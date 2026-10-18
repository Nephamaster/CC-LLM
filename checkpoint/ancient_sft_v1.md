# 古汉语 SFT 构建实现与验收边界

## 2026-10-18 决策

- 以 `agent/blueprint/data_build/ancient_sft.md` 为准，旧 Dashboard 的来源 60/25/15 比例已撤销。独立入口 `scripts/ancient_sft`，不修改 CPT V1/V2 行为。
- 用户确认输入在 `data/downstream/ancient`，不含 EvaHan；标点 24K 全部使用 Gujilab。Train 上限 80K，核心 72K；可选字书/出处未通过审核不导出。
- 服务器使用 Nephamaster/ms-swift；本地 `/home/nepham/Lab/ms-swift` 与 GitHub HEAD 均为 `1516d78d627381c2072be80037a7a64c3811480d`。该仓库已有用户训练脚本未跟踪文件，未修改。
- 编码契约是现有 Char tokenizer + `qwen3/enable_thinking=false/loss_scale=default`，不是 `qwen3_nothinking`，也不是 CPT 全文 loss。直接调用实际框架的 train-mode encode，按 causal shift 后非 -100 labels 计数。模型权重不加载，最大 2048，超长拒绝而非拆分问答。

## 数据工程关键点

- 本地 C³Bench 实际字段为 classical_Chinese/modern_Chinese，文件有 10,000 条记录。Fùxì test 共 11,490 条，few-shot 文件 105 条，均进入排除索引；不沿用 CPT 中 C3 可缺省的配置。
- HistoryTrans 使用 inputs/truth，不使用 results 或 BLEU 作为真值/人工质量标注。其官方 validation/test 必须同时排除其他来源训练候选。
- 简繁转换只用于比较；保留正文。双向翻译、共享译文与近重复连通分组，默认保守整书划分。无来源 HistoryTrans 先通过已知平行对或 corpus 全文包含恢复作品；无法恢复时隔离，Gujilab 承担不足配额。
- 去重采用 64-permutation 5-gram MinHash（16×4 LSH）+ 实际 Jaccard 校验，召回是近似的。去污染采用规范化精确匹配及采样锚点召回的 >=24 字共享连续片段；不宣称覆盖全部语义相似或短片段包含。
- Benchmark 精确匹配不可被人工审核绕过；疑似重合及所有可选知识任务先隔离。审核 ID 绑定候选内容，审核文件纳入哈希。短答案重合只作统计，避免按作者名等标签误删全书。
- SQLite 保存候选与近重复索引。prepare/finalize 不覆盖已有阶段；配置、源文件、实现变化需新目录。诊断/限量产物禁止 finalize；阶段间重新核验输入和数据库哈希。

## 验证

- 27 项定向测试通过（包含 V2 benchmark reader 回归 6 项），涵盖保留分区排除、跨源/双向归组、近重复、知识审核、禁止放行精确污染、篡改与陈旧产物拒绝、超长拒绝、输出与 metadata 对齐。
- 使用隔离 CPU 环境及真实 fork 完成临时全链路 finalize。依赖为 Transformers 5.19.0、tokenizers 0.23.3、PyTorch 2.14.1+cpu；这是本地验证环境，不等于已核验服务器依赖版本。运行 Manifest 记录实际版本。
- 真实数据每个训练来源前 3,000 条的诊断已跑通，使用完整 Benchmark 排除集与完整 corpus 定位目录。可选候选仍需审核；不代表全量去重/配额验收通过。
- 对上述真实候选分别抽取 c2m/m2c/punctuate 各 100 条，使用实际 ms-swift 编码，共 300 条全部成功且不超过 2048。最大长度分别 203/264/511；实际 loss token 合计分别 4,493/3,156/20,962。
- 所有实验产物、依赖环境位于 `/tmp/ccllm-ancient-*`；未运行训练，未生成可用于生产训练的全量版本。

## 当前阻塞与后续

初次实现时缺少 HistoryTrans 官方 validation/test，正式 prepare 因此拒绝运行。现用户已补齐三个分区，统一放在 `data/downstream/ancient/HistoryTrans/HistoryTrans-{train,validation,test}.jsonl`；配置和说明已同步，无须修改 JSONL 读取逻辑。

已完整读取并检查 inputs/truth 字段：train 749,988 条，validation/test 各 7,653 条。预检 `ready=true`、无缺失输入；两个保留分区均成功加入排除索引。嵌套目录测试及实际 ms-swift 集成回归共 27 项通过。此验证不代表已完成全量 prepare/finalize。

后续执行 inspect → prepare → 审核（如需）→ 新版本 prepare → finalize；旧诊断产物不可因补齐输入而直接转为正式产物。

检查 report 的实际任务数、缺额、Train/Dev 是否非空及模板信息。来源版本/许可证未提供时明确 unknown；SFT 清洁不代表 CPT 未污染，报告保留 `cpt_contamination_audited=false`。

## finalize 对无 Git 源码目录的兼容修复

- 服务器的 ms-swift 目录没有可识别的 `.git`，旧 SwiftEncoder 在 `git rev-parse HEAD` 阶段失败，尚未创建 final。不是数据损坏，不应通过 git init、修改 manifest 或重跑全量 prepare 解决。
- 新增 `provenance.py` 和 `provenance_profile.json`，从锁定 commit 的 Git 对象归档计算全部 540 个 `swift/**/*.py` 路径/内容聚合指纹。有 Git 时仍检查 commit、工作区和源码；无 Git 时直接校验源码，记录实际 Git commit=null、参考 commit 和验证方式。若目录只是位于另一个 Git 仓库之内，不误用父仓库的 commit。
- 旧 prepare 实现指纹 `a26014c8a6984f426ff9457f69527e9d5979238043126d460bda405f620b5c91` 支持本次有界兼容：核对原始准备模块及 pipeline 准备代码片段未变；仍检查配置、输入、数据库，不改写旧 manifest。未知版本、额外代码模块或准备逻辑变化继续拒绝。
- 服务器需同步整个 `scripts/ancient_sft/`（含新增 Python/JSON 文件），保持原配置和产物，然后直接重试 finalize，无需 overwrite。源码指纹不符时明确报错，不默默降级为“版本已验证”。
- 验证：46 项测试通过；从锁定 commit 导出无 `.git` 的真实 Swift 副本，在隔离 CPU 环境完成新/旧 prepare 两种临时数据的实际编码及 finalize；覆盖源码修改/缺失/增加、Git 不可用、父仓库误识别、旧产物输入/数据库篡改拒绝等。仅有 PyTorch 弃用警告，未在服务器运行全量 finalize 或训练。
