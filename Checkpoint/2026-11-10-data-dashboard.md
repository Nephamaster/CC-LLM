# 实验数据看板

统计日期：2026-11-10。代码基线：master / c072524。按当前训练与评估入口统计最终输入，原始、中间、备份文件不重复计入数据量。结果同步写入根目录 `data_dashboard.xlsx`，未修改训练代码、数据或模型。

## 计数口径

- 预训练语料存量采用历史最终报告的精确原文 token 数：不含聊天模板与额外 EOS。本次没有重新分词 Phase1/2 训练集；两份验证集已重新分词，原文 token 数和报告完全一致。报告从既有任务记录恢复，原报告文件已不在当前数据包中。
- 下游输入序列与有效监督 token 均为本次逐条重新编码实测：使用当前 Char 语义词表（154,019 个词项），ms-swift 4.6.0、Transformers 5.19.0、Tokenizers 0.23.3 与仓库当前模板。有效监督按 `labels[1:] != -100` 计数。计数只加载 tokenizer 和模板，没有载入模型权重或使用 GPU。
- 下面均为数据单遍统计，不乘训练 epoch，不计 DDP 尾部重复/补齐。音形记忆槽位不额外算作文本 token。1M=1,000,000，1B=1,000,000,000。
- 脚本、恢复报告副本、逐条计数结果和 tokenizer-only 计数模型全部在 `/tmp/cc-data-dashboard.iCn8I3/`，未放入代码目录。

## 一、预训练成品语料

| 阶段/语料池 | 训练文档数 | 训练原文 token | 验证文档数 | 验证原文 token | 依据 |
| --- | --- | --- | --- | --- | --- |
| Phase1：语义对齐 | 543,894 | 980,118,370 | 566 | 1,033,373 | 历史最终报告；验证集本次复核 |
| Phase2：继续预训练共享池 | 5,739,897 | 10,035,777,803 | 5,746 | 10,021,186 | 历史最终报告；验证集本次复核 |

文件位置为 `data/pretrain/phase1/{train,validation}` 与 `data/pretrain/phase2/{train,validation}`；本地训练部分分别有 6 和 58 个 tar.xz 分片，验证各 1 个 JSONL。这里描述本地数据快照，不据此推断服务器是否已经解压或训练完成。

### 当前预训练阶段使用量

| 训练阶段 | 训练有效监督 token | 验证有效监督 token | 说明 |
| --- | --- | --- | --- |
| Phase1 | 历史缓存可用量 979,857,374；默认上限 1B | 1,033,087 | 训练数来自此前 prepare_data 的 available 记录，本次未重算训练部分；验证本次实测 |
| Phase2-A | 100,000,000（默认预算） | 10,020,339 | 从 Phase2 共享池切出前一段监督目标 |
| Phase2-B | 9,900,000,000（默认预算） | 10,020,339 | 使用扣除 A 后的监督目标，A+B 默认上限 10B |

A/B 共享一份 Phase2 验证集，不重复累加成两份数据。其训练数是当前脚本默认监督预算，不是本次读取最新准备报告所得的实绩；缺少新 run 的 preparation_report.json，不能独立确认本轮实际预算和样本条数。A/B 的监督区间不重叠，分界处可共享输入上下文。

### 原始数据集来源

| 原始数据集 | Phase1 | Phase2 | 登记的采用范围 |
| --- | --- | --- | --- |
| CCI3-HQ | ✓ | ✓ | 本地 part 分片 |
| FineWeb-Edu-Chinese V2.2 | ✓ | ✓ | 4_5 子集 |
| WanJuan 1.0（万卷） | ✓ | ✓ | 文本分片 |
| FineWeb-Edu-English | ✓ | ✓ | sample/10BT 子集 |
| The Stack v3 | ✓ | ✓ | 本地下载子集；代码及 README/docs，经过许可过滤 |
| OpenWebMath | ✓ | ✓ | 数学网页 |
| FineWeb-zhtw | ✓ | ✓ | 繁体中文网页 |
| 中文 Wikisource | ✓ | ✓ | 20231201.zh |
| Chinese Cosmopedia | — | ✓ | 中文知识内容 |
| 中文 Wikipedia | — | ✓ | 20231101.zh |
| ECT-KRP | — | ✓ | v1.1.0 文本 |
| FineWeb2 Multilingual | — | ✓ | 日、韩、越、西班牙、俄、阿拉伯、印地语各一片 |
| peS2o v2 | — | ✓ | 只取 s2orc/train，不是 s2ag/train |

Phase1 共 8 个来源，Phase2 共 13 个来源。来源集合由历史实际入选 source_tokens 与最终报告交叉核对，范围来自删除数据构建代码前的来源登记。验证集从各阶段混合池分层切分；现有最终报告没有逐来源的验证明细，不保证每个极小来源在验证集都有样本。逐来源 token 只找到 Mixture 时的估算选择量，不能当作最终训练/验证精确账单，因此下面按最终报告有精确计数的类别列出。

### phase1 实际类别规模

| 类别 | 训练原文 token | 验证原文 token |
| --- | --- | --- |
| 中文通用 | 299,190,788 | 302,323 |
| 中文知识 | 200,098,127 | 201,248 |
| 英文 | 150,915,964 | 150,348 |
| 中英混排/技术文本 | 99,793,910 | 116,773 |
| 代码/数学/结构化 | 79,403,351 | 101,167 |
| 新增汉字覆盖增强 | 150,716,230 | 161,514 |

### phase2 实际类别规模

| 类别 | 训练原文 token | 验证原文 token |
| --- | --- | --- |
| 中文通用 | 2,997,035,197 | 3,000,280 |
| 中文知识 | 2,500,856,328 | 2,518,060 |
| 英文/多语/中英混排 | 2,177,528,571 | 2,000,004 |
| 数学/代码/科学 | 1,331,698,553 | 1,501,274 |
| 新增汉字覆盖增强 | 1,028,659,154 | 1,001,568 |

新增汉字覆盖增强是从相关原始来源中二次筛选的类别，不是额外的独立原始数据集；不要再将其与来源表重复相加。

## 二、下游训练与验证数据

| 任务 | 划分 | 所用数据 | 样本数 | 输入序列 token | 有效监督 token |
| --- | --- | --- | --- | --- | --- |
| CSC | 训练 | CSC 去重混合纠错集（成品未保留逐来源标签） | 352,019 | 31,884,567 | 14,886,197 |
| CSC | 验证 | CSCD-NS dev | 5,000 | 619,008 | 294,504 |
| CGEC | 训练 | CGED + FCGEC 训练来源 | 53,759 | 5,933,551 | 2,805,462 |
| CGEC | 验证 | MuCGEC 验证来源 | 1,137 | 112,595 | 53,495 |
| 古汉语 | 训练 | Gujilab + HistoryTrans | 53,000 | 7,178,773 | 3,076,464 |
| 古汉语 | 验证 | Gujilab，三个任务各 600 条 | 1,800 | 340,806 | 150,267 |

文件分别为 `data/finetune/csc/{csc_mix_dedup,cscd_dev}.jsonl`、`data/finetune/cgec/{train,validation}.jsonl`、`data/finetune/ancient/{train,dev}.jsonl`。CGEC 来源依据历史构建脚本：训练只取 train_CGED/train_FCGEC，验证取 valid_MuCGEC，并排除评测来源。

### 古汉语任务与来源构成

| 任务 | 训练样本 | 验证样本 | 实际原始来源 |
| --- | --- | --- | --- |
| 古译今 | 38,000 | 600 | 训练：Gujilab 37,743 + HistoryTrans 257；验证：Gujilab |
| 今译古 | 10,000 | 600 | Gujilab |
| 断句/标点 | 5,000 | 600 | Gujilab |

按当前最终 metadata，古汉语训练总计 Gujilab 52,743 条、HistoryTrans 257 条，验证全部来自 Gujilab。这里写实际入选量，不把旧规划的 HistoryTrans 配额当作实得。metadata 的 messages 与最终训练/验证文件逐行核对一致。

所有下游样本本次均有有效监督，且没有超过当前长度上限。训练最大序列长度：CSC 526、CGEC 700、古汉语 855；对应配置上限为 768、768、2048。

古汉语旧 metadata 记录的监督 token 为训练 3,288,464、验证 157,467，与当前模板实测不同；当前口径为上表的 3,076,464 / 150,267。每条相差 4 个 token，序列量与样本内容不变；这是旧记录与当前 labels 的监督口径差异，本次没有重跑旧框架归因。本看板采用当前仓库模板的逐条实测值。

## 三、CPT 后评估数据（不统计 token）

| 评估项 | 数据集/范围 | 使用划分/协议 | 规模/本地情况 |
| --- | --- | --- | --- |
| PPL | Skywork 中文领域语料：金融、游戏、通用、政务、影视、科技 | 独立评估文本 | 1,522 篇，6 个本地 JSONL |
| MMLU | 57 个学科 | test，5-shot | 运行时通过 HF 镜像获取/复用缓存；本次未下载重数 |
| ARC | 仅 ARC-Challenge | test，0-shot | 同上，不包含 ARC-Easy |
| GSM8K | 数学应用题 | test，5-shot | HF 镜像获取/复用缓存 |
| SQuAD v2 | 阅读理解与无答案判断 | validation，0-shot | HF 镜像获取/复用缓存 |
| CMMLU | 67 个中文学科 | test，5-shot | HF 镜像获取/复用缓存 |
| CLUE | AFQMC、TNEWS、IFLYTEK、CMNLI、CSL、WSC2020、CHID、C3、CMRC2018 | validation，0-shot | 9 项任务，HF 镜像获取/复用缓存 |

PPL 的 6 个文件与文档数：zh_finance 500；zh_game 158；zh_general 240；zh_government 192；zh_movie 346；zh_tech 86。

## 四、微调后 benchmark（不统计 token）

| 微调任务 | benchmark | 划分/范围 | 本地样本规模与说明 |
| --- | --- | --- | --- |
| CSC | CSCD-NS | test | 5,000 条 |
| CSC | LEMON V2 | 7 领域 | 21,731 条；与 CSCD-NS 合计 26,731 条 |
| CGEC | FCGEC | valid | 2,000 条，有标签，可本地评分 |
| CGEC | FCGEC | 公开 test | 3,000 条，无公开 gold；当前仅导出预测/提交文件 |
| CGEC | NaCGEC | test | 5,869 条，有多参考答案 |
| 古汉语 | C³Bench | 分类、出处检索、NER、标点、翻译 | 10,000 篇 × 5 任务 = 50,000 个任务样本 |
| 古汉语 | Fùxì / FuxiBench | 21 任务 | 11,490 条；部分任务需模型裁判 |

LEMON 各域：car 3,245；cot 993；enc 3,271；gam 393；mec 1,942；new 5,887；nov 6,000。

以下也在本地，但不属于当前默认评估入口：rSIGHAN13/14/15 分别 1,000 / 1,062 / 1,100 条；ECSpell 的 law/med/odw 各 500 条；NaCGEC dev 500 条可显式选择。评估目录中的 FCGEC train/train_filtered 为原始训练资料副本，不计作 benchmark test。Fuxi dev_fewshot 是官方提示示例，不计入 SFT 训练/验证规模。

## 统计依据与限制

- 预训练来源与精确类别量：恢复的 Phase1/2 最终报告、实际 Mixture 来源登记、历史 sources.yaml；其副本在临时目录，未计算任何代码/权重哈希。逐来源最终 train/val 精确 token 明细缺失，未用计划配比伪造。
- SFT 及 CPT 验证量：临时计数模型、`count_text_tokens.py`、`count_native_templates.py`，全部统计脚本均在 /tmp；结果为 `dashboard_counts.json` 与 `native_counts_*.json`。
- 下游统计使用正式模板逐条 encode，未训练模型，未修改数据。当前模板监督量可能与旧框架生成的 metadata 不同，因此以本次实际 labels 为准。
- benchmark 名称与协议来自当前 `scripts/evaluation/`，本地数量直接读取最终文件。通用 benchmark 没有为本看板另行下载；不同远程数据版本的样本数不在这里臆填。
- 不把 token 单遍统计解释为实际全部训练曝光量；epoch、采样器补齐、用户环境变量覆盖和具体 checkpoint 训练进度需以对应训练报告为准。
- `data/model_build/` 中的字表、Unihan 与结构表用于词表/音形构建，不是训练/验证样本集，未计入上述训练 token。

## Excel 回填

已更新 `data_dashboard.xlsx` 的「数据看板」「语义对齐数据」「全参CPT数据」三张表，区分原计划比例、实际成品 token 与 A/B 默认监督预算，补齐原始来源及下游训练/评测数据。其余三张表、原公式与内嵌图片保留。

已重新加载工作簿，核对类别合计、训练/验证与 SFT 数值、8/13 个来源及 XML 完整性；未在 Excel 客户端人工检查显示效果。原工作簿备份与回填脚本在 `/tmp/cc-data-dashboard.iCn8I3/`。

