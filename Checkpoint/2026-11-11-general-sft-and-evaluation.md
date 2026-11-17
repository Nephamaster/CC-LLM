# General-SFT 与评估协议改造

本轮依据根目录 TODO.md 实施。工作区为 master，核对时 HEAD 为 `2cae51f`；未切分支、提交或推送。保留用户先前删除 experiment.xlsx、phase2_encoding_probe.py 的状态，不改 src 模型结构或既有 CPT 超参。

## 已确认的决定

- Base、Char-CPT、PGCA-CPT 分别进行 General-SFT，采用 LoRA，验收后合并为完整 M-INS，再训练独立 CSC/CGEC/ACT Adapter。
- COIG/Tulu 原始数据尚未就绪；先实现数据准备、可配置监督预算、训练与评估入口，不编造正式数据量、配比验收或模型成绩。
- 正式预算数值待数据统计后确定。完整对话与 EOS 不为凑预算而截断，实际目标量可能略超配置；准备量、运行训练量、选中 checkpoint 的训练量分别记录。跨路线严格相等仍待核定。

## 关键核对与修复

1. **NaCGEC 的“不公开 Test gold”前提不成立。** 官方 README 明确记录 2023-08-22 已发布参考答案，本地也存在。仍按 TODO 改为默认 Dev 评分、Test 输入导出；Test 本地评分须显式指定参考文件，并逐条核对官方 Test 输入/ID。旧隐式 gold 输入、来源不同的旧预测会被拒绝。不能仅因默认协议改变，就断言历史 Test 结果无效。[官方说明](https://github.com/masr2000/NaCGEC)
2. **MaxMatch 金标顺序存在真实低估。** 原包装层直接使用路径回溯的逆序编辑；评分器 matchSeq 依赖源位置顺序。用本地 NaCGEC Dev 第一条参考答案作为预测，旧 MaxMatch 为 50、双评分器均值为 75；修正排序后两者均为 100。这里只证明该样例和相应回归，未推断所有历史结果的影响幅度。vendor 算法未改，只修正调用端金标顺序。
3. FCGEC 主指标为 ChERRANT F0.5；NaCGEC 为两种 F0.5 的算术平均，F1 为补充。多参考及 Insert/Modify/Switch 还原保持，已验证本地完整 FCGEC valid 能还原、官方格式示例正确。完整排行榜等价性仍未证明。
4. C³Bench 官方仓库当前只列出数据和简短 README，未取得可直接核对的评估实现。现有 NER/层级出处/标点仍明确为内部重建，`official_code_equivalence=false`，不伪称官方等价。[官方仓库](https://github.com/SCUT-DLVCLab/C3bench)

## 实施内容

- `scripts/train/instruction/prepare.py`：本地 JSON/JSONL/Parquet；多轮角色/正文过滤、首轮问题去重、分组混合、Train/Dev 划分、来源/原 ID/任务/许可保留。异构来源元数据统一可写入 Arrow 的标量类型。
- `decontamination.py`：evaluation 环境收集专项与 LMEH 基准输入；train 环境用 Aho-Corasick 检测规范化完整输入子串。最短 24 字符，不等于完整 n-gram/改写去污染。默认中文目标 40%、Tulu 上限 60%，数学/代码比例不预设研究结论，实际分布入报告。
- `run.sh`、`budget.py`、`finish.py`：独立 General-SFT；完整对话视图，所有 Assistant EOS/最后一轮 Prompt IDs 检查；每次保存轻量生成检查；保留的候选按有限 Dev loss 排序，完整生成/LoRA/Cache 验收；合并后复验，通过才发布 INS 完整权重。
- 复用下游模板与验收，不另造 LoRA 加载链；模板恢复逻辑集中到 `training_template`。合并配置 im_end/endoftext 停止，去掉继承的全局 max_new_tokens。已知仓库插件路径可随仓库迁移解析。
- 总训练入口三条路线接入 instruction，专项默认从合并 INS 出发。总评估读取 INS/专项指针；LMEH 显式 Base/Instruct 模式，默认 Instruct，few-shot 固定多轮，实际用保存的非思考 Swift 模板，核对每个 Prompt ID，保存前 32 个样例和停止配置。
- CSC 主报完整原始输出，清理/ChERRANT 为补充。专项按任务设置生成预算，逐条记录；三项统一统计停止、截断、重复率。后端 stop 不冒充精确 EOS 生成率。
- 集中冒烟接入 General-SFT/合并、INS 通用评测、NaCGEC Dev/Test 两条协议。缺少指令数据时如实失败/阻断；report 模式明确不验收少步模型生成质量。
- 根 README、各训练/评估说明、pipeline.md 的相关主线以及 TODO 状态已更新。新依赖只有 train 中未锁版本的 pyahocorasick，无运行时 Git/哈希检查。

## 本轮验证和证据

临时环境 `/tmp/cc-todo-env/`；脚本/样例/日志 `/tmp/cc-todo-check/`。Python 3.11.16、ms-swift 4.6.0、Transformers 5.19.0、Torch 2.15.0+cpu、lm_eval 0.4.13。正式环境仍按项目 Python 3.12 构建。本轮不是正式 GPU 实验。

| 验证 | 结果 / 边界 | 证据 |
| --- | --- | --- |
| NaCGEC 真实输入 | 500 Dev、5869 Test；Test 导出无评分；显式 Test gold 全量逐条匹配；误传 Dev gold/旧隐式输入被拒绝 | data_eval.log、plumbing.log |
| FCGEC 参考还原 | 本地全部 valid 还原、官方操作示例与多参考分支通过 | data_eval.log |
| 两种实际评分器 | 小样例运行真实 ChERRANT、THULAC/NUS MaxMatch；修复前异常和修复后完美参考 100 已复现 | score-real.log、score-real-fixed.log |
| MaxMatch 回归 | 多位置插入/删除/替换、空目标、无编辑，完美参考均为 100 | 本轮临时执行 |
| 指令数据准备 | 受控 COIG/Tulu 样例：合法多轮、清洗/去重/泄漏剔除、40:60 小池、Train/Dev 无相同首问；Parquet 和许可/ID 保留通过 | data_eval.log、plumbing.log |
| 普通 Qwen3 完整生命周期 | 受控小模型真实 ms-swift 两步 LoRA → 候选完整验收 → 合并 → 重载复验 → 完整模型指针；最新输入视图流程通过 | fit-native-final.log、native-ins/v1/ |
| 上下文 PGCA 生命周期 | 受控小模型同样完整跑通，主干/音形 Cache、LoRA 生效和合并后自包含重载通过；两步实际监督量记录为 4 | fit-pgca.log、pgca-ins/v0/ |
| 选模失败路径 | 替身验收：低 loss 生成失败选后续合格候选；全失败不发布新指针。属于控制逻辑测试 | negative_selection.py |
| harness 实际执行 | 真实 Qwen Char Tokenizer + 受控小模型：2-shot 多轮选择题、生成题、Prompt IDs 和 EOS 停止通过；不是正式 benchmark 分数 | harness-real.log、harness-real/chat_prompts.jsonl |
| 六项任务配置 | MMLU/ARC/SQuAD/GSM8K/CMMLU/CLUE 配置生成通过，未下载/运行全量题目 | bundles.log |
| 脚本接续/预算 | 三路线完成指针跳过、任务预算分组和顺序、插件迁移路径、预算完整对话计数通过 | plumbing.log、budget.log |
| 输入与模式接口 | 超长完整对话过滤及混合元数据 Arrow JSON 加载通过；Base/Instruct 参数与单阶段自定义 OUT 通过替身控制测试 | final-views.log、entry-modes.log |
| 静态检查 | 新模块 Ruff F/E9、相关 Python AST、shell bash -n、git diff --check | 本轮工具执行 |

受控小模型的输出模式预先构造用于验证生命周期，不证明真实训练学会指令或 EOS。最初用 WordLevel 玩具 Tokenizer 做 harness 选择题时无法保留 ChatML 尾部换行，导致空 continuation；改用实际 Qwen Tokenizer 后通过，未为适配测试而修改正式协议。

## 尚未验收

- 真实 COIG/Tulu 全量数据清洗/混合/去污染、来源许可与质量抽检；确切比例、可用量和三路实际监督预算。
- 真实规模 General-SFT 效果、所有专项训练效果、GPU BF16/FP16、多卡监督计数及 vLLM 实际路径。
- 六套全量通用 benchmark 的实际 Prompt/停止行为和分数；运行时检查已接入。
- C³Bench 与官方评分的等价性；NaCGEC 分词/完整评测流水线与排行榜严格等价性。

数据就绪后按 `scripts/train/instruction/README.md` 准备并核对报告，再启动正式训练；现阶段已有 CPT 步骤可单独继续，不依赖尚未下载的指令数据。
