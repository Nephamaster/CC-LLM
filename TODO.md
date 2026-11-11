## 一、训练流程与数据准备

1\. 【P0】新增通用指令微调 General-SFT 阶段

- 在 `scripts/train/instruction` 中新增独立训练入口。
- Base、Char-CPT、PGCA-CPT 分别进行 General-SFT。
- 三条路线使用相同的训练数据、监督 Token 预算及评估协议。
- 训练好模型名称后缀加INS

2\. 【P0】构建 COIG-CQIA + Tulu 3 通用 SFT 数据集

- 统一转换为 `messages` 格式，支持合法多轮对话。
- 进行数据清洗、去重、质量过滤及 Benchmark 去污染。
- 保留来源、原始 ID、任务类型和许可证信息。
- 控制中英文、数学、代码等类型的比例，避免 Tulu 3 主导数据分布。
- 中文和其他比例预期为 40:60，不是必须固定的最终比例。

3\. 【P0】统一 Chat Template 与特殊 Token 训练协议

- General-SFT 使用 Qwen3 ChatML，统一采用非思考模式。
- 仅对 Assistant 回答及其结束符计算监督损失。
- 确认 `<|im_end|>` 有效参与训练。
- 校验训练、推理生成的 Prompt Token IDs 完全一致。
- 正确配置 `generation_config.json` 中的 EOS/停止条件。

4\. 【P0】建立 General-SFT 的生成能力验收

- 检查 EOS 生成率、正常停止率、截断率、重复率。
- 检查训练 Loss 与自由生成表现是否一致。
- 不再仅根据 Dev Loss 选取最终 checkpoint。
- 避免此前 ACT 出现的“Loss 正常但无法终止生成”。

5\. 【P0】调整下游专项 SFT 的模型来源

- CSC、CGEC、Ancient 的正式新增路线改为从 General-SFT 模型出发。
- 三条路线分别训练独立的专项 Adapter。
- 若 General-SFT 使用 LoRA，需要明确合并或多 Adapter 加载方案，确保下游训练与评估加载的是正确权重。

## 二、通用能力评测

6\. 【P0】新增 Base / Instruct 两套通用评测协议

- 修改 `scripts/evaluation/lmeh/`，增加显式评测模式。
- Base/CPT：`apply_chat_template=False`。
- General-SFT：`apply_chat_template=True`。
- Instruct 模式须使用与训练一致的非思考 Chat Template。
- Few-shot 的消息组织方式固定并记录，不能在模型间随意切换。

7\. 【P0】验证通用 Benchmark 的实际 Prompt 与停止行为

- 所有通用Benchmark采用训练好的Chat模型进行评估
- 确认模板、标记等都正确。

## 三、下游任务评测与代码修复


8\. 【P0】修复 NaCGEC 的 Dev/Test 数据协议

- Dev：使用公开的 `nacgec.dev.ref.para` 计算本地指标。
- Test：使用公开的 `nacgec.test.input` 生成提交文件。
- 不再默认读取 `nacgec.test.ref.para`，需要本地 Test 评分时显式指定。
- 核对修正：官方已于 2023-08-22 公开 Test gold，本地也有该文件；不能把已有 Test 结果仅因这一前提误判为无效。
- 无公开参考答案时不得生成本地 Test 分数。

9\. 【P1】规范 FCGEC、NaCGEC 的评分口径

- FCGEC 主结果采用官方 F0.5，F1 保留为补充指标。
- NaCGEC 采用 ChERRANT 与 MaxMatch F0.5 的规定汇总方式。
- 验证 FCGEC 多参考答案及操作标签还原的正确性。

10\. 【P1】核对 C³Bench 评分器与官方协议的一致性

- 重点检查 NER、出处检索、标点评价方式。
- 当前重建评分器可继续用于内部对照。
- 未确认等价前，不将结果直接标记为官方 C³Bench 分数。

11\. 【P1】调整专项生成评估和可靠性统计

- 古汉语分类、NER、出处检索、翻译与标点分别设置合理的生成预算。
- CSC 保留原始输出，不仅使用第一行清理后的答案评价模型。
- CSC、CGEC、Ancient 均记录正常停止率、长度截断率和重复率。
- 专项评估所用模板要能适配。

## 实施优先级

建议按以下顺序推进，而不是同时修改所有模块：

1. 先修复现有 P0 评测缺陷

   任务 8。先确保不会复用错误结果，也不会使用错误的数据协议。

2. 建设 指令 SFT 完整链路

   任务 1–5。包括数据、训练入口、模板、EOS 验收和下游模型接续。

3. 改造通用评测协议

   任务 6、7。分别 Instruct 模型评测。

其中 P0 属于开展下一轮可靠实验的必要前置项；P1 属于正式论文结果定稿前必须完成的评测规范化与可比性工作。

COIG/Tulu 的具体混合比例、是否增加辅助任务权重、是否扩展 PGCA 结构等仍属于待验证的研究选择，不纳入这次已经确定的必要工程工作。

## 2026-11-11 实施进度

| 任务 | 当前状态 |
| --- | --- |
| 1 General-SFT | 已新增独立入口，总入口已接入三条路线；LoRA → 完整模型合并。 |
| 2 指令数据 | 本地 JSON/JSONL/Parquet 转换、清洗、去重、来源/ID/许可、多轮、混合与基准输入去污染代码已具备；真实 COIG/Tulu 尚未就绪，未执行正式数据构建。 |
| 3 模板/EOS | 复用 Qwen3 非思考 Swift 模板；逐条检查各轮 Assistant EOS 和最后一轮 Prompt IDs；合并模型配置停止 Token。 |
| 4 生成验收 | 保存时轻量检查；最终按有限 Dev loss 排序并要求完整生成验收，合并后复验，无合格者不发布。 |
| 5 专项接续 | 新主线默认从 M-INS 完整权重开始，三项各训独立 LoRA；显式旧模型输入仍可用于单独对照。 |
| 6 Base/Instruct 协议 | 已显式区分；默认 Instruct，非思考模板，few-shot 固定为多轮，报告记录模式。 |
| 7 实际 Prompt/停止 | 每个 Instruct Prompt 核对 Token IDs，保留 Prompt 样例和停止配置；真实 Qwen Tokenizer + 小模型已跑通 harness 选择题和生成题。六套完整正式基准及 GPU/vLLM 尚未验收。 |
| 8 NaCGEC | 默认 Dev 评分/Test 提交导出；显式公开 Test gold 要与官方输入逐条匹配，旧隐式 gold 输入在推理/评分前拒绝。 |
| 9 CGEC 评分 | 主指标改为 F0.5；NaCGEC 双评分器平均。另修复 MaxMatch 金标编辑逆序造成漏计；官方完整排行榜等价性仍未证明。 |
| 10 C³Bench | 保持内部重建并明确非官方等价标记；官方仓库当前只提供数据/简短说明，NER/出处/标点等价性仍待证据。 |
| 11 专项生成 | 已按任务配置预算；CSC 主报完整原始输出，清理结果为补充；三项均报告停止/截断/重复率。 |

数据就绪后仍需完成：正式混合比例与质量抽检、全量去污染、三路共同可用样本和实际监督 token 对齐、真实规模 General-SFT/专项训练，以及 GPU/vLLM/全量 benchmark 验收。`SUPERVISED_TOKENS=0` 当前表示全量一遍，不代表已经确定正式预算；完整对话选择可能略超目标，报告保留超额及真实训练计数。

代码、数据目录和命令见 [General-SFT](scripts/train/instruction/README.md)。本轮核对与验证边界见 [实施记录](Checkpoint/2026-11-11-general-sft-and-evaluation.md)。
