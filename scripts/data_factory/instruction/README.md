# 指令数据准备（main）

主线是 `Base → INS`、`Char-CPT → INS`、`PGCA-CPT → INS`，随后各自独立训练 CSC、CGEC、ACT Adapter。General-SFT 使用 all-linear LoRA，PGCA 分支也参与，辅助损失关闭。验收后合并为完整 `M-INS`；专项训练只加载这个完整模型，不叠加 General-SFT Adapter。

## 首轮正式规模

| 项目 | 首轮设置 |
| --- | --- |
| 清洗后高质量样本池 | 目标约 150,000 条，含 Train/Dev |
| 默认划分 | 约 147,000 条训练、3,000 条验证（2%） |
| 每 Epoch 完整输入 | 预计 0.12–0.23B Token，含 Prompt、回答和模板标记，不含 padding |
| 训练 Epoch | 默认 1，可显式设置 2；两遍预计 0.24–0.46B 输入 Token |
| Assistant 监督量 | 独立按有效 labels 统计，不将输入 Token 估计当作监督预算 |

以上 Token 量是规划范围，实际以对应 Tokenizer 的报告为准，不作为运行硬门槛。清洗、去污染或混合约束导致不足 15 万时，不重复补样；报告保留目标、实际 Train/Dev 数量及差额。

## 原始数据与准备

本轮不下载数据。直接读取以下两个 JSONL 文件，不扫描子目录：

```text
data/instruction/
├── raw/
│   ├── COIG-CQIA.jsonl
│   └── tulu-3-sft-mixture.jsonl
└── prepared/                   # 准备命令自动生成 vN、latest.txt
```

已有成品 `train.jsonl`、`dev.jsonl` 时也可以通过 `INSTRUCTION_DATA` 直接指定其目录；集中冒烟还接受 `data/instruction/final/`。

当前两份 JSONL 均含 `sample_id`、`source`、`messages` 和 `metadata`。转换直接使用原始 `messages`，从嵌套 `metadata` 读取原始 ID、任务类型、人工校验、版权及许可字段；同时将完整元数据保存在标量字符串 `source_metadata` 中，避免不同来源的嵌套结构影响 Arrow 读取。顶层同名字段优先，仍支持原生 COIG 的 `instruction/input/output` 格式。`dedup_text`、`first_user_prompt` 不作为训练内容，清洗和去重均从真实对话重新计算。

当前本地原始文件共 COIG 44,693 条、Tulu 931,481 条（2026-11-12 全量解析计数）。保持默认 Tulu 占比上限 60% 时，即使 COIG 全部保留，总池也最多约 111,732 条；清洗、去重、去污染和语言比例约束后还可能更少。目标 150,000 不代表当前数据一定能达到，配比保持不变，缺口由准备报告记录。

新环境按根 README 安装。已有训练环境只需补新增依赖：

```bash
.venv/train/bin/python -m pip install pyahocorasick -i https://pypi.tuna.tsinghua.edu.cn/simple
```

两套环境和本地评估数据准备好后执行：

```bash
HF_ENDPOINT=https://hf-mirror.com \
.venv/train/bin/python -B -m scripts.data_factory.instruction.prepare --samples 150000
```

该命令通过 evaluation 环境读取本地专项评估输入、通过 HF 镜像读取 LMEH 的评估输入，再由 train 环境清洗训练数据。不会调用 Git，不固定依赖或数据 revision，也不会训练模型。`--coig`、`--tulu` 可以覆盖两个 JSONL 文件的位置，`--benchmarks`、`--evaluation-python` 可以覆盖评估数据和解释器位置。

准备过程保留合法多轮，统一成 `messages`，检查角色交替、空回答、控制字符、模板标记及过长文本；按首轮 user 问题去重，避免同问题不同答案跨 Train/Dev。保留来源、原 ID、原任务类型、版权和许可证描述。COIG 数据卡的许可证未明确，不伪造授权；Tulu 的集合许可与子集许可不能混为一谈。

默认中文会话目标占比 40%、Tulu 会话占比上限 60%，按可用数据下采样、不重复扩增。中文用汉字占字母比例估计，任务分类也包含来源/文本启发式，报告会记录；这些不是精确语言识别标签。`--chinese-share`、`--max-tulu-share` 可调整，数学/代码份额需要同时显式设置 `--math-share` 和 `--code-share`，不在代码中规定最终研究配比。样本不足时报告实际比例及未满足的数量，不能把配置比例当作实际比例。

`--samples` 是包含验证集的总样本池上限，默认 150000；设为 0 则保留混合约束下全部可用数据。`sample_shortfall` 显示距目标的差额，`train_rows` 和 `dev_rows` 显示实际划分。

去污染检测覆盖**规范化后的完整基准输入子串**，最短 24 字符，不等价于完整 n-gram 或改写检测。若明确使用自备的完整评估输入清单，可以重复传 `--benchmark-text`（每行一个输入），此时替换默认基准收集；不要只放一两个示例却宣称已覆盖全量基准。成品报告在 `prepared/vN/preparation_report.json`，数据尚未就绪时不宣称正式清洗/去污染已完成。

数据格式和许可来源：[COIG-CQIA 数据卡](https://huggingface.co/datasets/m-a-p/COIG-CQIA)、[Tulu 3 数据卡](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)。

## 发布成品

本目录仅在 main 分支维护。清洗与去污染完成后，将选中的 prepared/vN/train.jsonl、dev.jsonl 放入 data/instruction/，随其他最终数据一起打包。master 只读取这两个成品文件，不需要 raw、prepared 或 latest 指针。

正式训练入口、训练前的模型编码与监督 Token 统计、验收及合并见 [General-SFT](../../train/instruction/README.md)。
