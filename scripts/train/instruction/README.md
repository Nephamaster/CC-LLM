# 通用指令微调（General-SFT）

主线是 `Base → INS`、`Char-CPT → INS`、`PGCA-CPT → INS`，随后各自独立训练 CSC、CGEC、ACT Adapter。General-SFT 使用 all-linear LoRA，PGCA 分支也参与，辅助损失关闭。验收后合并为完整 `M-INS`；专项训练只加载这个完整模型，不叠加 General-SFT Adapter。

## 数据准备好后

本轮不下载数据。将两份原始数据的 JSON、JSONL 或 Parquet 放到：

```text
data/instruction/
├── raw/
│   ├── COIG-CQIA/
│   └── tulu-3-sft-mixture/
└── prepared/                   # 准备命令自动生成 vN、latest.txt
```

已有成品 `train.jsonl`、`dev.jsonl` 时也可以通过 `INSTRUCTION_DATA` 直接指定其目录；集中冒烟还接受 `data/instruction/final/`。

新环境按根 README 安装。已有训练环境只需补新增依赖：

```bash
.venv/train/bin/python -m pip install pyahocorasick -i https://pypi.tuna.tsinghua.edu.cn/simple
```

两套环境和本地评估数据准备好后执行：

```bash
HF_ENDPOINT=https://hf-mirror.com \
.venv/train/bin/python -B -m scripts.train.instruction.prepare
```

该命令通过 evaluation 环境读取本地专项评估输入、通过 HF 镜像读取 LMEH 的评估输入，再由 train 环境清洗训练数据。不会调用 Git，不固定依赖或数据 revision，也不会训练模型。`--coig`、`--tulu`、`--benchmarks`、`--evaluation-python` 可以覆盖位置。

准备过程保留合法多轮，统一成 `messages`，检查角色交替、空回答、控制字符、模板标记及过长文本；按首轮 user 问题去重，避免同问题不同答案跨 Train/Dev。保留来源、原 ID、原任务类型、版权和许可证描述。COIG 数据卡的许可证未明确，不伪造授权；Tulu 的集合许可与子集许可不能混为一谈。

默认中文会话目标占比 40%、Tulu 会话占比上限 60%，按可用数据下采样、不重复扩增。中文用汉字占字母比例估计，任务分类也包含来源/文本启发式，报告会记录；这些不是精确语言识别标签。`--chinese-share`、`--max-tulu-share` 可调整，数学/代码份额需要同时显式设置 `--math-share` 和 `--code-share`，不在代码中规定最终研究配比。样本不足时报告实际比例及未满足的数量，不能把配置比例当作实际比例。

去污染检测覆盖**规范化后的完整基准输入子串**，最短 24 字符，不等价于完整 n-gram 或改写检测。若明确使用自备的完整评估输入清单，可以重复传 `--benchmark-text`（每行一个输入），此时替换默认基准收集；不要只放一两个示例却宣称已覆盖全量基准。成品报告在 `prepared/vN/preparation_report.json`，数据尚未就绪时不宣称正式清洗/去污染已完成。

数据格式和许可来源：[COIG-CQIA 数据卡](https://huggingface.co/datasets/m-a-p/COIG-CQIA)、[Tulu 3 数据卡](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)。

## 独立训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL=/absolute/path/to/full-model MODEL_NAME=Qwen3-Base-PGCA-CPT \
INSTRUCTION_DATA=/absolute/path/to/prepared-data \
TRAIN_BATCH_SIZE=2 GRAD_ACCUMULATION=8 \
bash scripts/train/instruction/run.sh
```

Base 用原基座路径/名称；Char 用 Char-CPT；PGCA 用 PGCA-CPT。参数在 `run.sh` 顶部，同名环境变量优先；LoRA/优化器等共用 `finetune/common.sh`。默认长度 4096、学习率 1e-4、单遍训练，均为可调整的起始配置。

`SUPERVISED_TOKENS=0` 表示全量一遍，**正式统一预算尚未确定**。设置正数后，按实际模板 `labels[1:] != -100` 选择完整对话，达到目标后保留最后一条完整对话，epoch 固定为 1。不会截断答案或屏蔽 EOS 来凑整数，因此实际值可能超过目标，超过量、超长排除数均写进 `inputs/vN/train.report.json`。训练日志另记录跨卡累计的 `train_supervised_tokens`，提前停止、重复 epoch 或分布式补齐也不能用准备量冒充实际训练量。

三条路线共用同一清洗池、Dev 和目标预算；Tokenizer 及长度过滤会影响实际保留量。数据就绪后必须比较三份报告，确定共同可用样本和正式预算。当前不声称已经实现严格相等的实际监督 token，也不为了凑数引入部分答案监督。

## 模板、选模和产物

`cc_llm_instruction` 使用 Qwen3 非思考 ChatML，仅监督 Assistant 回答及每轮 `<|im_end|>`。每条保留的 Train/Dev 对话会核对 EOS label 和最后一轮训练/推理 Prompt IDs；过长对话完整排除并计数。

每次保存继续执行轻量固定 Dev 检查。训练结束后在保留的 checkpoint 中按有限 Dev loss 排序，逐个重新加载 LoRA 做完整生成、EOS、Cache 和 Adapter 生效验收，选择第一个通过者。没有通过者则不发布新 INS 模型。合并后再检查一次，防止只验证未合并 Adapter。默认门槛和 `DEV_CHECK_*` 配置见 [固定 Dev 验收](../finetune/README.md#固定-dev-生成验收)。

产物：

```text
outputs/train/M-INS/
├── inputs/vN/                  # 本轮完整对话视图及真实标签计数
├── adapters/vN-日期/           # LoRA checkpoint、每次保存检查
├── vN/
│   ├── instruction_selection.json
│   ├── candidate_checks/
│   ├── merged_check/
│   └── merged/                 # 完整模型、Tokenizer、EOS 配置及模板参数
└── next_stage_model.txt         # 仅在合并和验收成功后指向 merged/
```

合并的 `generation_config.json` 配置 im_end/endoftext 停止，不继承 checkpoint 的全局 `max_new_tokens`。`DEV_CHECK_MODE=report` 仅供隔离冒烟/排障，报告明确标注 `smoke_only`，不能当作正式生成验收通过。

## 接续专项训练与评测

总训练入口默认已经包含 instruction：

```bash
MODEL_VARIANT=pgca MODEL_NAME=Qwen3-Base-PGCA \
STAGES=instruction,csc,cgec,ancient bash scripts/train/run.sh
```

默认从该模型的 CPT 指针开始；专项 Adapter 分别写入 `M-INS-{CSC,CGEC,ACT}`。只运行专项脚本时使用 `MODEL_NAME=M-INS`，或显式指定 `MODEL=合并后的完整模型`。

通用正式结果使用 INS 模型：

```bash
MODEL_NAME=Qwen3-Base-PGCA-CPT-INS \
EVAL_MODE=instruct bash scripts/evaluation/run.sh
```

原始 Base/CPT 的辅助对照通过 LMEH 的 `EVAL_MODE=base` 单独运行；不要与 Instruct 协议混填同一组结果。
