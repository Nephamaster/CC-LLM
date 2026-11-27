# 通用指令微调（General-SFT）

主线是 Base → INS、Char-Align-CPT → INS、Char-PGCA-Align-CPT → INS，随后各自独立训练 CSC、CGEC、ACT Adapter。使用 all-linear LoRA，PGCA 分支也参与，辅助损失关闭；验收后合并为完整 M-INS，再进行专项 LoRA。

## 成品数据

数据包解压到项目根目录，结构为：

```text
data/instruction/
├── train.jsonl
└── dev.jsonl
```

两个文件已完成原始数据清洗、去重和去污染，每行包含 messages（role/content），支持合法多轮并以 Assistant 回答结束；来源等附加字段可保留。训练和集中冒烟默认直接读取这两个文件，无需原始数据、prepared 目录或 latest.txt。INSTRUCTION_DATA 可指定其他成品目录，TRAIN、DEV 可分别覆盖文件。

原始 COIG-CQIA/Tulu 3 的数据处理代码只在 main 分支 scripts/data_factory/instruction/ 维护；master 保留指令阶段的训练、模型编码、监督 Token 统计、验收与合并。

## 训练前处理

budget.py 使用实际模型 Tokenizer 和训练模板，排除超长完整对话、核对 Assistant/EOS 监督及训练/推理 Prompt IDs，并记录完整输入与有效监督 Token。这是每个模型训练时必要的处理，不会修改成品数据。样本数和 Token 量以数据包及本次报告为准，不假定必然达到原先 15 万条或 0.12–0.23B 的规划目标。

## 独立训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL=/absolute/path/to/full-model MODEL_NAME=Qwen3-Base-Char-PGCA-Align-CPT \
TRAIN_BATCH_SIZE=2 GRAD_ACCUMULATION=8 \
bash scripts/train/instruction/run.sh
```

Base 用原基座路径/名称；Char 用 Char-Align-CPT；默认 PGCA 用 Char-PGCA-Align-CPT。只完成 Align 或 CPT 时，显式传入实际输入名称，例如 `MODEL_NAME=B-Char-Align` 输出 `B-Char-Align-INS`，`MODEL_NAME=B-CPT` 输出 `B-CPT-INS`。参数在 `run.sh` 顶部，同名环境变量优先；LoRA/优化器等共用 `downstream/common.sh`。默认长度 4096、学习率 1e-4、1 Epoch；训练两遍时在同一命令前加 `NUM_TRAIN_EPOCHS=2`。

`SUPERVISED_TOKENS=0` 表示不按监督 Token 裁样本视图，Epoch 独立由 `NUM_TRAIN_EPOCHS` 控制。可选的正数预算按实际模板 `labels[1:] != -100` 选择每 Epoch 的完整对话，达到目标后保留最后一条完整对话，不再强制覆盖 Epoch。不会截断答案或屏蔽 EOS 来凑整数，因此实际值可能超过目标。

`inputs/vN/train.report.json` 分别记录每 Epoch 的 `input_tokens`（完整输入）和 `supervised_tokens`（有效监督），以及超额、超长排除数量；两遍的计划量按这两个计数各乘 2。训练日志中的 `train_supervised_tokens` 另记跨卡、跨 Epoch 的实际监督量，提前停止或分布式补齐也不能用准备量冒充实际训练量。0.12–0.23B 的输入规划不写入 `SUPERVISED_TOKENS`。

三条路线共用同一清洗池、Dev 和目标预算；Tokenizer 及长度过滤会影响实际保留量。数据就绪后必须比较三份报告，确定共同可用样本和正式预算。当前不声称已经实现严格相等的实际监督 token，也不为了凑数引入部分答案监督。

## 模板、选模和产物

`cc_llm_instruction` 使用 Qwen3 非思考 ChatML，仅监督 Assistant 回答及每轮 `<|im_end|>`。每条保留的 Train/Dev 对话会核对 EOS label 和最后一轮训练/推理 Prompt IDs；过长对话完整排除并计数。

每次保存继续执行轻量固定 Dev 检查。训练结束后在保留的 checkpoint 中按有限 Dev loss 排序，逐个重新加载 LoRA 做完整生成、EOS、Cache 和 Adapter 生效验收，选择第一个通过者。没有通过者则不发布新 INS 模型。合并后再检查一次，防止只验证未合并 Adapter。默认门槛和 `DEV_CHECK_*` 配置见 [固定 Dev 验收](../downstream/README.md#固定-dev-生成验收)。

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
MODEL_VARIANT=pgca MODEL_NAME=Qwen3-Base-Char-PGCA \
STAGES=instruction,csc,cgec,ancient bash scripts/train/run.sh
```

默认从该模型的 CPT 指针开始；专项 Adapter 分别写入 `M-INS-{CSC,CGEC,ACT}`。只运行专项脚本时使用 `MODEL_NAME=M-INS`，或显式指定 `MODEL=合并后的完整模型`。

通用正式结果使用 INS 模型：

```bash
MODEL_NAME=Qwen3-Base-Char-PGCA-Align-CPT-INS \
EVAL_MODE=instruct bash scripts/evaluation/run.sh
```

原始 Base/CPT 的辅助对照通过 LMEH 的 `EVAL_MODE=base` 单独运行；不要与 Instruct 协议混填同一组结果。
