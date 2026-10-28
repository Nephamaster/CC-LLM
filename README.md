# CC-LLM

CC-LLM 在 Qwen3 Base 模型上构建单汉字语义词表，并加入音形特征、因果上下文编码器和 PGCA 分支。仓库包含模型构建、继续预训练、下游 LoRA 微调与完整评估代码，不包含数据集、模型权重和运行产物。

代码面向 Qwen3 Base 架构，不把参数规模写死在运行命令中；隐藏维度、注意力头数和 PGCA 层位置从所选模型配置推导。其他模型架构需要单独适配。不同规模的显存和计算需求不同，正式训练前请按机器资源调整各阶段参数。

## 目录

```text
src/                         模型、词表、embedding 与 PGCA 实现
scripts/environment/         train/evaluation 两套环境
scripts/model_build/         Char 与 PGCA 模型构建
scripts/train/               Phase1、Phase2 和下游 LoRA
scripts/evaluation/          PPL、通用 benchmark 与下游评估
requirements/                两套环境的普通 pip 依赖
data/                        用户提供的数据，不提交 Git
models/                      Base/Char/PGCA 模型，不提交 Git
outputs/                     训练缓存、checkpoint 和评估结果，不提交 Git
```

## 数据目录

将准备好的数据目录直接放到仓库根目录并命名为 `data`。运行代码不需要数据 manifest。

```text
data/
├── model_build/
│   ├── hanzi/
│   │   ├── tghz2013.txt
│   │   ├── common_traditional.txt
│   │   ├── structure.tsv
│   │   └── rare_high_freq.txt          # 可选
│   └── unihan/
│       ├── Unihan.zip                  # 可选
│       └── Unihan_*.txt
├── pretrain/
│   ├── phase1/
│   │   ├── train/*.tar.xz              # 每个压缩包内一个同名 JSONL
│   │   └── validation/*.jsonl
│   └── phase2/
│       ├── train/*.tar.xz
│       └── validation/*.jsonl
├── finetune/
│   ├── csc/{csc_mix_dedup,cscd_dev}.jsonl
│   ├── cgec/{train,validation}.jsonl
│   └── ancient/
│       ├── train.jsonl
│       ├── train.metadata.jsonl
│       ├── dev.jsonl
│       └── dev.metadata.jsonl
└── evaluation/
    ├── ppl/*.jsonl
    ├── csc/*.jsonl
    ├── cgec/{FCGEC,NaCGEC}/...
    └── ancient/{c3bench,FuxiBench}/...
```

### 训练前必须解压预训练分片

数据包保留 `tar.xz`。首次训练前必须在仓库根目录执行以下命令，将 JSONL 原子解压到压缩包所在目录；原压缩包不会删除：

```bash
set -euo pipefail
find data/pretrain -type f -path '*/train/*.tar.xz' -print0 |
while IFS= read -r -d '' archive; do
  jsonl="${archive%.tar.xz}.jsonl"
  [[ -s "$jsonl" ]] && continue
  temporary="${jsonl}.tmp"
  tar -xJOf "$archive" "$(basename "$jsonl")" > "$temporary"
  mv -- "$temporary" "$jsonl"
done
```

Phase1/2 脚本只读取同目录下的 `*.jsonl`；如果只检测到压缩包，会停止并提示先执行上述命令。

MMLU、ARC-Challenge、GSM8K、SQuAD v2、CMMLU 和 CLUE 由 lm-evaluation-harness 通过 `HF_ENDPOINT=https://hf-mirror.com` 获取并复用 Hugging Face 缓存，因此不放入数据包，也不固定数据 revision。

Base 模型必须是本地完整 Hugging Face 模型（包含配置、tokenizer 和权重）。默认放在 `models/base/`，也可以在构建时通过 `BASE_MODEL` 指定其他位置。

## 四步运行

### 1. 环境

先安装 Python 3.12。环境脚本默认使用 `python3.12` 创建虚拟环境；若解释器不在 PATH，可设置 `PYTHON_BIN=/path/to/python3.12`。已有的 `.venv/train` 和 `.venv/evaluation` 也必须使用 Python 3.12；其他版本的虚拟环境需先移到其他目录，再重新创建。

```bash
bash scripts/environment/run.sh
```

该命令创建：

```text
.venv/train
.venv/evaluation
```

普通 pip 包默认使用清华镜像，PyTorch 默认使用国内 CUDA wheel 镜像，Hugging Face 端点统一设置为：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

如果目标机器连国内 pip 镜像也无法访问，先在可联网机器准备 wheel 目录，然后执行：

```bash
WHEELHOUSE=/path/to/wheels bash scripts/environment/run.sh
```

此时 pip 使用 `--no-index`，不会访问网络。

### 2. 模型构建

模型构建依次执行词表与特征索引、embedding 迁移、PGCA 构建和模型校验。以一个本地 Qwen3 Base 模型为例，将以下路径和名称替换成自己的实际值：

```bash
BASE_MODEL=/path/to/your/Qwen3-Base \
BASE_MODEL_NAME=Qwen3-Base \
bash scripts/model_build/run.sh
```

`BASE_MODEL` 是**路径**，`BASE_MODEL_NAME` 是用于生成目录名的**名称**。此例默认生成 `models/Qwen3-Base-Char/` 和 `models/Qwen3-Base-PGCA/`。省略 `BASE_MODEL_NAME` 时使用 Base 路径的最后一级目录名；省略 `BASE_MODEL` 时使用 `models/base/`。

构建完成后，`models/next_stage_model.txt` 保存 PGCA 模型路径，`models/model_name.txt` 保存 PGCA 模型名称，供训练入口默认读取。每次构建成功都会更新这两个全局指针；在同一仓库实验多个模型时，建议在训练命令中明确指定模型，避免使用最后一次构建的其他模型。

### 3. 训练

训练入口 [scripts/train/run.sh](scripts/train/run.sh) 默认依次执行：

```text
Phase1 → Phase2-A → Phase2-B → CSC LoRA → CGEC LoRA → Ancient LoRA
```

指定上述示例中构建的 PGCA 模型，并从 Phase1 开始运行完整流程：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_NAME=Qwen3-Base-PGCA \
PHASE1_MODEL="$PWD/models/Qwen3-Base-PGCA" \
bash scripts/train/run.sh
```

`MODEL_NAME` 是 PGCA **模型名称**，不带 `-ALIGN` 或 `-CPT`，决定输出、缓存目录及各阶段指针的位置；`PHASE1_MODEL` 是 Phase1 的**输入模型路径**。总入口不会把 `MODEL=...` 当成 Phase1 输入，应使用 `PHASE1_MODEL`。

如果不指定这两个变量，总入口使用 `models/model_name.txt`，Phase1 使用 `models/next_stage_model.txt`。**只指定 `MODEL_NAME` 不会改变 Phase1 的默认输入路径**；切换模型时应同时指定 `PHASE1_MODEL`。

只运行继续预训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_NAME=Qwen3-Base-PGCA \
PHASE1_MODEL="$PWD/models/Qwen3-Base-PGCA" \
STAGES=phase1,phase2_a,phase2_b \
bash scripts/train/run.sh
```

只运行下游微调（默认读取该模型已完成的 CPT 指针）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_NAME=Qwen3-Base-PGCA \
STAGES=csc,cgec,ancient \
bash scripts/train/run.sh
```

每个阶段默认使用前一阶段发布的 `next_stage_model.txt`。若需要指定某个已有 checkpoint，可按阶段传入对应变量：

| 阶段 | 输入路径变量 | 默认输入 |
| --- | --- | --- |
| Phase1 | `PHASE1_MODEL` | `models/next_stage_model.txt` 指向的模型 |
| Phase2-A | `PHASE2_A_MODEL` | `outputs/train/M-ALIGN/next_stage_model.txt` |
| Phase2-B | `PHASE2_B_MODEL` | `outputs/train/M-CPT/adaptation/next_stage_model.txt` |
| CSC | `CSC_MODEL` | `outputs/train/M-CPT/next_stage_model.txt` |
| CGEC | `CGEC_MODEL` | `outputs/train/M-CPT/next_stage_model.txt` |
| 古汉语 | `ANCIENT_MODEL` | `outputs/train/M-CPT/next_stage_model.txt` |

表中 `M` 表示 `MODEL_NAME`。例如，从指定 Phase1 checkpoint 开始运行 Phase2-A/B：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_NAME=Qwen3-Base-PGCA \
PHASE2_A_MODEL=/path/to/phase1/checkpoint-N \
STAGES=phase2_a,phase2_b \
bash scripts/train/run.sh
```

指定 CPT checkpoint 单独进行 CSC 微调：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_NAME=Qwen3-Base-PGCA \
CSC_MODEL=/path/to/cpt/checkpoint-N \
STAGES=csc \
bash scripts/train/run.sh
```

阶段产物写入 `outputs/train/M-ALIGN/`、`outputs/train/M-CPT/{adaptation,joint}/` 和 `outputs/train/M-CPT-{CSC,CGEC,ACT}/`；训练缓存默认位于 `outputs/cache/M/`。多个模型请使用各自的 `MODEL_NAME`，不要共用缓存。

总入口只检查对应阶段的完成指针是否存在：存在时默认跳过该阶段，即使传入了新的模型路径也不会重新训练。`FORCE=true` 会重新执行本次 `STAGES` 选择的所有阶段；它不会自动清除已有训练缓存。断点恢复应调用相应的单阶段脚本并设置 `RESUME_FROM_CHECKPOINT`，例如：

```bash
MODEL_NAME=Qwen3-Base-PGCA \
MODEL=/path/to/phase2-b/checkpoint-N \
RESUME_FROM_CHECKPOINT=/path/to/phase2-b/checkpoint-N \
bash scripts/train/cpt/phase2_b.sh
```

恢复时需沿用原阶段的 `OUT`、缓存、数据及训练参数；如果显式设置 `MODEL`，它必须与断点对应。不要在总入口同时对多个阶段传同一个 `RESUME_FROM_CHECKPOINT`。

每个阶段的学习率、batch、长度、保存间隔等参数位于对应脚本顶部：

```text
scripts/train/cpt/phase1.sh
scripts/train/cpt/phase2_a.sh
scripts/train/cpt/phase2_b.sh
scripts/train/finetune/common.sh
scripts/train/finetune/{csc,cgec,ancient}.sh
```

`CUDA_VISIBLE_DEVICES` 选择 GPU；脚本默认按可见 GPU 数推导进程数。训练的有效 batch size = GPU 数 × 每卡 batch size × 梯度累积步数。显存不足时先调整相应阶段的 batch、长度及卡数，再进行小规模冒烟。

### 4. 评估

在当前模型训练完成后，评估入口默认读取模型构建名称与 CPT 完成指针。多模型实验建议明确指定模型名称：

```bash
CUDA_VISIBLE_DEVICES=0 \
PGCA_MODEL_NAME=Qwen3-Base-PGCA \
bash scripts/evaluation/run.sh
```

指定已有 CPT checkpoint 时，传入完整模型路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
PGCA_MODEL_NAME=Qwen3-Base-PGCA \
MODEL=/path/to/cpt/checkpoint-N \
bash scripts/evaluation/run.sh
```

下游评估默认读取对应任务的 LoRA 完成指针。评估具体 adapter 时，可直接使用 [下游评估入口](scripts/evaluation/README.md)。

要评估 Base、Char 或 ALIGN 等对照模型，可以直接调用单项入口，并为每个模型指定真实路径和名称：

```bash
# Base 模型的 PPL；Char 或 ALIGN 模型替换 MODEL 和 MODEL_NAME 即可。
CUDA_VISIBLE_DEVICES=0 \
MODEL=/path/to/your/Qwen3-Base \
MODEL_NAME=Qwen3-Base \
bash scripts/evaluation/ppl/run.sh

# 指定模型的通用 benchmark；BENCHMARK 可设置为 all、mmlu、arc 等。
CUDA_VISIBLE_DEVICES=0 \
MODEL=/path/to/your/model-or-checkpoint \
MODEL_NAME=Qwen3-Base-Char \
BENCHMARK=mmlu \
bash scripts/evaluation/lmeh/run.sh
```

默认运行 PPL、六类通用 benchmark、CSC、CGEC、C³Bench 和 Fùxì。也可以选择：

```bash
BENCHMARKS=ppl,lmeh bash scripts/evaluation/run.sh
BENCHMARKS=csc,fcgec_valid,nacgec,ancient_c3 bash scripts/evaluation/run.sh
```

所有评估共用 `.venv/evaluation`。标准 Qwen 模型在 `INFERENCE_BACKEND=auto` 时使用 vLLM；当前 PGCA 因音形上下文使用独立因果缓存，自动选择 HF 后端以保证结果正确。可显式设置 `INFERENCE_BACKEND=vllm` 试用 vLLM Transformers backend，但在对应模型通过实际一致性验收前，不应将其结果与 HF 正式结果混用。

```bash
INFERENCE_BACKEND=vllm TENSOR_PARALLEL_SIZE=2 \
bash scripts/evaluation/run.sh
```

评估结果自动写入 `outputs/evaluation/<task>/<model>/.../vN/`，重复运行自动创建新版本。FCGEC test 没有公开标签，只生成提交文件；Fùxì 未配置裁判接口时，模型裁判项目保持 pending。

## 产物命名

设 Base 名称为 `B`：

```text
B-Char                 单汉字模型
B-PGCA                 完整 PGCA 模型
B-PGCA-ALIGN           Phase1
B-PGCA-CPT             Phase2
B-PGCA-CPT-CSC         CSC LoRA
B-PGCA-CPT-CGEC        CGEC LoRA
B-PGCA-CPT-ACT         古汉语 LoRA
```

模型规模只体现在用户选择的 Base 名称与权重中，不写死在源码、脚本或实验文档里。
