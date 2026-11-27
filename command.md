下面沿用你当前的 **Qwen3-1.7B-Base**，完成默认级联 PGCA 主线：

**环境 → 数据 → 模型构建 → 冒烟 → Phase1 → Phase2-A → Phase2-B → INS → 三项下游微调 → 评估。**

Phase2-A/B 都显式关闭辅助任务，通过梯度累积维持各阶段有效 batch。默认模型是 **三个四层上下文编码器串行级联**，构建名称为 `Qwen3-1.7B-Base-Char-PGCA`。

名称按实际经历的构建和训练步骤累计。`Char` 表示字符级改造，`Char-PGCA` 表示在 Char 上加入 PGCA，`PGCA` 表示在原词表模型上加入 PGCA 的命名。本入口默认构建的是 Char-PGCA，未新增原词表 PGCA 构建路线。

```text
Qwen3-1.7B-Base
→ Qwen3-1.7B-Base-Char
→ Qwen3-1.7B-Base-Char-PGCA
→ Qwen3-1.7B-Base-Char-PGCA-Align
→ Qwen3-1.7B-Base-Char-PGCA-Align-CPT
→ Qwen3-1.7B-Base-Char-PGCA-Align-CPT-INS
→ Qwen3-1.7B-Base-Char-PGCA-Align-CPT-INS-{CSC,CGEC,ACT}
```

跳过某阶段时不添加对应后缀，例如 `M-Align-INS`、`M-CPT-INS`。单阶段训练脚本的 `MODEL_NAME` 是其输入模型的规范名称；总训练入口的 `MODEL_NAME` 是该路线的构建模型名。

完整命令也保存在 [八卡执行清单](/home/nepham/Lab/CC-LLM/Checkpoint/2026-11-26-hierarchical-eight-gpu-runbook.md)。以下命令已核对脚本参数和 Shell 语法，尚未在新机器实跑。

**1. 设置仓库、基座和 GPU**

先修改前两条路径。后续保持在项目根目录、同一个终端执行。

```bash
cd /absolute/path/to/CC-LLM

export PROJECT_ROOT="$PWD"
export BASE_MODEL=/absolute/path/to/Qwen3-1.7B-Base
export BASE_MODEL_NAME=Qwen3-1.7B-Base
export PGCA_NAME="${BASE_MODEL_NAME}-Char-PGCA"
export ALIGN_NAME="${PGCA_NAME}-Align"
export CPT_NAME="${ALIGN_NAME}-CPT"
export INS_NAME="${CPT_NAME}-INS"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PYTHONDONTWRITEBYTECODE=1

nvidia-smi
```

需要 HF Token 时执行一次：

```bash
read -rsp "HF Token: " HF_TOKEN
echo
export HF_TOKEN
```

**2. 构建环境**

机器已有 Python 3.12 时：

```bash
export PYTHON_BIN="$(command -v python3.12)"

bash scripts/environment/train.sh
bash scripts/environment/evaluation.sh
```

如果没有 Python 3.12，但已经安装 Conda，先执行：

```bash
conda create -n cc-python312 python=3.12 -y \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

conda activate cc-python312
export PYTHON_BIN="$(command -v python)"
```

然后再执行两个环境脚本。它们会根据新机器驱动选择 Torch/CUDA 渠道。

补上之前 FlashInfer 报错缺少的工具：

```bash
.venv/evaluation/bin/python -m pip install ninja \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

export PATH="$PROJECT_ROOT/.venv/evaluation/bin:$PATH"
```

检查环境：

```bash
.venv/train/bin/python -B -c \
'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'

.venv/evaluation/bin/python -B -c \
'import torch, lm_eval; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

两条应均能看到 8 张 GPU。

**3. 放置数据、解压预训练分片**

还需要完整成品数据包。解压后应包含：

```text
data/
├── model_build/
├── pretrain/phase1/
├── pretrain/phase2/
├── instruction/{train,dev}.jsonl
├── downstream/{csc,cgec,ancient}/
└── evaluation/
```

外层压缩包如果已经包含 `data/`，解压到项目根目录：

```bash
tar -xf /absolute/path/to/dataset.tar.xz -C "$PROJECT_ROOT"
```

如果实际是 `.tar.zst`，替换文件名即可，系统需要有 `zstd` 工具。避免解压成 `data/data/`。

预训练分片原地解压：

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

**4. 构建默认级联模型**

```bash
BASE_MODEL="$BASE_MODEL" \
BASE_MODEL_NAME="$BASE_MODEL_NAME" \
bash scripts/model_build/run.sh
```

生成：

```text
models/Qwen3-1.7B-Base-Char/
models/Qwen3-1.7B-Base-Char-PGCA/
```

无需设置 `CONTEXT_MODE`。当前 1.7B 默认 PGCA 插入索引为 `9–17`，三级编码器的分段起点为 **9、13、17**。

这里有一项现存限制：构建校验仍使用 CPU 上的模型保存精度。如果再次出现此前“仅 Cache 校验失败、BF16 误差 0.5”的情况，需要先做 FP32 复核；这项默认行为尚未修正，不应反复重建权重或直接放宽容差。

**5. 集中冒烟**

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
bash scripts/smoke/run.sh "$BASE_MODEL"
```

查看最新的：

```text
outputs/smoke/vN/summary.md
```

该入口会检查 Base、Char 和默认级联 PGCA 的少步链路。正式训练再使用全部八卡。

**6. Phase1**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL_VARIANT=pgca \
MODEL_NAME="$PGCA_NAME" \
MODEL="$PROJECT_ROOT/models/$PGCA_NAME" \
TRAIN_BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRAD_ACCUMULATION=32 \
DATASET_NUM_PROC=4 \
ENCODING_BATCH_SIZE=32 \
MAX_LENGTH=2048 \
PHASE1_TOKENS=1000000000 \
WARMUP_RATIO=0.03 \
MAX_STEPS=-1 \
RESUME_FROM_CHECKPOINT= \
bash scripts/train/cpt/phase1.sh
```

有效 batch 为 **256**。完成后自动发布 `${ALIGN_NAME}` 的 checkpoint 指针。

**7. Phase1 后的 PPL**

```bash
ALIGN_MODEL="$(cat "$PROJECT_ROOT/outputs/train/$ALIGN_NAME/next_stage_model.txt")"
```

分别测试原 Base、刚构建的 Char、完成 Phase1 的 PGCA：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL="$BASE_MODEL" \
MODEL_NAME="$BASE_MODEL_NAME" \
RUN_NAME=before-cpt \
bash scripts/evaluation/ppl/run.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL="$PROJECT_ROOT/models/${BASE_MODEL_NAME}-Char" \
MODEL_NAME="${BASE_MODEL_NAME}-Char" \
RUN_NAME=before-cpt \
bash scripts/evaluation/ppl/run.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL="$ALIGN_MODEL" \
MODEL_NAME="$ALIGN_NAME" \
RUN_NAME=after-phase1 \
bash scripts/evaluation/ppl/run.sh
```

**8. Phase2-A**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL_NAME="$ALIGN_NAME" \
TRAIN_BATCH_SIZE=2 \
EVAL_BATCH_SIZE=2 \
GRAD_ACCUMULATION=4 \
DATASET_NUM_PROC=32 \
ENCODING_BATCH_SIZE=32 \
MAX_LENGTH=4096 \
PHASE2_TOKENS=10000000000 \
ADAPTATION_TOKENS=100000000 \
PHASE2_A_LR=5e-5 \
WARMUP_RATIO=0.03 \
ENABLE_AUXILIARY_TASKS=false \
MAX_STEPS=-1 \
RESUME_FROM_CHECKPOINT= \
bash scripts/train/cpt/phase2_a.sh
```

有效 batch 为 **64**，自动读取 Phase1 结果。首次运行会先导出 Phase2 编码缓存。

**9. Phase2-B**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL_NAME="$ALIGN_NAME" \
TRAIN_BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRAD_ACCUMULATION=32 \
DATASET_NUM_PROC=32 \
ENCODING_BATCH_SIZE=32 \
MAX_LENGTH=4096 \
PHASE2_TOKENS=10000000000 \
ADAPTATION_TOKENS=100000000 \
PHASE2_B_LR=1e-5 \
PHASE2_B_BRANCH_LR=3e-5 \
WARMUP_RATIO=0.01 \
ENABLE_AUXILIARY_TASKS=false \
MAX_STEPS=-1 \
RESUME_FROM_CHECKPOINT= \
bash scripts/train/cpt/phase2_b.sh
```

有效 batch 为 **256**，自动读取 A 的结果。更新步数由实际数据量计算，保持单遍训练。

**10. 通用指令微调 INS**

读取完成的 CPT 模型：

```bash
CPT_MODEL="$(cat "$PROJECT_ROOT/outputs/train/$CPT_NAME/next_stage_model.txt")"
```

执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL="$CPT_MODEL" \
MODEL_NAME="$CPT_NAME" \
TRAIN_BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRAD_ACCUMULATION=16 \
DATASET_NUM_PROC=4 \
MAX_LENGTH=4096 \
LEARNING_RATE=1e-4 \
NUM_TRAIN_EPOCHS=1 \
WARMUP_RATIO=0.05 \
LORA_RANK=16 \
LORA_ALPHA=32 \
LORA_DROPOUT=0.05 \
MAX_STEPS=-1 \
RESUME_FROM_CHECKPOINT= \
bash scripts/train/instruction/run.sh
```

有效 batch 为 **128**。脚本完成训练、生成验收和模型合并，通过后发布完整 INS 模型。

```bash
INS_MODEL="$(cat "$PROJECT_ROOT/outputs/train/$INS_NAME/next_stage_model.txt")"
```

如果这里没有指针，先检查 INS 的 `candidate_checks`、`merged_check` 报告。

**11. 三项下游 LoRA**

三项分别从同一个 INS 模型开始，顺序运行，每项使用八卡、有效 batch **64**。

CSC：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL="$INS_MODEL" MODEL_NAME="$INS_NAME" \
TRAIN_BATCH_SIZE=1 EVAL_BATCH_SIZE=1 GRAD_ACCUMULATION=8 \
DATASET_NUM_PROC=4 MAX_LENGTH=768 \
LEARNING_RATE=1e-4 NUM_TRAIN_EPOCHS=5 WARMUP_RATIO=0.05 \
MAX_STEPS=-1 RESUME_FROM_CHECKPOINT= \
bash scripts/train/downstream/csc.sh
```

CGEC：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL="$INS_MODEL" MODEL_NAME="$INS_NAME" \
TRAIN_BATCH_SIZE=1 EVAL_BATCH_SIZE=1 GRAD_ACCUMULATION=8 \
DATASET_NUM_PROC=4 MAX_LENGTH=768 \
LEARNING_RATE=1e-4 NUM_TRAIN_EPOCHS=5 WARMUP_RATIO=0.05 \
MAX_STEPS=-1 RESUME_FROM_CHECKPOINT= \
bash scripts/train/downstream/cgec.sh
```

古汉语 ACT：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL="$INS_MODEL" MODEL_NAME="$INS_NAME" \
TRAIN_BATCH_SIZE=1 EVAL_BATCH_SIZE=1 GRAD_ACCUMULATION=8 \
DATASET_NUM_PROC=4 MAX_LENGTH=2048 \
LEARNING_RATE=1e-4 NUM_TRAIN_EPOCHS=5 WARMUP_RATIO=0.05 \
MAX_STEPS=-1 RESUME_FROM_CHECKPOINT= \
bash scripts/train/downstream/ancient.sh
```

下游脚本固定关闭辅助损失，主干和 PGCA 的线性层参与 LoRA。

**12. 六项通用 benchmark**

训练结束、GPU 释放后，六张卡各负责一个任务并行评估。PGCA 当前使用 HF 后端。

```bash
pids=()

for spec in '0 mmlu' '1 arc' '2 gsm8k' '3 squadv2' '4 cmmlu' '5 clue'; do
  read -r gpu benchmark <<< "$spec"

  CUDA_VISIBLE_DEVICES="$gpu" \
  MODEL="$INS_MODEL" \
  MODEL_NAME="$INS_NAME" \
  EVAL_MODE=instruct \
  INFERENCE_BACKEND=hf \
  BENCHMARK="$benchmark" \
  BATCH_SIZE=1 \
  MAX_LENGTH=8192 \
  LIMIT= \
  RUN_NAME=full \
  bash scripts/evaluation/lmeh/run.sh &

  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" == 0 ]]
```

MMLU、CMMLU、GSM8K 默认 **5-shot**，其余 **0-shot**。

**13. 下游正式评估**

统一入口自动读取三个下游 Adapter 的完成指针，按任务顺序评估：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL="$INS_MODEL" \
MODEL_NAME="$INS_NAME" \
INFERENCE_BACKEND=hf \
BATCH_SIZE=1 \
BENCHMARKS=csc,fcgec_valid,fcgec_test,nacgec,nacgec_test,ancient_c3,ancient_fuxi \
RUN_NAME=full \
bash scripts/evaluation/run.sh
```



Fuxi 的裁判项需要额外配置 `JUDGE_MODEL`、`JUDGE_BASE_URL` 和 `JUDGE_API_KEY`；没有配置时相关指标保持 pending。无标签测试集产出提交文件。

以上完成的是**级联 PGCA 主线**。八卡 40G 的配置以当前 1.7B 为前提；新机器仍需确认正式长序列训练的显存和 loss。重新打开终端后，需恢复第一步变量，再从完成指针读取相应阶段模型。

命名调整不会自动搬迁已存在的模型、checkpoint 或缓存。接续旧运行时，用 `MODEL` 显式指定旧 checkpoint，并将 `MODEL_NAME` 设置为该输入模型的新规范名。使用旧完成指针时也可显式设置 `ALIGN_OUT`、`ADAPTATION_OUT`；需要保留原运行目录进行 resume 时应显式沿用原 `OUT`。

例如旧 Phase1 使用的是 `${BASE_MODEL_NAME}-PGCA-ALIGN`，可以先读取旧指针，再在第 8 步命令中额外传入 `MODEL="$ALIGN_MODEL"`：

```bash
ALIGN_MODEL="$(cat "$PROJECT_ROOT/outputs/train/${BASE_MODEL_NAME}-PGCA-ALIGN/next_stage_model.txt")"
```

此时第 8 步的 `MODEL_NAME` 仍用新的 `$ALIGN_NAME`，新 CPT 结果按 `$CPT_NAME` 命名。
