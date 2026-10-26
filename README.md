# CC-LLM

CC-LLM 在 Qwen3 Base 模型上构建单汉字语义词表，并加入音形特征、因果上下文编码器和 PGCA 分支。仓库包含模型构建、继续预训练、下游 LoRA 微调与完整评估代码，不包含数据集、模型权重和运行产物。

代码不绑定某个参数规模。选择不同的本地 Qwen3 Base 权重即可复用同一套流程；隐藏维度、注意力头数和 PGCA 层位置均从所选模型配置推导。

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

Base 模型也必须是本地完整 Hugging Face 模型。默认放在 `models/base/`，也可以在构建时传入 `BASE_MODEL=/path/to/model`。

## 四步运行

### 1. 环境

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

```bash
bash scripts/model_build/run.sh
```

执行顺序：词表与特征索引 → embedding 迁移 → PGCA 构建 → 模型校验。输出名称由 Base 模型目录名自动生成，也可显式设置：

```bash
BASE_MODEL=/path/to/base BASE_MODEL_NAME=Qwen3-Base \
bash scripts/model_build/run.sh
```

构建完成后，`models/next_stage_model.txt` 和 `models/model_name.txt` 供训练入口自动读取。

### 3. 训练

```bash
bash scripts/train/run.sh
```

默认依次执行：

```text
Phase1 → Phase2-A → Phase2-B → CSC LoRA → CGEC LoRA → Ancient LoRA
```

只运行部分阶段：

```bash
STAGES=phase1,phase2_a,phase2_b bash scripts/train/run.sh
STAGES=csc,cgec,ancient bash scripts/train/run.sh
```

每个阶段的学习率、batch、长度、保存间隔等参数都直接位于对应脚本顶部，不通过隐藏配置层转发：

```text
scripts/train/cpt/phase1.sh
scripts/train/cpt/phase2_a.sh
scripts/train/cpt/phase2_b.sh
scripts/train/finetune/common.sh
scripts/train/finetune/{csc,cgec,ancient}.sh
```

完整阶段通过 `next_stage_model.txt` 自动衔接。已完成阶段默认跳过；`FORCE=true` 可启动新一轮。中断训练使用各阶段脚本的 `RESUME_FROM_CHECKPOINT`，不会猜测最大编号的 checkpoint。

### 4. 评估

```bash
bash scripts/evaluation/run.sh
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
