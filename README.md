# CC-LLM

CC-LLM 在 Qwen3 Base 上构建单汉字语义词表，并加入音形特征、因果上下文编码器和 PGCA 分支。本仓库包含模型构建、继续预训练、下游 LoRA 微调与评估入口；数据和模型权重单独准备。

| 路线 | 正式训练顺序 | 下游产物 |
| --- | --- | --- |
| Base | 原基座 → 三项 LoRA | `B-CSC`、`B-CGEC`、`B-ACT` |
| Char | 字符模型 → Phase1 → Char CPT → 三项 LoRA | `B-Char-CPT-{CSC,CGEC,ACT}` |
| PGCA | 完整模型 → Phase1 → Phase2-A → Phase2-B → 三项 LoRA | `B-PGCA-CPT-{CSC,CGEC,ACT}` |

`B` 为原基座名称。构建从模型配置推导隐藏维度、注意力头维度和 PGCA 层位置，不绑定参数规模；其他架构需要单独适配。以下以 Qwen3-4B-Base 为例，完成后更换模型变量即可进行 Qwen3-8B-Base 或其他 Qwen3 Base 实验。

所有命令均在仓库根目录执行。按下列编号顺序操作；环境和数据准备一次，模型相关步骤按规模分别执行。`pipeline.md` 描述研究方法，实际执行以本文与 `scripts/` 为准。

## 1. 配置环境

需要 Linux x86_64、NVIDIA GPU/驱动和 Python 3.12。先检查机器：

```bash
nvidia-smi
python3.12 --version
```

如果已有 Python 3.12：

```bash
export PYTHON_BIN="$(command -v python3.12)"
```

如果没有 Python 3.12，但机器已有 Conda，可以使用清华社区镜像准备解释器：

```bash
conda create -n cc-python312 python=3.12 -y \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda activate cc-python312
export PYTHON_BIN="$(command -v python)"
```

Conda 在此仅提供解释器。训练、评估仍各使用一个项目虚拟环境。解释器位于其他位置时，也可直接设置 `PYTHON_BIN=/absolute/path/to/python3.12`。

配置国内端点并安装：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export TORCH_INDEX_BASE=https://mirrors.nju.edu.cn/pytorch/whl

bash scripts/environment/run.sh
```

| 项目 | 位置 / 行为 |
| --- | --- |
| 训练环境 | `.venv/train`：模型构建、CPT、LoRA |
| 评估环境 | `.venv/evaluation`：PPL、LMEH、vLLM、下游评分 |
| CUDA 构建渠道 | 从 `nvidia-smi` 推导；可用 `TORCH_CHANNEL=cu128` 等显式覆盖，但必须匹配驱动能力 |
| 在线依赖安装 | 安装助手 uv 将 Torch 与其余依赖共同解析；CUDA 镜像优先、普通包使用清华镜像 |
| 完成检查 | `pip check`、关键模块导入及 CUDA 张量测试 |

环境脚本不固定具体包版本、不生成依赖锁或 Torch 约束文件，不调用 Git 命令。重新运行会重新解析依赖并从所选 CUDA 源重新安装 Torch；已完成配置的环境不需要为每个模型重复安装。选择 CUDA wheel 渠道不等于锁定 Torch 版本；渠道内必须存在与其余依赖兼容的组合。源优先策略见 [uv 文档](https://docs.astral.sh/uv/concepts/indexes/)。

安装完成后可查看实际环境：

```bash
.venv/train/bin/python -c \
'import torch, transformers, swift; print(torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.is_available())'

.venv/evaluation/bin/python -c \
'import torch, lm_eval; from vllm import LLM; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

后续脚本会使用对应环境，不需要反复 activate。不要用系统 Python 替代上述解释器。已有 `.venv` 如果使用其他 Python 版本，先移到备份位置，再重新创建。

包依赖与导入检查通过后，仍需用第 5 步验证实际模型路径。vLLM 预编译 wheel 与 Torch/CUDA 的二进制构建也需匹配，见 [vLLM 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。

如果国内镜像也无法访问，在可联网机器准备与目标 Linux、Python、CUDA 相匹配的完整 wheel 目录，再执行：

```bash
WHEELHOUSE=/absolute/path/to/wheels bash scripts/environment/run.sh
```

离线模式使用 `--no-index`，不会访问包索引。通用 benchmark 的数据也需要已有缓存，或在运行时能够访问 HF 镜像。

## 2. 放置数据并解压

先查看外层压缩包内容，确定是否自带 `data/` 目录：

```bash
# tar 包
tar -tf /absolute/path/to/data.tar.xz | head -20
# zip 包
unzip -l /absolute/path/to/data.zip | head -25
```

包内已有 `data/` 时解压到仓库根目录；包内直接是 `model_build/`、`pretrain/` 等目录时，解压到仓库的 `data/`。按实际格式选择一个命令：

```bash
# 包内已有 data/；tar 会识别常见压缩格式。
tar -xf /absolute/path/to/data.tar.xz -C .
# 或者
unzip /absolute/path/to/data.zip -d .
```

避免形成 `data/data/`。最终目录应为：

```text
data/
├── model_build/
│   ├── hanzi/
│   │   ├── tghz2013.txt
│   │   ├── common_traditional.txt
│   │   ├── structure.tsv
│   │   └── rare_high_freq.txt          # 可选
│   └── unihan/
│       ├── Unihan.zip                  # zip 或解压后的 Unihan 文本至少有一种
│       └── Unihan_*.txt
├── pretrain/
│   ├── phase1/
│   │   ├── train/*.tar.xz
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

预训练分片还需在训练前原地解压。每个 tar.xz 内为同名 JSONL，执行：

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

该命令保留压缩包，仅在成功解压后发布 JSONL。重复执行会跳过已有的非空 JSONL。正式训练只读取 `train/*.jsonl`；集中冒烟也可以从未解压的训练分片读取少量记录。

数据包不需要 manifest，也不需要重新运行语料生产流程。MMLU、ARC-Challenge、GSM8K、SQuAD v2、CMMLU、CLUE 由 LMEH 通过 HF 镜像获取并复用缓存，不要求放在数据包中，不固定数据 revision。

## 3. 选择本轮基座与 GPU

在同一个终端设置本轮变量。修改下面两项即可换模型或权重位置：

```bash
B=Qwen3-4B-Base
BASE="$PWD/models/$B"

# 按机器实际可用索引修改，下面是四卡示例。
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
```

如果基座已在其他目录，直接设置 `BASE=/absolute/path/to/model`。它应是完整 Hugging Face 模型目录，包含配置、tokenizer 和全部权重分片。`B` 是用于命名产物的名称，`BASE` 才是加载路径。

尚未准备基座时，可下载对应官方模型：

```bash
.venv/train/bin/hf download "Qwen/$B" --local-dir "$BASE"
```

`hf download` 使用上面配置的 HF 镜像；已有完整模型时跳过。这里准备的是 Base 权重，不用 Instruct、量化权重或 LoRA adapter 代替。

## 4. 构建正式 Char 与 PGCA

```bash
BASE_MODEL="$BASE" BASE_MODEL_NAME="$B" \
bash scripts/model_build/run.sh
```

依次执行词表/特征索引构建、语义 embedding 与 LM head 迁移、音形 embedding、PGCA 构建及校验。生成：

```text
models/<B>-Char/
models/<B>-PGCA/
```

维度和层号从所选 Base 配置推导。模型构建允许输出目录已经存在，会覆盖同名构建产物；训练依赖这些模型时不要再次向同一路径构建。

构建会更新 `models/model_name.txt` 与 `models/next_stage_model.txt`。它们只指向最近一次构建，多个规模的实验应像下文一样显式传入模型名称和路径。`MODEL_ROOT` 可整体更改构建输出与指针位置。

## 5. 集中冒烟

```bash
bash scripts/smoke/run.sh "$BASE"
```

检查 `outputs/smoke/vN/summary.md` 与 `summary.json`。三条路线各占一张可见 GPU 并行，卡不足时排队；每条路线内部串行。冒烟独立构建 Char/PGCA，产物放在本轮目录，不使用正式训练产物，也不覆盖正式模型指针。

默认每阶段训练 2 步、每个领域/任务评估少量样本。必测失败会标记 `failed`，依赖项标记 `blocked`，并给出日志位置。Fuxi 未配置裁判时为 `pending`，核心项目通过后的总体状态为 `partial`，退出码仍为 0。完整说明见 [集中冒烟](scripts/smoke/README.md)。

冒烟是单卡、batch 1、短序列验证，不代表正式 batch、长序列、多卡 DDP 或模型效果已经验收。先按机器资源确定正式参数。当前脚本没有启用 ZeRO/FSDP；普通数据并行不会把各 GPU 显存合成一个大模型空间。不能把多卡总显存直接当作单卡可用容量。

## 6. 三条路线的正式训练

以下三个入口均包含对应的 CSC、CGEC、Ancient LoRA，不需要在完成后再重复训练下游任务。按顺序执行，避免不同路线争用同一组 GPU。

```bash
# Base：保持原基座，直接做三项 LoRA。
MODEL_VARIANT=base MODEL_NAME="$B" BASE_MODEL="$BASE" \
bash scripts/train/run.sh

# Char：Phase1 → Char CPT → 三项 LoRA。
MODEL_VARIANT=char MODEL_NAME="${B}-Char" \
PHASE1_MODEL="$PWD/models/${B}-Char" \
bash scripts/train/run.sh

# PGCA：Phase1 → Phase2-A → Phase2-B → 三项 LoRA。
MODEL_VARIANT=pgca MODEL_NAME="${B}-PGCA" \
PHASE1_MODEL="$PWD/models/${B}-PGCA" \
bash scripts/train/run.sh
```

默认更新范围与配置入口：

| 阶段 | 更新范围 | 参数文件 |
| --- | --- | --- |
| Char Phase1 | 主干 | `scripts/train/cpt/phase1.sh`，`MODEL_VARIANT=char` |
| PGCA Phase1 | 主干，冻结 PGCA 与辅助头 | `scripts/train/cpt/phase1.sh` |
| Char CPT | 主干全参数 CLM | `scripts/train/cpt/char.sh` |
| PGCA-A | 音形分支与辅助头，冻结主干 | `scripts/train/cpt/phase2_a.sh` |
| PGCA-B | 主干、分支和辅助头联合训练 | `scripts/train/cpt/phase2_b.sh` |
| CSC/CGEC/Ancient | 主干及 PGCA 线性层的 LoRA，关闭辅助损失 | `scripts/train/finetune/{common,csc,cgec,ancient}.sh` |

阶段参数放在各脚本顶部，同名环境变量优先。正式默认 batch 如下，8 卡数值不代表固定使用 8 个进程：

| 阶段 | 每卡 batch | 梯度累积 | 8 卡有效 batch |
| --- | ---: | ---: | ---: |
| Phase1 / Char CPT | 8 | 4 | 256 |
| PGCA-A | 8 | 1 | 64 |
| PGCA-B | 16 | 2 | 256 |
| CSC / CGEC | 8 | 1 | 64 |
| Ancient | 2 | 4 | 64 |

有效 batch = 实际 GPU 数 × `TRAIN_BATCH_SIZE` × `GRAD_ACCUMULATION`。例如四卡每卡 batch 1、累积 64，对应有效 batch 256。修改卡数时按公式调整，不能将表中每卡 batch 直接当作全局 batch。显存是否容纳还需实卡验证，不因模型能成功加载就视为可以训练。

选择非连续显卡也可以：

```bash
CUDA_VISIBLE_DEVICES=1,3 \
MODEL_VARIANT=pgca MODEL_NAME="${B}-PGCA" \
STAGES=phase2_b \
bash scripts/train/run.sh
```

默认按可见卡数推导 `NPROC_PER_NODE`；手动指定时应与选卡数一致。需要覆盖参数时可在阶段命令前加 `TRAIN_BATCH_SIZE=1 GRAD_ACCUMULATION=... DATASET_NUM_PROC=1`，或修改对应脚本顶部。不要把本轮实验的参数改动当作不影响比较的透明优化。

训练日志在启动时打印的 `.log` 文件中。每个版本目录包含 `checkpoint_selection.json`，记录 best/last、选择原因、验证指标及有效目标统计。Phase2-A 选择 last，其他阶段优先选择具有有限指标的 best，否则记录回退到 last。出现 NaN 时应检查选优报告，不能把正常退出当作验证通过。

下游 LoRA 每次保存 checkpoint，还会自动执行每任务 1 条固定 Dev 自由生成，检查结束符监督及生成健康状况，写入 `checkpoint-N/dev_check/vN/`。失败保留报告并继续训练，不改变原有 loss 选模。正式评估前会对选中的模型再做完整验收，见下一节。

## 7. 三条路线的正式评估

vLLM 正式默认 batch 32、TP 4，需要至少四张可见卡。卡数少时显式降低 TP；PGCA 默认 HF，TP 参数不会为 HF 自动启用四卡张量并行。

```bash
# 本例 CUDA_VISIBLE_DEVICES 是四卡。
export TENSOR_PARALLEL_SIZE=4

MODEL_NAME="$B" MODEL="$BASE" \
bash scripts/evaluation/run.sh

MODEL_NAME="${B}-Char-CPT" \
bash scripts/evaluation/run.sh

MODEL_NAME="${B}-PGCA-CPT" \
bash scripts/evaluation/run.sh
```

入口读取该模型的 CPT 完成指针和三项 adapter 指针。Base 显式使用 `BASE`；Char/PGCA 使用训练阶段发布的权重。每条路线覆盖：

| 项目 | 数据 / 协议 |
| --- | --- |
| PPL | 六个中文领域，原始文本 |
| LMEH | MMLU、CMMLU、GSM8K 为 5-shot；ARC-Challenge、SQuAD v2、CLUE 为 0-shot |
| CSC | CSCD-NS、LEMON 七域，句子级纠正 P/R/F1 |
| CGEC | FCGEC valid、NaCGEC test；公开 FCGEC test 仅导出提交文件 |
| Ancient | C³Bench 五任务、Fuxi 21 任务 |

如需完成 Fuxi 模型裁判项，在执行对应评估前配置可用接口：

```bash
export JUDGE_MODEL=your-judge-model
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
# 接口需要鉴权时再设置 JUDGE_API_KEY。
```

未配置裁判的任务保留 pending，不填零分；普通评分不会自动调用未指定的裁判服务。GSM8K/SQuAD v2 的生成预算分别是 512/256；本地 `MAX_NEW_TOKENS` 不覆盖这些 LMEH 预算。

CSC、CGEC、Ancient 的正式推理会先自动重新加载基座和 adapter，按真实训练模板执行固定 Dev 验收：每任务 10 条，ACT 三任务共 30 条。每任务正常停止率及 `<|im_end|>` 结束率均须至少 90%，截断、重复、异常率各不超过 10%；同时检查结束符 label、LoRA 实际生效、Cache 数值与生成状态。失败会阻止该项下游 benchmark 推理，报告写入本轮 `dev_check/vN/{report.json,samples.jsonl}`。PPL/LMEH 不受此 SFT 门槛影响。

采样量和阈值可通过 `DEV_CHECK_*` 配置，完整说明和独立验收命令见 [固定 Dev 生成验收](scripts/train/finetune/README.md#固定-dev-生成验收)。迁移数据后如原 Dev 路径失效，设置 `DEV_CHECK_DEV`。集中冒烟只报告生成质量，正式评估默认启用硬门槛。

评估结果位于 `outputs/evaluation/<任务>/<模型>/.../vN/`，重复执行自动新建版本。LMEH 的 `table_metrics.json`、其他评估 `metrics.json` 中的 `table_metrics` 可用于实验表；取数定义见 [评估说明](scripts/evaluation/README.md)。只读取完整运行的结果，不把冒烟分数作为正式实验结果。

## 8. 补齐 Phase1 后 PPL，切换下一个规模

统一评估已经包含 Base、Char-CPT、PGCA-CPT 的 PPL。实验表若比较 Phase1 后模型，还需评估初始 Char 与 PGCA-ALIGN：

```bash
MODEL="$PWD/models/${B}-Char" MODEL_NAME="${B}-Char" \
bash scripts/evaluation/ppl/run.sh

IFS= read -r ALIGN_MODEL < "outputs/train/${B}-PGCA-ALIGN/next_stage_model.txt"
MODEL="$ALIGN_MODEL" MODEL_NAME="${B}-PGCA-ALIGN" \
bash scripts/evaluation/ppl/run.sh
```

这两条分别使用未训练的 Char 与完成 Phase1 的 PGCA。Base/Char tokenizer 不同，token PPL 不能直接横向解释为优劣，应同时查看 BPC/BPB；六域算术平均与按总 NLL/总目标数计算的整体 PPL 也需区分。

完成一个规模后，重新设置变量并从第 3 步继续：

```bash
B=Qwen3-8B-Base
BASE="$PWD/models/$B"
```

环境和数据不需要重复准备。不同规模/路线各使用自己的名称、权重和缓存，后续不要依赖最近一次构建的全局指针。

## 继续、重跑与指定检查点

| 目的 | 操作 |
| --- | --- |
| 只运行某些阶段 | `STAGES=phase1`、`STAGES=phase2_a,phase2_b`、`STAGES=csc,cgec,ancient` 等 |
| 跳过已完成阶段 | 不显式指定该阶段输入，不设置 FORCE；有效完成指针存在时会跳过 |
| 完整重跑选中阶段 | `FORCE=true`；新训练版本，保留训练缓存 |
| 从保存的训练状态恢复 | 单阶段脚本设置 `RESUME_FROM_CHECKPOINT`；不是重跑总入口 |

第一次执行三路训练的命令显式指定了初始输入，因此原样再次执行会启动新版本，并重跑其依赖阶段。中断后不要用这一方式冒充断点恢复。

总入口的输入变量：

| 路线 / 阶段 | 输入变量 | 未指定时 |
| --- | --- | --- |
| Base 下游 | `BASE_MODEL` 或各任务输入变量 | `models/<MODEL_NAME>` |
| PGCA Phase1 | `PHASE1_MODEL` | 最近一次构建指针 |
| Char Phase1 | `PHASE1_MODEL` | `models/<MODEL_NAME>` |
| Char CPT | `CPT_MODEL` | 该 Char 的 ALIGN 指针 |
| PGCA-A | `PHASE2_A_MODEL` | 该 PGCA 的 ALIGN 指针 |
| PGCA-B | `PHASE2_B_MODEL` | 该 PGCA 的 adaptation 指针 |
| CSC / CGEC / Ancient | `CSC_MODEL` / `CGEC_MODEL` / `ANCIENT_MODEL` | 该路线的 CPT 指针 |

例如跳过已完成的 PGCA Phase1，接着运行 A/B：

```bash
MODEL_VARIANT=pgca MODEL_NAME="${B}-PGCA" \
STAGES=phase2_a,phase2_b bash scripts/train/run.sh
```

指定某个 CPT checkpoint 做 CSC：

```bash
MODEL_VARIANT=pgca MODEL_NAME="${B}-PGCA" \
CSC_MODEL=/absolute/path/to/cpt/checkpoint-N \
STAGES=csc bash scripts/train/run.sh
```

恢复 PGCA-B 的同一次训练：

```bash
MODEL_NAME="${B}-PGCA" \
MODEL=/absolute/path/to/joint/vN/checkpoint-N \
RESUME_FROM_CHECKPOINT=/absolute/path/to/joint/vN/checkpoint-N \
bash scripts/train/cpt/phase2_b.sh
```

恢复 CSC LoRA 时，脚本可从 adapter 配置读取原基座：

```bash
MODEL_NAME="${B}-PGCA-CPT" \
RESUME_FROM_CHECKPOINT=/absolute/path/to/CSC/vN/checkpoint-N \
bash scripts/train/finetune/csc.sh
```

恢复需保持原 GPU 数、batch、数据、预算、长度、种子和缓存，并使用原 `OUT`。自定义 `TRAIN_ROOT` 或 `OUT` 时恢复命令也要带上；checkpoint 必须属于该输出目录。正式训练默认保存优化器状态，冒烟的 model-only checkpoint 不用于此类恢复。

## 缓存与输出位置

| 产物 | 默认位置 |
| --- | --- |
| 模型源码、词表、权重 | `models/<B>-Char`、`models/<B>-PGCA` |
| Char 训练 | `outputs/train/<B>-Char-ALIGN`、`outputs/train/<B>-Char-CPT` |
| PGCA 训练 | `outputs/train/<B>-PGCA-ALIGN`、`outputs/train/<B>-PGCA-CPT/{adaptation,joint}` |
| 下游 adapter | `outputs/train/<输入模型名称>-{CSC,CGEC,ACT}` |
| 缓存 | `outputs/cache/<模型名称>/` |
| 评估 | `outputs/evaluation/` |

`TRAIN_ROOT` 可整体修改训练根目录，评估时需传同一个值以读取指针。单阶段可通过 `OUT` 修改输出目录；模型名称与检查点路径分开填写，不能将 `checkpoint-N` 当成模型名称。

PGCA-A/B 共用 `phase2_<长度>.raw` 原始编码，分别使用 `phase2_<长度>_adaptation` 和 `phase2_<长度>_joint` prepared 视图。缓存记录卡数、每卡 batch、长度、种子和预算；改变这些设置会触发校验。显式指定 `CACHE_DIR` 时默认 RAW 为它后面的 `.raw`，需要复用其他原始缓存时应显式设置 `RAW_CACHE`。

只改变卡数/batch、需要新的 prepared 视图时，可按当前阶段设置两个路径，例如：

```bash
MODEL_NAME="${B}-PGCA" \
RAW_CACHE="$PWD/outputs/cache/${B}-PGCA/phase2_4096.raw" \
CACHE_DIR="$PWD/outputs/cache/${B}-PGCA/phase2_4096_joint_new" \
bash scripts/train/cpt/phase2_b.sh
```

该例还需沿用所需的 GPU、batch、种子、预算，并确保 adaptation 指针存在，或显式指定 `MODEL`。如果更换 tokenizer、输入数据或切分长度，应重新导出原始编码，不能只换 prepared 路径。

## 常见意外与处理

| 现象 | 处理 |
| --- | --- |
| 找不到 Python 3.12 / 实际是 3.9 | 回到第 1 步准备解释器，设置 PYTHON_BIN。已有错误版本的 .venv 先移到备份位置，再安装。 |
| nvidia-smi 失败 / CUDA 不可用 | 先检查驱动、容器 GPU 挂载和可见设备，再安装 Python 包；CPU 安装不能替代 GPU 实验。 |
| 依赖解析失败，Torch/vLLM 无可用组合 | 核对驱动与 CUDA 源。使用完整匹配的 wheelhouse，或由管理员升级驱动后重建环境；不要用 --no-deps 绕过依赖。 |
| undefined symbol / vLLM 扩展导入失败 | 在新评估环境联合安装依赖，检查实际 Torch/CUDA。不要只升级 Torch 后继续使用旧 vLLM 二进制。 |
| 模块缺失 / PreTrainedConfig 等导入失败 | 确认使用项目 .venv 解释器；检查实际安装包与代码 API。重新完成联合安装和导入检查，再做模型构建。 |
| DNS、连接镜像超时 | 检查国内端点连通性；环境脚本已隔离额外包索引。模型与数据下载重试同一目录，或准备离线 wheel/数据缓存。 |
| 训练提示没有 JSONL | 完成外层数据解压及预训练分片解压，排除 data/data 嵌套。损坏的包先重新取得。 |
| 构建校验失败 | 查看 Char/PGCA 输出中的 reports；确认原基座、汉字资源、词表/embedding/特殊 token 映射。修正原因后重建，不绕过校验继续训练。 |
| CUDA OOM | 区分训练或推理；先减对应的 TRAIN_BATCH_SIZE、EVAL_BATCH_SIZE 或 BATCH_SIZE。训练需按目标有效 batch 调整梯度累积；卡数/batch 改变时准备新的缓存视图。模型状态本身仍装不下时需分片训练适配，当前没有可直接生效的 ZeRO/FSDP 开关。 |
| Map 阶段退出 / Killed | 查看实际阶段 .log 的退出码、信号和容器 OOM 增量；尝试 DATASET_NUM_PROC=1。CPT 单进程编码会在当前进程执行。不能仅凭进度条判断是内存还是崩溃。 |
| prepared cache geometry mismatch | 保持原卡数/batch，或按上面示例新建 prepared 视图，复用正确的 RAW_CACHE。 |
| raw cache incomplete | 保留原目录，用 RAW_CACHE 指向一个新的导出位置重试；完整 raw 的准备失败可直接重试。 |
| address already in use | 默认端口自动分配；需要手动指定时换 MASTER_PORT。不要终止不属于本任务的进程。 |
| device ordinal / TP 卡数错误 | 检查 CUDA_VISIBLE_DEVICES、NPROC_PER_NODE、TENSOR_PARALLEL_SIZE。进程内 cuda:0 是可见列表中的第一张卡。单卡 vLLM 设置 TP=1；HF 不使用该 TP。 |
| checkpoint 指针缺失 | 确认上游阶段成功、MODEL_NAME 与 TRAIN_ROOT 一致。需要从已有结果继续时显式传对应阶段的输入变量。 |
| 验证 loss 为 NaN | 查看有效监督数、标签切分与 logits 是否有限；无效监督和数值问题分开定位。检查选优报告，不能依赖无效 eval_loss 选 best。 |
| 生成持续重复、全部 length 结束 / Dev 验收失败 | 先看本轮 `dev_check/vN/report.json` 与 `samples.jsonl`，区分 EOS 监督、adapter、Cache 和生成质量失败。检查参考是否超过预算；不能只增大生成上限解决模型重复。 |
| 评估缺 args.json / 插件路径失效 | 使用对应 SFT adapter，保留其训练参数及可访问的插件源码。迁移后修正插件路径，不能用猜测模板替代。 |
| FCGEC test 没有分数 | 公开 test 无本地 gold，只导出提交文件；本地评分用 valid。 |
| Fuxi 结果 pending | 补充裁判接口后对原 RUN_DIR 执行 judge，再执行 score；不需要重新训练。 |
| 重复运行出现新 vN | 这是新版本输出。若要继续同一次训练，使用 resume；继续已有评估，使用 RUN_DIR 和 infer/score。 |

数据读取、缓存和训练会在阶段日志中分开计时。日志中的 `container_oom_kill_delta` 是容器范围计数，不是对某个进程的单独归因。

继续已有的 CSC 推理与评分示例：

```bash
MODEL=/absolute/path/to/base-or-cpt \
LORA=/absolute/path/to/CSC/adapter-checkpoint \
RUN_DIR=/absolute/path/to/evaluation/vN STAGE=infer \
bash scripts/evaluation/csc/run.sh

RUN_DIR=/absolute/path/to/evaluation/vN STAGE=score \
bash scripts/evaluation/csc/run.sh
```

保持原模型、adapter、输入与生成设置。CGEC 同时保留原 DATASET/SPLIT；古汉语保留 DATASET/FEWSHOT。为已有 Fuxi 预测补裁判后需重新评分：

```bash
RUN_DIR=/absolute/path/to/fuxi/vN DATASET=fuxi STAGE=judge \
bash scripts/evaluation/ancient/run.sh

RUN_DIR=/absolute/path/to/fuxi/vN DATASET=fuxi STAGE=score \
bash scripts/evaluation/ancient/run.sh
```

上述 judge 命令需已配置 JUDGE_MODEL/JUDGE_BASE_URL。实际通过需以完整预测、评分文件和对应检查点记录为准，不能只凭目录存在或命令退出码判断模型效果。

## 代码目录

```text
src/                         词表、embedding、Qwen3-PGCA 模型
scripts/environment/         两套环境的安装入口
scripts/model_build/         Char 与 PGCA 构建
scripts/smoke/               集中小规模全流程
scripts/train/               三路正式训练与检查点衔接
scripts/evaluation/          通用和下游评估
requirements/                不固定版本的普通包依赖
```

本仓库不包含完整语料生产或所有研究消融脚本。安装/入口检查与少步冒烟不等于已经完成目标机器上的完整模型训练和正式评测。
