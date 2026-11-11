
# CC-LLM 完整实验流程与复现方案

## 1. 实验概述

### 1.1 研究目标

CC-LLM 旨在研究汉字级语义表示与显式音形特征融合，是否能够改善中文语言模型的表征能力，并进一步提升中文拼写纠错、语法纠错及古汉语处理等任务的表现。

研究包含三个主要技术环节：

1. **字符级语义重构**：将目标汉字映射到独立 Token，并迁移原始 Qwen3 Base 的预训练权重。
2. **音形信息建模**：为汉字构建多候选拼音和字形结构表示，通过因果上下文编码器和 PGCA 融入主干。
3. **分阶段继续预训练**：依次完成字符级语义适配、音形分支适配和全参数联合优化。

为判断上述方法的实际效果，实验设置 Base、Char 和 PGCA 三条路线。

### 1.2 实验模型与对照组

设原始 Qwen3 Base 模型为 \(B\)。

| 模型路线 | 模型构成 | 继续预训练 | 实验目的 |
|---|---|---|---|
| Base | 原始 Qwen3 Base | 不额外进行 CPT | 原始基础模型参照 |
| Char | Qwen3 + 汉字级 Tokenizer | Phase1 + Char-CPT | 衡量字符级重构及继续预训练的综合效果 |
| PGCA | Qwen3 + Char Tokenizer + 音形编码器 + PGCA | Phase1 + Phase2-A/B | 评价完整音形增强方案 |

完整实验树：

```text
Qwen3 Base
│
├── Base
│   ├── CSC LoRA ────────── CSC Evaluation
│   ├── CGEC LoRA ───────── CGEC Evaluation
│   ├── Ancient LoRA ────── Ancient Evaluation
│   └── PPL / General Benchmarks
│
└── Character Vocabulary Construction
    │
    ├── Char
    │   └── Phase1: Semantic Alignment
    │       └── Char-ALIGN
    │           └── Phase2: Full-parameter CLM
    │               └── Char-CPT
    │                   ├── CSC LoRA
    │                   ├── CGEC LoRA
    │                   ├── Ancient LoRA
    │                   └── PPL / General Benchmarks
    │
    └── PGCA Construction
        ├── Phonetic-Glyph Embedding
        ├── Causal Feature Context Encoder
        ├── PGCA Cross-Attention
        └── Auxiliary Prediction Heads
            │
            └── Phase1: Semantic Alignment
                └── PGCA-ALIGN
                    └── Phase2-A: Branch Adaptation
                        └── PGCA-ADAPTATION
                            └── Phase2-B: Joint CPT
                                └── PGCA-CPT
                                    ├── CSC LoRA
                                    ├── CGEC LoRA
                                    ├── Ancient LoRA
                                    ├── PPL / General Benchmarks
                                    └── PGCA Mechanism Analysis
```

重要的比较边界：

- Base 没有接受额外 CPT，因此 Base 与 Char-CPT/PGCA-CPT 的差异同时包含训练数据和训练预算的影响。
- Char 与 PGCA 使用相同的预训练语料及名义 Token 预算，但参数规模、Phase2 优化阶段和辅助目标不同。
- Char-CPT 与 PGCA-CPT 是研究音形架构增益的重要对照，但仍需要在论文中说明上述优化差异。
- 同一模型 CPT 前后的对比，用于判断继续预训练效果；不能直接替代跨架构比较。

---

## 2. 实验环境与数据准备

### 2.1 实验环境

源码入口：

```text
scripts/environment/run.sh
```

当前 master 要求：

- Linux x86_64
- Python 3.12
- NVIDIA GPU 和兼容的 CUDA 驱动
- 独立的训练与评估环境

执行：

```bash
bash scripts/environment/run.sh
```

生成：

```text
.venv/train/
.venv/evaluation/
```

训练环境包含 PyTorch、Transformers、ms-swift、PEFT 等依赖。评估环境包含通用评测及下游评分依赖。

默认 Hugging Face 端点：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

PyTorch CUDA Wheel 版本由环境脚本根据 GPU 驱动能力选择。正式实验应记录实际安装的 CUDA、PyTorch、Transformers、ms-swift 和 PEFT 版本，而非仅记录脚本默认值。

### 2.2 数据目录

当前 master 不包含历史 Data Factory 的完整原始语料生产流程，而是直接消费已经准备完成的数据。

应将数据放在：

```text
data/
├── model_build/
│   ├── hanzi/
│   │   ├── tghz2013.txt
│   │   ├── common_traditional.txt
│   │   ├── structure.tsv
│   │   └── rare_high_freq.txt
│   └── unihan/
│       ├── Unihan.zip
│       └── Unihan_*.txt
│
├── pretrain/
│   ├── phase1/
│   │   ├── train/
│   │   └── validation/
│   └── phase2/
│       ├── train/
│       └── validation/
│
├── finetune/
│   ├── csc/
│   ├── cgec/
│   └── ancient/
│
└── evaluation/
    ├── ppl/
    ├── csc/
    ├── cgec/
    └── ancient/
```

预训练数据按照 Phase1、Phase2 独立组织。

训练文件可能以 `.tar.xz` 形式提供。正式训练之前，必须先解压得到对应的 JSONL 文件。

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

预训练脚本只读取解压后的 JSONL，不自动解压压缩包。

实验数据需要额外固定：

- 训练与验证集的文件清单、哈希和样本数量；
- 语料来源、实际 Token 数和类别分布；
- 精确去重、近重复去重和 Benchmark 去污染记录；
- Phase1/Phase2 的数据版本和随机种子。

上述原始语料构建信息需要由独立数据生产记录提供，不能仅根据当前 master 源码还原。

---

## 3. 各类模型构建

源码入口：

```text
scripts/model_build/run.sh
scripts/model_build/build_vocab.sh
scripts/model_build/build_embedding.sh
scripts/model_build/build_pgca.sh
```

建议在构建开始前固定 Base 模型及源码版本。

### 3.1 基础模型：Qwen3 Base

首先准备完整的 Qwen3 Base 模型。

以 Qwen3-1.7B-Base 为例：

```bash
BASE_MODEL=/absolute/path/to/Qwen3-1.7B-Base
BASE_MODEL_NAME=Qwen3-1.7B-Base
```

基础模型必须包含：

```text
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
model.safetensors 或权重分片
```

Base 不改变模型结构和 Tokenizer，作为原始能力及下游适配基线。

不同参数规模的 Qwen3 Base 可以复用构建入口，但 GPU 资源、模型大小和 PGCA 层数需要依据具体配置确认。

### 3.2 Char 模型：汉字级语义词表构建

主要源码：

```text
src/vocab/hanzi_set.py
src/vocab/semantic_vocab.py
src/vocab/feature_vocab.py
src/vocab/feature_index.py
src/embedding/semantic_embedding.py
```

#### 步骤 3.2.1：构建目标汉字集合

整合《通用规范汉字表》、常用繁体字表，以及可选的长尾高频汉字资源。

构建目标集合：

$$\mathcal C = \{c_1,c_2,\ldots,c_N\}$$

此处的 \(N\) 由实际资源决定，不是源码中固定的常量。

验证目标是使集合内的汉字能够拥有独立的语义 Token。

#### 步骤 3.2.2：重构语义词表

基于原始 Qwen3 BPE 词表：

1. 保留特殊 Token 与必要的非汉字 Token。
2. 移除符合规则的多汉字及汉字混合 Token。
3. 保留构成目标单字所必需的 BPE 依赖。
4. 补全尚无独立 Token 的目标汉字。
5. 重建连续 Token ID，并保存新旧词表映射。

Char 是目标汉字级的语义重构，不是将所有英文、数字和符号强制拆分为单字符。

#### 步骤 3.2.3：迁移语义权重

对原词表中已有的 Token，复制输入 Embedding 和 LM Head 对应行。

对于新增 Token，使用原始 Tokenizer 编码后的旧 Token 权重均值初始化：

$$
E_{\mathrm{new}}(c)=
\frac{1}{|\pi(c)|}
\sum_{u\in\pi(c)}E_{\mathrm{base}}(u)
$$

LM Head 采用对应的迁移策略。

这一过程避免了因重构词表而完全丢失原始预训练知识。

需要验证的关键对象包括：

- `new2old_token_id.json`
- `new_token_init_token_ids.json`
- 实际 Tokenizer ID
- 输入 Embedding
- LM Head
- BOS/EOS/ChatML 特殊 Token 映射

尤其需要检查特殊 Token 的新旧 ID 与权重行一致性。

### 3.3 音形特征与索引构建

Char 模型构建时还会生成音形资源，供 PGCA 模型使用。

#### 拼音特征

每个汉字保留多个合法拼音候选。

每个候选包含：

```text
完整拼音
声母
韵母
声调
```

数据来自 pypinyin 及 Unihan 拼音属性。

对一个拼音候选 \(p_j\)，编码为：

$$
f^p_j=\operatorname{MLP}_p
([E_p(p_j);E_i(i_j);E_f(f_j);E_t(t_j)])
$$

#### 字形特征

字形表示包含：

```text
总笔画数
部首笔画组合
汉字结构类别
```

编码为：

$$f^g=\operatorname{MLP}_g([E_s(s);E_r(r);E_z(z)])$$

这里使用的是结构化正字法属性，而非汉字图片或视觉字形编码。

#### 特征槽

默认每个汉字最多包含：

```text
16 个拼音候选槽
 1 个字形结构槽
----------------
17 个音形特征槽
```

不足 16 个拼音候选时通过 Padding 和 Mask 补齐。

对每个语义 Token ID 生成对应的音形特征索引：

```text
features/feature_index.pt
```

这样在模型推理时，可以根据输入 Token ID 直接查表，获取该位置的候选特征。

### 3.4 执行 Char 构建

上述步骤已经被统一脚本封装。

```bash
BASE_MODEL="$BASE_MODEL" \
BASE_MODEL_NAME="$BASE_MODEL_NAME" \
bash scripts/model_build/run.sh
```

该命令首先完成 Char 的词表、音形索引及权重迁移，再继续构建 PGCA。

Char 模型的主要输出：

```text
models/Qwen3-1.7B-Base-Char/
```

它是一个仍采用标准 Qwen3 Decoder 的模型，尚未引入 PGCA 交叉注意力。

因此可以将它用于单独研究：

**仅修改语义 Tokenizer，在相同 CPT 数据下，能产生怎样的效果？**

### 3.5 PGCA 模型构建

主要源码：

```text
src/embedding/feature_embedding.py
src/embedding/context_encoder.py
src/pgca/attention.py
src/pgca/auxiliary.py
src/pgca/build_pgca.py
src/modeling_qwen3_pgca.py
```

PGCA 在 Char 模型之上新增音形建模路径。

#### 步骤 3.5.1：构建静态音形 Embedding

把各汉字的拼音候选和字形结构属性转换为独立特征向量。

每个位置的初始特征记忆为：

\[
F_t^{(0)}=
[f^p_{t,1},\ldots,f^p_{t,K},f^g_t]
\]

其中默认 \(K=16\)。

#### 步骤 3.5.2：构建因果音形上下文编码器

静态音形特征本身无法区分“银行”和“运行”等语境中的候选读音。

因此构建独立的轻量级因果 Transformer：

\[
C_t=
\operatorname{CausalEncoder}(F_{\leq t}^{(0)})
\]

编码器只访问当前位置及其左侧信息，避免自回归预测时发生未来信息泄漏。

源码默认配置：

- 4 层 Transformer；
- 隐藏维度 512；
- 8 个注意力头；
- 前馈中间维度 1536；
- 采用 RoPE 和因果注意力。

上下文编码器先汇总每个 Token 的有效拼音与字形信息，再将上下文增量注入各个独立特征槽。

这样既引入了上下文，又保留多音字的不同候选表示。

#### 步骤 3.5.3：构建 PGCA 交叉注意力

PGCA 使用语义主干当前层的隐藏状态作为 Query，使用音形记忆作为 Key/Value。

\[
Q_l=W_Q\operatorname{Norm}(H_l)
\]

\[
K_l=W_KF,\qquad V_l=W_VF
\]

\[
P_l=
W_O\operatorname{Attention}(Q_l,K_l,V_l)
\]

注意：PGCA 不是把当前 Token 与整段音形序列做全局 Cross-Attention，而是在每个位置的有效音形槽上计算候选注意力。

#### 步骤 3.5.4：并行融合

Self-Attention 与 PGCA 从同一层输入的归一化隐藏状态出发。

两者独立计算，然后融合：

\[
H_l'=
H_l+\operatorname{SelfAttn}_l(H_l)
+\operatorname{PGCA}_l(H_l,F)
\]

随后继续原有 MLP 残差。

该结构没有自适应 Gate，而是直接采用并行残差相加。

设计目的是在保留预训练语义主干原有路径的同时，让音形特征成为独立的隐藏状态更新来源。

#### 步骤 3.5.5：初始化新增分支

PGCA 输出投影矩阵采用零初始化：

\[
W_O=0
\]

因此，模型刚构建完成时，PGCA 输出为零。

训练开始后，输出投影逐渐更新，使分支在早期尽量降低对原始语义路径的扰动。

#### 步骤 3.5.6：构建辅助预测头

新增两类辅助任务：

1. 下一汉字合法拼音集合预测。
2. 下一汉字的笔画数、部首笔画组合和结构类别预测。

其目的在于为音形相关表示提供额外训练信号。

字音任务使用合法读音集合似然，不直接监督语境中唯一正确的多音字读音。

### 3.6 PGCA 构建命令

`build_pgca.sh` 默认启用上下文编码器、辅助头和零初始化。

```bash
bash scripts/model_build/build_pgca.sh
```

默认辅助损失权重：

```text
phonetic_weight = 0.03
glyph_weight    = 0.03
```

模型输出：

```text
models/Qwen3-1.7B-Base-PGCA/
```

同时运行：

```text
src/pgca/validate_pgca.py
```

进行模型结构和参数校验。

#### 重要：PGCA 层选择必须显式记录

当前 `master` 不再固定某个参数规模的 PGCA 层号。

当没有传入 `--pgca-layers` 时，`build_pgca.py` 默认选取主干中间三分之一层。

例如，对于 28 层主干，该算法选择 0-based 第 9–17 层，共 9 层。

这与此前实验记录中的第 11–16 层不同。

如果需要严格复现原先的六层 PGCA 实验，应该在独立输出目录中显式构建：

```bash
.venv/train/bin/python -m src.pgca.build_pgca \
  --model-path models/Qwen3-1.7B-Base-Char \
  --output-path models/Qwen3-1.7B-Base-PGCA-L11to16 \
  --pgca-layers 11,12,13,14,15,16 \
  --context-encoder \
  --auxiliary-heads \
  --zero-init-output \
  --aux-phonetic-weight 0.03 \
  --aux-glyph-weight 0.03
```

然后执行：

```bash
.venv/train/bin/python -m src.pgca.validate_pgca \
  --model-path models/Qwen3-1.7B-Base-PGCA-L11to16
```

之后训练必须使用该模型目录及对应的独立 `MODEL_NAME`。

不能把九层 PGCA 和六层 PGCA 的训练结果视为同一配置。

---

## 4. 继续预训练（CPT）

### 4.1 预训练目标

采用标准自回归语言建模：

\[
\mathcal L_{\mathrm{CLM}}
=
-\frac{1}{N}\sum_{t\in\Omega}
\log P(x_{t+1}\mid x_{\leq t})
\]

其中 \(\Omega\) 是有效监督位置集合。

PGCA Phase2 进一步引入多任务目标：

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{CLM}}
+
\gamma(n)
\left(
0.03\mathcal L_{\mathrm{phon}}
+
0.03\mathcal L_{\mathrm{glyph}}
\right)
\]

其中 \(\gamma(n)\) 是基于有效训练 Token 数的辅助任务权重预热，默认在前 10M 有效 Token 内线性增长至 1。

CPT 不使用 LoRA，而是根据阶段冻结规则进行全参数或分支参数训练。

### 4.2 Phase1：字符级语义适配

执行入口：

```text
scripts/train/cpt/phase1.sh
```

输入为新构建的 Char 或 PGCA 模型。

对于 PGCA 路线：

- 关闭 PGCA 前向；
- 冻结音形 Embedding、上下文编码器、PGCA 和辅助头；
- 优化原始语义主干与重构后的词嵌入；
- 仅使用 CLM 损失。

对于 Char 路线：

- 直接训练标准 Char 主干；
- 不存在 PGCA 分支。

目的：

使模型适应新 Tokenizer 的汉字级序列分布，并恢复或提升字符级语言建模能力。

默认配置：

| 参数 | Phase1 |
|---|---|
| Token 上限 | 1B |
| 最大序列长度 | 2048 |
| 学习率 | 1e-5 |
| 每卡 Batch | 1 |
| 梯度累积 | 32 |
| Warm-up | 3% |
| Weight Decay | 0.01 |
| 训练精度 | BF16 |
| 优化器 | AdamW |

训练完成：

```text
Char-ALIGN
PGCA-ALIGN
```

PGCA-ALIGN 虽然保留 PGCA 模块结构，但 Phase1 的计算过程中不使用该分支。

### 4.3 Phase2-A：音形分支适配

执行入口：

```text
scripts/train/cpt/phase2_a.sh
```

仅用于 PGCA 路线。

输入：

```text
PGCA-ALIGN
```

训练策略：

- 启用 PGCA；
- 冻结语义主干；
- 解冻音形 Embedding、上下文编码器、PGCA 和辅助头；
- 使用 CLM + 字音 + 字形联合损失。

默认配置：

| 参数 | Phase2-A |
|---|---|
| 有效 Token 上限 | 100M |
| 最大序列长度 | 4096 |
| 分支学习率 | 5e-5 |
| 每卡 Batch | 1 |
| 梯度累积 | 8 |
| Warm-up | 3% |
| 模型选择 | Last checkpoint |

此阶段相当于让新分支在固定语义主干上学习如何辅助预测。

模型输出：

```text
PGCA-CPT/adaptation/
```

### 4.4 Phase2-B：联合继续预训练

执行入口：

```text
scripts/train/cpt/phase2_b.sh
```

输入：

```text
PGCA-CPT/adaptation 的完成 checkpoint
```

训练策略：

- 解冻整个模型；
- 联合优化语义主干、音形分支和辅助预测头；
- 主干与新增分支使用不同学习率。

默认配置：

| 参数 | Phase2-B |
|---|---|
| Phase2 总 Token 上限 | 10B |
| Phase2-A 预算 | 100M |
| Phase2-B 预算 | 剩余约 9.9B |
| 最大序列长度 | 4096 |
| 主干学习率 | 1e-5 |
| 音形分支学习率 | 3e-5 |
| 每卡 Batch | 1 |
| 梯度累积 | 32 |
| Warm-up | 3% |
| Weight Decay | 0.01 |
| 模型选择 | 最优验证 CLM Loss |

注意：这里记录的是当前 `master` 的默认 Warm-up 3%，不应与此前单独实验采用的 1% 配置混为一谈。

Phase2-A 与 Phase2-B 使用同一 Phase2 数据池，通过有效监督 Token 区间进行划分，避免两个阶段重复使用同一监督目标。

模型输出：

```text
PGCA-CPT/joint/
```

完成后发布：

```text
PGCA-CPT/next_stage_model.txt
```

### 4.5 Char 对照模型的 Phase2

Char 不执行音形分支适配。

执行入口：

```text
scripts/train/cpt/char.sh
```

输入：

```text
Char-ALIGN
```

使用同一 Phase2 语料、默认 10B Token 预算进行全参数 CLM 继续预训练。

与 PGCA 路线不同：

- 无 PGCA；
- 无音形上下文编码器；
- 无音形辅助损失；
- 不划分独立的 Branch Adaptation 阶段。

输出：

```text
Char-CPT
```

Char-CPT 是判断完整 PGCA-CPT 是否具有额外价值的关键比较对象。

### 4.6 三路线训练命令

设：

```bash
B=Qwen3-1.7B-Base
BASE_MODEL=/absolute/path/to/Qwen3-1.7B-Base
```

#### Base 路线

Base 不进行上述额外 CPT，直接进行下游 LoRA。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_VARIANT=base \
MODEL_NAME="$B" \
BASE_MODEL="$BASE_MODEL" \
bash scripts/train/run.sh
```

#### Char 路线

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_VARIANT=char \
MODEL_NAME="${B}-Char" \
PHASE1_MODEL="$PWD/models/${B}-Char" \
bash scripts/train/run.sh
```

#### PGCA 路线

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_VARIANT=pgca \
MODEL_NAME="${B}-PGCA" \
PHASE1_MODEL="$PWD/models/${B}-PGCA" \
bash scripts/train/run.sh
```

上述命令均使用各路线的默认阶段集合。

若仅训练 CPT：

```bash
MODEL_VARIANT=pgca \
MODEL_NAME="${B}-PGCA" \
PHASE1_MODEL="$PWD/models/${B}-PGCA" \
STAGES=phase1,phase2_a,phase2_b \
bash scripts/train/run.sh
```

若已完成 CPT，只训练下游：

```bash
MODEL_VARIANT=pgca \
MODEL_NAME="${B}-PGCA" \
STAGES=csc,cgec,ancient \
bash scripts/train/run.sh
```

每个阶段的参数均允许通过同名环境变量覆盖。

如果要恢复训练，应使用对应单阶段脚本与 `RESUME_FROM_CHECKPOINT`，不能把已有断点路径误当成新阶段输入。

### 4.7 CPT 训练记录

每个阶段至少保存：

```text
args.json
trainer_state.json
checkpoint_selection.json
next_stage_model.txt
```

还需要记录：

- 当前源码 commit；
- 初始模型与 Tokenizer；
- 实际训练 Token 数；
- Train/Validation 数据版本；
- optimizer step；
- CLM Loss、Total Loss；
- Token Accuracy；
- 字音与字形辅助指标；
- 主干和 PGCA 分支梯度；
- 学习率、有效 Batch、随机种子；
- 所选 checkpoint 的依据。

`checkpoint_selection.json` 记录实际采用 Best 还是 Last，以及对应原因。

不能仅依据最大 checkpoint 编号推测最终选中的模型。

---

## 5. 下游任务微调（SFT）

下游任务统一使用 LoRA。

执行入口：

```text
scripts/train/finetune/
├── common.sh
├── csc.sh
├── cgec.sh
└── ancient.sh
```

三条实验路线均执行相同类型的 SFT。

### 5.0 通用指令微调（General-SFT）

专项 SFT 之前，三条路线均先完成 General-SFT：Base → Base-INS，Char-CPT → Char-CPT-INS，PGCA-CPT → PGCA-CPT-INS。使用同一 COIG-CQIA/Tulu 3 清洗池和统一的目标监督预算，正式预算待真实数据统计后确定。仅监督 Assistant 及 EOS，采用保存的 Qwen3 非思考 ChatML。

入口为 `scripts/train/instruction/run.sh`。General-SFT 使用 LoRA，按有限 Dev loss 排序并要求完整生成/Adapter/Cache 验收，合并后复验，通过后发布完整 `M-INS` 权重。专项任务从该完整权重开始，各自训练独立 Adapter，不叠加 General-SFT Adapter。完整对话预算可能略超目标，必须用准备报告与实际训练日志对齐三路监督量，不能仅凭相同配置宣称严格相等。数据与执行细节见 [General-SFT](scripts/train/instruction/README.md)。

### 5.1 LoRA 微调机制

默认配置：

```text
LoRA Rank    = 16
LoRA Alpha   = 32
LoRA Dropout = 0.05
Target       = all-linear
```

对于 PGCA 模型，LoRA 覆盖主干以及 PGCA 和上下文编码器中的相应 Linear 模块。

SFT 不再计算字音和字形辅助损失，辅助预测头被冻结。

实际哪些 Linear 模块进入可训练范围，以生成的 `finetune_report.json` 为准。

三项任务分别训练独立 Adapter，不能将 CSC Adapter 继续训练成 CGEC Adapter 再进行公平比较。

### 5.2 中文拼写纠错（CSC）

输入模型：

```text
Base
Char-CPT
PGCA-CPT
```

训练数据：

```text
data/finetune/csc/csc_mix_dedup.jsonl
```

验证数据：

```text
data/finetune/csc/cscd_dev.jsonl
```

训练目标：

给定包含拼写错误的文本，直接生成纠正后的文本。

默认参数：

```text
Max Length    = 768
Learning Rate = 1e-4
Epochs        = 5
```

执行：

```bash
MODEL=/path/to/PGCA-CPT-INS/merged \
MODEL_NAME=Qwen3-Base-PGCA-CPT-INS \
bash scripts/train/finetune/csc.sh
```

输出：

```text
PGCA-CPT-INS-CSC/
```

### 5.3 中文语法纠错（CGEC）

数据：

```text
data/finetune/cgec/train.jsonl
data/finetune/cgec/validation.jsonl
```

目标：

生成语法纠正后的文本。

默认参数：

```text
Max Length    = 768
Learning Rate = 1e-4
Epochs        = 5
```

执行：

```bash
MODEL=/path/to/PGCA-CPT-INS/merged \
MODEL_NAME=Qwen3-Base-PGCA-CPT-INS \
bash scripts/train/finetune/cgec.sh
```

输出：

```text
PGCA-CPT-INS-CGEC/
```

### 5.4 古汉语多任务微调（ACT）

训练数据：

```text
data/finetune/ancient/
├── train.jsonl
├── train.metadata.jsonl
├── dev.jsonl
└── dev.metadata.jsonl
```

训练包括三个任务：

1. 古文 → 现代汉语（c2m）
2. 现代汉语 → 古文（m2c）
3. 古文标点恢复（punctuate）

脚本会根据 metadata 构建带 `channel` 的训练视图，并校验数据与 metadata 的对应关系。

默认使用 `cc_llm_ancient` 模板，保留 Qwen3 mixed ChatML 的非思考格式。

默认参数：

```text
Max Length    = 2048
Learning Rate = 1e-4
Epochs        = 5
Batch/GPU     = 2
Grad Accum    = 4
```

实际有效 Batch 取决于 GPU 数。例如四卡时：

\[
B_{\mathrm{effective}}=4\times2\times4=32
\]

checkpoint 按三个任务的验证 Loss 宏平均选取：

\[
L_{\mathrm{macro}}
=
\frac{
L_{\mathrm{c2m}}+
L_{\mathrm{m2c}}+
L_{\mathrm{punctuate}}
}{3}
\]

执行：

```bash
MODEL=/path/to/PGCA-CPT-INS/merged \
MODEL_NAME=Qwen3-Base-PGCA-CPT-INS \
bash scripts/train/finetune/ancient.sh
```

输出：

```text
PGCA-CPT-INS-ACT/
```

### 5.5 下游训练的生成式验收

三个下游 SFT 任务已接入固定 Dev 自由生成检查：每次保存 checkpoint 时执行轻量检查，正式评估前对选中模型执行完整验收。

这是此前 ACT 实验暴露出来的重要问题：验证 Loss 正常下降，并不保证模型能够正常终止生成。

至少需要检查：

- 是否能够在合理长度内结束回答；
- 是否正确生成 `<|im_end|>`；
- 是否频繁触及 `max_new_tokens`；
- 是否存在重复输出与异常字符；
- Adapter 是否加载并生效；
- PGCA Cache 与非 Cache 路径是否存在显著生成差异。

保存时默认每任务 1 条，结果写入 `checkpoint-N/dev_check/vN/`；失败只记录，不中断训练，也不改变原有 loss 选模。

下游评估入口 `STAGE=all|infer` 自动重新加载完整模型和已保存 adapter，每任务检查前 10 条（ACT 三任务共 30 条）。各任务正常停止率及 `<|im_end|>` 结束率均须至少 90%，截断、重复、异常率各不超过 10%；EOS 监督、模板、LoRA 生效或 Cache 检查失败也会阻止正式推理。完整结果写入本轮评估目录的 `dev_check/vN/`。阈值、样本数、生成上限和 Cache 容差均可配置，见 [固定 Dev 生成验收](scripts/train/finetune/README.md#固定-dev-生成验收)。

Cache 对照使用真实单序列、无 padding 的 HF 前向，比较同一前缀的完整计算与逐 token 计算，并核对 PGCA 主干及音形缓存长度。Cache 两路自由生成不强求逐字相同，但数值须在容差内，停止、截断、重复、异常标记须一致。LoRA 对照检查 logits 变化和两路生成。该检查不代替 vLLM/批量推理验收，也不判断参考答案是否正确。

若训练内任务仍系统性不终止，应依据报告区分训练、模板和推理问题，再启动正式测试。集中冒烟使用报告模式，不以少步训练的生成质量阻止连通性检查；结构、LoRA 或 Cache 失败仍会中止对应步骤。

---

## 6. 全面评测

主要源码：

```text
scripts/evaluation/
├── run.sh
├── ppl/
├── lmeh/
├── csc/
├── cgec/
└── ancient/
```

### 6.1 语言建模能力（PPL）

数据：

```text
data/evaluation/ppl/*.jsonl
```

包含六类文本：

- 政务
- 影视
- 通用
- 游戏
- 科技
- 金融

评估使用原始文本，不添加 Chat Template，不人为拼接 BOS/EOS，并采用文档独立滑动窗口计算 NLL。

默认窗口长度 2048，Stride 为 1024。

执行：

```bash
MODEL=/path/to/checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
bash scripts/evaluation/ppl/run.sh
```

主要指标：

\[
\operatorname{PPL}=\exp(\operatorname{NLL}_{token})
\]

同时报告 BPC、BPB。

由于 Base 与 Char/PGCA 的 Tokenizer 不同，Token PPL 不宜直接作为跨模型的唯一比较依据。

跨 Tokenizer 更适合报告归一化到原始字符或 UTF-8 字节的语言建模损失。

### 6.2 通用语言理解与推理

使用 lm-evaluation-harness。所有通用主结果使用 General-SFT 后的 INS 模型，默认 `EVAL_MODE=instruct`，从保存的 Swift 非思考模板恢复实际 Prompt；few-shot 固定为多轮 user/assistant 消息。原始 Base/CPT 对照须显式设置 `EVAL_MODE=base`，不可混用协议。

覆盖：

| 基准 | 主要能力 |
|---|---|
| MMLU | 多学科知识 |
| ARC-Challenge | 科学常识与推理 |
| GSM8K | 数学推理 |
| SQuAD v2 | 阅读理解 |
| CMMLU | 中文多学科知识 |
| CLUE | 中文语言理解 |

当前协议：

- MMLU：5-shot
- CMMLU：5-shot
- GSM8K：5-shot
- 其他：0-shot
- ARC：Challenge 子集

执行：

```bash
MODEL=/path/to/checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS \
BENCHMARK=all \
bash scripts/evaluation/lmeh/run.sh
```

也可选择单个任务：

```bash
BENCHMARK=mmlu
BENCHMARK=arc
BENCHMARK=squadv2
BENCHMARK=gsm8k
BENCHMARK=cmmlu
BENCHMARK=clue
```

CLUE 在论文表中按照源码定义的任务组汇总，不能直接称为官方 CLUE 总分。

### 6.3 CSC 正式评估（完整原始输出为主）

评估数据：

```text
CSCD-NS
LEMON
```

使用对应模型训练得到的 CSC Adapter。

```bash
MODEL=/path/to/PGCA-CPT-INS/merged \
LORA=/path/to/PGCA-CPT-INS-CSC/adapter-checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS-CSC \
bash scripts/evaluation/csc/run.sh
```

主要报告：

- Detection Precision / Recall / F1
- Correction Precision / Recall / F1
- 句子级纠正指标
- LEMON 各领域及宏平均结果

LEMON 需要按照七个领域分别计算，再按既定规则汇总。

### 6.4 CGEC 正式评估

主要基准：

```text
FCGEC
NaCGEC
```

执行 FCGEC Valid：

```bash
MODEL=/path/to/PGCA-CPT-INS/merged \
LORA=/path/to/PGCA-CPT-INS-CGEC/adapter-checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS-CGEC \
DATASET=fcgec SPLIT=valid \
bash scripts/evaluation/cgec/run.sh
```

执行 NaCGEC：

```bash
MODEL=/path/to/PGCA-CPT-INS/merged \
LORA=/path/to/PGCA-CPT-INS-CGEC/adapter-checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS-CGEC \
DATASET=nacgec SPLIT=dev \
bash scripts/evaluation/cgec/run.sh
```

主要指标：

- 编辑级 Precision
- 编辑级 Recall
- F0.5（FCGEC 主指标；NaCGEC 双评分器算术平均）
- 编辑级 F1（补充）
- NaCGEC 对应评分协议

FCGEC 默认 Test 仅导出提交。NaCGEC 默认 Dev 使用 `nacgec.dev.ref.para` 评分，Test 使用 `nacgec.test.input` 导出。NaCGEC 官方已公开 Test gold；本地 Test 评分需显式传 `REFERENCE_FILE`，不能再描述为“官方没有公开参考”。[官方说明](https://github.com/masr2000/NaCGEC)

### 6.5 古汉语能力评估

主要基准：

```text
C³Bench
Fùxì
```

C³Bench 覆盖：

1. 古文类别分类
2. 典籍出处检索
3. 命名实体识别
4. 标点恢复
5. 古文翻译

Fùxì 包含多种古汉语知识、理解和生成任务。

执行 C³Bench：

```bash
MODEL=/path/to/PGCA-CPT-INS/merged \
LORA=/path/to/PGCA-CPT-INS-ACT/adapter-checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS-ACT \
DATASET=c3 \
bash scripts/evaluation/ancient/run.sh
```

执行 Fùxì：

```bash
MODEL=/path/to/PGCA-CPT-INS/merged \
LORA=/path/to/PGCA-CPT-INS-ACT/adapter-checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS-ACT \
DATASET=fuxi \
bash scripts/evaluation/ancient/run.sh
```

Fùxì 部分任务需要额外的模型裁判。

未配置 `JUDGE_MODEL` 和 `JUDGE_BASE_URL` 时，这些任务应明确记录为 pending，而不是零分。

还需要区分：

- ACT 直接监督的翻译、标点任务；
- 没有直接监督的分类、出处和 NER 等迁移任务。

两者反映不同的泛化问题，不应简单合并解释。

### 6.6 三条路线的统一评估

建议分别执行：

#### Base

```bash
MODEL_NAME=Qwen3-Base-INS \
MODEL=/path/to/Base-INS/merged \
bash scripts/evaluation/run.sh
```

#### Char-CPT-INS

```bash
MODEL_NAME=Qwen3-1.7B-Base-Char-CPT-INS \
bash scripts/evaluation/run.sh
```

#### PGCA-CPT-INS

```bash
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS \
bash scripts/evaluation/run.sh
```

默认评估范围包含 PPL、六类通用 Benchmark、CSC、CGEC、C³Bench 和 Fùxì。

如果只希望先完成通用能力评估：

```bash
BENCHMARKS=ppl,lmeh \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS \
bash scripts/evaluation/run.sh
```

如果只评估下游任务：

```bash
BENCHMARKS=csc,fcgec_valid,nacgec,ancient_c3 \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-INS \
bash scripts/evaluation/run.sh
```

评估器默认：

- 标准 Base/Char：优先使用 vLLM；
- 当前 PGCA：使用 HF 后端。

如需跨后端比较吞吐或数值，应先完成一致性验收。

---

## 7. PGCA 机制实验与消融设计

除了最终任务性能，还需要解释 PGCA 为什么有效。

以下内容属于完整研究实验方案，其中部分诊断已在此前实验版本执行，但**当前 master 没有包含全部 P0/Uniform 独立诊断脚本**。

因此，不应将这些项目写成可直接通过 master 统一入口运行的命令。

### 7.1 Char-only vs PGCA-CPT

比较：

```text
Char-CPT
PGCA-CPT
```

使用相同原始语料与尽量匹配的 Token 预算。

目的：

评价引入音形分支后，相对于字符级语义模型是否获得额外收益。

这是最终架构比较的重要实验。

需要注意，两者的训练阶段和新增参数不同，不能单纯把差距归因于某一个 PGCA 组件。

### 7.2 PGCA On/Off

固定同一个已训练的 PGCA checkpoint。

比较：

```text
PGCA On
PGCA Off
```

其他模型权重不变，仅在前向过程中关闭 PGCA 分支。

记录：

- CLM NLL
- Token Accuracy
- 逐样本 NLL
- 配对差值与置信区间

该实验衡量已训练模型对 PGCA 的功能性依赖，不等价于独立训练的 Char-only 基线。

### 7.3 Normal vs Uniform Attention

固定 PGCA checkpoint。

比较：

```text
Normal PGCA
Uniform PGCA
PGCA Off
```

Uniform 将实际学习到的 Attention 权重替换为对所有有效音形槽的均匀分配，但保留 Value、Output Projection 与上下文编码器。

该实验回答：

**PGCA 的收益是否不仅来自音形特征注入，还依赖模型学习到的 Attention 分配？**

需要明确，Uniform 同时改变模态权重和候选内部权重，不能单独证明多音字消歧机制有效。

### 7.4 多音字候选分析

构建同字不同语境的配对测试：

```text
银行：行 / hang2
运行：行 / xing2
```

分析：

- 正确读音候选 Top-1；
- 候选 Attention 熵；
- 不同语境的候选切换；
- 拼音与字形模态权重；
- 不同 PGCA 层的差异。

需要将该实验与正式语言建模评价区分。

当前合法读音集合辅助任务，并不直接提供唯一语境读音的监督。

### 7.5 梯度与分支贡献分析

分析：

\[
r_l=
\frac{\|\mathrm{PGCA}_l\|}
{\|\mathrm{SelfAttn}_l\|+\epsilon}
\]

并分别统计：

- PGCA Q/K/V/O 梯度范数；
- Feature Embedding 梯度；
- Context Encoder 梯度；
- CLM 与字音/字形辅助梯度余弦；
- 不同层的 Attention 模态分布。

该实验用于判断：

1. PGCA 是否真正参与优化；
2. 主干是否忽略音形分支；
3. 辅助任务是否与 CLM 形成梯度竞争；
4. 不同层的 PGCA 作用是否一致。

### 7.6 更严格的结构消融

如论文需要单独证明各组件作用，还应增加独立训练对照，例如：

```text
Char-only
Static-PGCA
Context-PGCA
Context-PGCA without auxiliary loss
Full CC-LLM
```

这些属于建议补充的研究对照，并非当前 master 已经提供的完整自动化训练矩阵。

需要通过新的配置和独立训练运行实现，不能仅靠推理时修改配置，把训练后干预结果当作结构训练消融。

---

## 8. 实验记录、验收与论文取数

### 8.1 必须固定的实验身份

每组实验应至少记录：

```text
Git Commit
Base Model
Tokenizer
Vocabulary / Feature Index
PGCA Layers
PGCA Context Config
Auxiliary Loss Weights
Dataset Version
Actual Token Budget
Optimizer / Scheduler
Batch Size
Seed
Selected Checkpoint
Evaluation Protocol
```

尤其需要固定：

```text
PGCA 层号
词表哈希
特殊 Token 映射
数据版本
有效训练 Token 数
```

不能仅根据模型目录名推断结构完全一致。

### 8.2 训练验收

CPT：

- 模型与 Tokenizer 能完整加载；
- 验证 CLM Loss 有限且可解释；
- Phase2-A 的分支参数确实更新；
- Phase2-B 的主干和分支均参与训练；
- 辅助任务监督覆盖正常；
- checkpoint 选优依据明确。

SFT：

- 训练 Adapter 与基础模型对应；
- `finetune_report.json` 中参数范围符合预期；
- assistant 答案结束符受到有效监督；
- 固定短 Dev 可以正常自由生成；
- 终止率、重复率和长度上限命中率满足预先制定的验收条件。

正式评估：

- 输入数据、Few-shot 协议和评分版本固定；
- 模型和 Adapter 与结果一一对应；
- 预测完整且没有未处理推理错误；
- 不把缺失指标填成零；
- 不把验证 Loss 当作最终任务性能。

### 8.3 论文实验表建议

#### Table 1：模型与预训练配置

```text
Base
Char-CPT
PGCA-CPT
```

记录参数量、Tokenizer、CPT Token、PGCA 层数及训练资源。

#### Table 2：通用能力

```text
PPL / BPC / BPB
MMLU
ARC-Challenge
GSM8K
SQuAD v2
CMMLU
CLUE
```

#### Table 3：中文纠错能力

```text
CSCD-NS
LEMON
FCGEC
NaCGEC
```

#### Table 4：古汉语能力

```text
C³Bench
Fùxì
```

区分直接微调任务和未直接监督任务。

#### Table 5：PGCA 机制分析

```text
PGCA On / Off
Normal / Uniform
Attention Mass
Polyphone Selection
Gradient Alignment
```

需标明诊断所用 checkpoint、训练轮次及样本数。

#### Table 6：训练阶段表现

```text
Phase1
Phase2-A
Phase2-B
Char-CPT
```

横轴优先使用有效训练 Token 数，同时保留 optimizer step。

---

## 9. 完整实验执行顺序

推荐按以下顺序实际组织。

### Step 1：准备环境与数据

```bash
bash scripts/environment/run.sh
```

准备完整 Base 模型、汉字资源、预训练数据、下游数据和测试集。

解压预训练 `.tar.xz`。

冻结数据清单和版本。

### Step 2：构建 Char 与 PGCA

```bash
BASE_MODEL=/absolute/path/to/Qwen3-1.7B-Base \
BASE_MODEL_NAME=Qwen3-1.7B-Base \
bash scripts/model_build/run.sh
```

完成后验证两个模型：

```text
Qwen3-1.7B-Base-Char
Qwen3-1.7B-Base-PGCA
```

并确认 PGCA 层号是否符合本次实验方案。

### Step 3：先做集中冒烟

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/smoke/run.sh \
/absolute/path/to/Qwen3-1.7B-Base
```

冒烟覆盖三条模型路线的少步训练与少量评测。

结果写入：

```text
outputs/smoke/vN/summary.md
```

冒烟仅检验链路连通性，不代表正式模型性能、长期训练稳定性或 EOS 学习已通过验收。

### Step 4：执行三组正式训练

```text
Base  → General-SFT/合并 → 下游专项 SFT
Char  → Phase1 → Char-CPT → General-SFT/合并 → 下游专项 SFT
PGCA  → Phase1 → Phase2-A → Phase2-B → General-SFT/合并 → 下游专项 SFT
```

使用各自的 `MODEL_VARIANT`、`MODEL_NAME` 和输入模型路径。

正式实验不要依赖可能被其他构建覆盖的全局模型指针。

### Step 5：固定正式 checkpoint

分别保存：

```text
Base 原始 checkpoint
Char-CPT 最终选中 checkpoint
PGCA-CPT 最终选中 checkpoint
```

同时保存各自的 Tokenizer、配置和训练参数。

检查是否存在不同运行之间的模型路径混用。

### Step 6：执行三类下游 LoRA

每个基础模型独立训练：

```text
CSC
CGEC
Ancient
```

共形成九组模型—任务组合。

每个训练结果保存独立 Adapter，记录实际训练参数和选优依据。

### Step 7：先验收自由生成

对 CSC、CGEC 和 Ancient 的短 Dev 子集进行生成检查。

特别检查：

```text
EOS
异常 Token
重复输出
长度上限
Adapter 加载
KV Cache
```

确认结果可用后，再执行正式测试集评估。

### Step 8：执行全部正式评测

分别运行：

```bash
MODEL_NAME=Qwen3-Base-INS \
MODEL=/absolute/path/to/Base-INS/merged \
bash scripts/evaluation/run.sh
```

```bash
MODEL_NAME=Qwen3-Base-Char-CPT-INS \
bash scripts/evaluation/run.sh
```

```bash
MODEL_NAME=Qwen3-Base-PGCA-CPT-INS \
bash scripts/evaluation/run.sh
```

结果汇总自各评估目录中的 `metrics.json` 与 `table_metrics.json`。

### Step 9：执行 PGCA 机制实验

固定模型与数据，分别进行：

```text
On / Off
Normal / Uniform
Multi-pronunciation Analysis
Gradient Alignment
```

若增加独立架构消融，则需要单独训练对应模型。

### Step 10：归档与论文汇报

统一保存：

```text
experiment_manifest
model_config
train_args
training_logs
checkpoint_selection
evaluation_protocol
prediction_results
table_metrics
mechanism_probe_results
```

最终按照：

```text
模型配置
    ↓
预训练效果
    ↓
下游主结果
    ↓
架构及训练消融
    ↓
机制分析
    ↓
局限性讨论
```

组织 ACL 论文的 Experiment 章节。

---

## 10. 当前源码与历史实验的关键边界

本流程描述的是 `Nephamaster/CC-LLM` 的 `master` 分支，而不是旧版 `main` 的目录结构。

需要特别注意：

1. `master` 采用 `scripts/model_build`、`scripts/train/cpt`、`scripts/train/finetune` 路径，不能使用旧的 `scripts/train/ms_swift/cpt` 路径。
2. PGCA 插入层默认按模型中间三分之一自动选择，不保证等于历史的第 11–16 层。
3. Phase2-B 当前默认 Warm-up 是 3%，不能直接认定与此前实际训练采用的 1% 配置相同。
4. 当前仓库不包含原始数据及训练权重，也不完整包含历史 Data Factory 的源数据生产过程。
5. P0 和 Uniform 等旧版机制诊断需要独立保留对应实验实现，不能用新的主训练入口自动代替。
6. Base、Char、PGCA 的模型比较应固定数据、评估协议、Adapter 训练方式及 checkpoint 身份。
7. 正式论文中的结果必须来自实际运行日志，不能将脚本默认值或计划训练预算填写为已经完成的实测值。

本实验最核心的最终比较是：

**在字符级语义表示、继续预训练数据和下游任务协议尽可能匹配的条件下，PGCA 的上下文化音形信息是否能够提供稳定、可复现的额外增益。**

完整研究结论应同时依赖跨模型任务成绩与同 checkpoint 的机制证据，而不是仅根据语言建模 Loss 或某一次推理干预得出。
