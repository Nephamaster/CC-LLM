# CC-LLM 第二轮实施方案

已经确认：第二轮从未训练的初始 PGCA 模型（包含 Char 词表）重新执行 Phase1 → Phase2-A → Phase2-B，不复用第一轮训练权重。保留现有并行残差拓扑：

$$S_l=H_l+\operatorname{SelfAttn}(\operatorname{Norm}(H_l)) +\operatorname{PGCA}(\operatorname{Norm}(H_l),F\_{\mathrm{ctx}})$$

之后执行常规的 `Post-RMSNorm → MLP → 残差相加`。不恢复可学习 gate，不增加 Transformer 层数。

## 0. 最新日志核查与第二轮超参数

下表保留方案制定时的历史日志：截至 step 10,430，最近完整验证为 step 10,000；当前第一轮停在用于诊断的 checkpoint-11500，后续 P0＋Uniform 结论见下文，不以此表作为当前进度。

| Phase2-B 验证指标 | Step 0  | Step 5,000 | Step 10,000 |
| ------------- | ------- | ---------- | ----------- |
| CLM Loss      | 1.90105 | 1.87231    | 1.86061     |
| Total Loss    | 2.09354 | 2.05680    | 2.04242     |
| 字音准确率         | 14.21%  | 17.67%     | 19.08%      |
| 结构属性准确率       | 47.01%  | 48.29%     | 48.86%      |

这说明第一轮 B 阶段持续收敛，但后期改善速度正在减缓。没有证据要求进一步降低学习率。

考虑到第二轮计划增大有效 Batch，我建议保留主干与分支的峰值 LR，缩短 B 的 warm-up。这也修正了此前暂定的 `8e-6/2e-5` 建议：较大的 Batch 已减少同等 token 预算下的优化步数，不应在没有发散证据时再同时减少更新幅度。

| 参数                    | Phase1  | Phase2-A | Phase2-B |
| --------------------- | ------- | -------- | -------- |
| Per-device batch，8 卡  | 1       | 1        | 1        |
| Gradient accumulation | 32      | 8        | 32       |
| 估算有效 tokens/step      | 约 0.32M | 约 0.10M  | 约 0.39M  |
| 主干 LR                 | `1e-5`  | 冻结       | `1e-5`   |
| 分支 LR                 | 冻结      | `5e-5`   | `3e-5`   |
| LR warm-up            | 3%      | 3%       | 1%       |
| Weight decay          | 0.01    | 0.01     | 0.01     |
| Gradient clipping     | 1.0     | 1.0      | 1.0      |

这里有意让 Phase2-A 保留较小 Batch：它只有 100M tokens，如果也使用 accumulation=32，优化步数会从约 1,027 降到约 260，可能不足以完成新分支适配。Phase2-B 才使用约 0.4M 的有效 Token Batch。

这些 Token 数是按第一轮平均监督密度估算，实际值必须由第二轮 `preparation_report.json` 和训练日志确认。

## 1. 工作包总览

以下三个工作包的总体顺序不变。P0（含 Uniform 补充实验）已完成；P1 最小代码及本地验证已完成，下一步准备独立初始模型并做服务器实卡验收。P2 收敛为保持现有架构及稳定性观测，不再要求开训前实现新的融合机制。

| 优先级 | 工作包            | 主要目的                          | 预计计算开销    |
| --- | -------------- | ----------------------------- | --------- |
| P0  | PGCA 机制诊断与记录系统 | 确认分支及学习到的 Attention 分配有作用          | 已完成 P0＋Uniform |
| P1  | 辅助任务改进         | 降低字音、字形任务失衡，并验证其梯度是否有效进入 PGCA | 少量梯度测试    |
| P2  | PGCA 架构保持与观测  | 保留学习型 Attention，观察模态份额和分支幅度      | 现有结构回归测试 |

执行和记录基准统一使用最新 [CC-LLM 仓库](https://github.com/Nephamaster/CC-LLM)。

## P0｜PGCA 机制诊断与实验记录

已完成；以下保留原诊断协议，供第二轮固定点复测。

目标：用第一轮现有 checkpoint 判断 PGCA 有没有贡献。只做一次诊断，不重新训练。

### 已确认结果

见 [P0 分析](refactor-p0-ob.md) 和 [Uniform 分析](refactor-p0-uniform.md)。在相同 314 条、540,315 个有效目标上，Normal / Uniform / Off 的 NLL 为 **1.97647 / 2.05444 / 2.23860**；Uniform−Normal 为 **0.07796，95% CI [0.06684, 0.09222]**。因此保留学习型 Cross-Attention，不采用 Uniform 作为训练或默认推理方式。

这证明当前 checkpoint 对分支及学习到的 Attention 分配有功能性依赖；尚不能归因于精确多音字消歧，也不是与等预算 Char-only 模型的最终优劣比较。Uniform 同时改变 head 偏好、模态份额和候选内分配；上下文编码器的独立收益仍未分离。

### 诊断协议（已执行）

1. 使用 `scripts/diagnostics/pgca_probe.py`，加载第一轮 Phase2-B 的 `checkpoint-11500` 对应 `for_exp` 副本，固定 `eval()`、`use_cache=False`、相同输入和标签。
2. 从本轮实际使用的 `phase2_4096/val` 固定抽取 512 条编码后的序列，保留原始 labels，分别计算 PGCA 开启／关闭时的 CLM NLL、Token Accuracy 和辅助任务准确率。
3. 在 0-based 第 11–16 层注册 forward hook，收集 Self-Attention、PGCA 输出以及 PGCA slot attention。额外准备 100 条有明确左侧语境的多音字候选样本（50 个字、每字两种读音），由用户审核后才纳入正式消歧统计；核对真实读音候选与 attention 排名。
4. 使用一个固定训练 batch，分别对 CLM 和两个辅助损失求梯度，不执行 optimizer step，检查各模块的梯度强度及方向。

### 必须记录的指标

| 指标                           | 记录内容                                                                     |
| ---------------------------- | ------------------------------------------------------------------------ |
| `delta_clm_nll`              | $L_{\mathrm{off}}-L_{\mathrm{on}}$，大于 0 表示 PGCA 对当前 checkpoint 有帮助 |
| `pgca_sa_norm_ratio`         | 每层$\\|F_l\\|/\\|A_l\\|$的 P50、P95                               |
| `pgca_residual_cosine`       | PGCA 输出与主干 Attention 输出的余弦相似度                                            |
| `phonetic_mass / glyph_mass` | 两种模态的 Attention 总权重                                                      |
| `polyphone_top1`             | 正确读音在实际候选中的 Top-1 准确率                                                    |
| `gradient_norm`              | Feature Embedding、Context Encoder、PGCA Q/K/V/O、辅助头各自梯度范数                 |
| `gradient_cosine`            | CLM 与辅助损失对 PGCA 共享参数的梯度夹角                                                |

特别注意：只有一个有效拼音候选的汉字，不能用于检验多音字消歧。某个模块没有梯度时，记录零范数和 `cosine=null`，不能计算虚假的夹角。

### 诊断结果如何决定后续工作

| P0 结果                       | 处理方向                                   |
| --------------------------- | -------------------------------------- |
| PGCA 开启后 NLL 更低             | 保留分支，优先调整辅助任务权重                        |
| PGCA 开启后 NLL 更高             | 优先检查候选融合偏置、分支幅度，再启动第二轮                 |
| On/Off 几乎无差异                | 优先检查 PGCA 梯度及 Auxiliary Heads 是否主要依赖主干 |
| Attention 随语境变化明显，但 NLL 无收益 | 不再增加上下文层数，优先调整信息融合方式                   |

这些都是筛查规则，而不是统计显著性结论。512 条样本应报告配对差异和 bootstrap 置信区间，不能仅比较两个平均数的大小。

验收： 固定输入上重复运行得到一致结果；关闭 PGCA 不改变模型权重；输出完整的 `probe_summary.json` 和 `probe_samples.jsonl`。

实施入口及精确统计口径见 [P0 诊断说明](../../scripts/diagnostics/README.md) 和 [Uniform 对照说明](../../scripts/diagnostics/UNIFORM_ATTENTION.md)。工具独立于训练，使用 checkpoint 配套代码；`scripts/diagnostics/polyphones_review.tsv` 的 100 条样本已完成审核。梯度使用固定 `train_b[0]`，沿用 CLM token 分母和三项字形 CE 平均，不执行参数更新。P0 阶段未修改 P1/P2 或在训练进程中插入日志。

## P1｜辅助任务改进

目标：平衡字音与字形两项辅助损失的相对贡献，同时保留 Phase2-A 对新分支的训练压力；不把当前结果解释为辅助梯度整体压倒 CLM。

### 实施步骤

第一步：保留当前三任务定义，不重新设计预测头。

目前的下一汉字合法拼音集合损失、三个字形属性 CE、无效标签 mask 和 next-token shift 没有发现需要立即推翻的问题。第一版不增加 G2P 模型或新的辅助数据集。

第二步：调整两个损失权重。

将模型配置修改为：

$$\boxed{ L\_{\mathrm{total}} =L\_{\mathrm{CLM}} +\alpha(t) \left(0.03L\_{\mathrm{phon}}+0.03L\_{\mathrm{glyph}}\right) }$$

其中：

$$L\_{\mathrm{glyph}} =\frac{L\_{\mathrm{stroke}}+L\_{\mathrm{radical}}+L\_{\mathrm{structure}}}{3}.$$

按第一轮 Phase2-A 末尾的损失量级估算，原本约 80:20 的字音／字形加权贡献会调整为约 61:39，整体辅助损失占 CLM 的比例也会从约 10% 降至约 8%。

这是基于第一轮实际损失的保守工程选择，P0＋Uniform 没有证明 `0.03/0.03` 最优，也没有证明它能解决语境读音选择。Phase2-A 的 10M-token 辅助权重 warm-up 保持不变；各训练 loss 继续使用 CLM 有效目标分母，不更换辅助标签或分类头。

第三步：调整配置入口与记录。

修改：

- `src/configuration_qwen3_pgca.py`：设置新的默认辅助权重；
- `src/pgca/build_pgca.py`：增加明确的权重构建参数，并写入最终 config；
- `scripts/train/ms_swift/cpt/plugin.py`：保留原计算公式，新增每项加权损失比例以及分模块梯度记录。

建议在 Phase2-B 持续记录 `weighted_phon_to_clm`、`weighted_glyph_to_clm`，而不只记录两者之和。

必须将 `0.03/0.03` 显式写入第二轮初始模型的 config；只改 Python 默认值不会更新已有模型配置。第一轮权重与配置保持原样。P1 已实现新构建默认 `0.03/0.03` 及 `--aux-phonetic-weight` / `--aux-glyph-weight` 参数，并补充分项比例与分模块联合梯度日志；本地回归通过，但尚未准备真实第二轮模型或完成服务器实卡验收。

第四步：保留 Phase2-A 的冻结策略。

Phase2-A 冻结主干，使 CLM 和辅助任务的梯度都需要通过新增分支适配；Phase2-B 再全部解冻。P0 已确认 CLM 和辅助梯度均能进入 PGCA，但不代表辅助头的全部效果来自分支。保留接在最终 Hidden States 上的 Auxiliary Heads，不新增“直接从 PGCA 输出预测下一字”的分类头或上下文读音监督任务。

### 最少测试与记录

只需要三个测试情形：正常汉字 batch、全英文 batch、存在未知音形属性的 batch。

| 项目                | 验收与记录                           |
| ----------------- | ------------------------------- |
| `loss_components` | 逐项 Loss、加权比例、有效监督 Token 数       |
| `aux_accuracy`    | 拼音、笔画、部首笔画、结构各自准确率              |
| `gradient_cosine` | CLM 与辅助目标对 PGCA 的梯度方向           |
| 数值正确性             | 全英文 batch 辅助损失为 0；所有 Loss 与梯度有限 |
| 分布式一致性            | 与单卡参考计算在相同有效 Token 数下匹配         |

验收： 不修改已有辅助目标标签；上述测试通过，A/B 阶段均能正确计算并记录损失；不出现辅助分母为零导致的 NaN。

## P2｜PGCA 架构保持与观测

目标：保留已验证有效的 Normal PGCA，继续观察模态份额和残差幅度。第二轮不同时改 Softmax、Attention Head 结构或残差缩放。

### 实施步骤

第一步：保留原 Attention；候选数量校正暂缓，不列为本轮必做。

当前 PGCA 在所有有效拼音候选和字形 slot 上一起 softmax。以下数量校正只保留为后续备选研究，不要求本轮修改 `src/pgca/attention.py`：

$$s'\_{p}=s_p-\log\big(\max(1,N_p)\big), \qquad s'\_g=s_g$$

其中 \\(s_p,s_g\\) 分别是拼音与字形 slot 的注意力 logit，\\(N_p\\) 是该 token 的有效拼音候选数。

在两种模态均有效、所有 slot 原始分数相等时，校正后拼音模态与字形模态各获得 50% 总权重。它不改变单个 head 内有效拼音候选之间的相对排序，不能直接提供语境读音监督。

第二轮不新增或启用 `pgca_balance_modalities`。现有拼音偏好尚未被证明有害，Normal 优于 Uniform 也不能证明更均衡的模态配比更好；将来只有独立控制实验支持时再讨论启用。

第二步：保留分支残差幅度观测。

优先复用 P0 的离线 hook，对实际参与相加的 PGCA 输出计算范数；不为此改写 decoder forward 或增加模型持久化状态。

记录：

$$r_l=\frac{\\|F_l\\|\_2}{\\|A_l\\|\_2+\epsilon}.$$

现有 On/Off 和 Uniform 对照均支持保留 Normal，不因第 16 层范数比偏高就将 `pgca_residual_scale` 设为 0.5。第二轮保持直接相加（等效 scale=1），不新增可学习 gate 或缩放逻辑。

第三步：保留当前 4 层上下文编码器。

不新增独立的音形 Transformer、不扩大 hidden size，也不调整六个 PGCA 插入位置。第二轮首先要判断现有结构是否产生增益，而不是再次改变模型容量。

### 最少测试与记录

| 测试            | 验收要求                     |
| ------------- | ------------------------ |
| 1／4／16 个拼音候选  | 沿用原有效 slot mask 和学习型 softmax，不强制模态各 0.5 |
| 无拼音、只有字形      | 字形权重为 1                  |
| 非汉字、全无效 slots | PGCA 输出严格为 0             |
| 因果前缀不变性       | 改变后缀不影响前缀输出              |
| BF16 与反向传播    | 输出和梯度有限，PGCA 正常更新        |

验收： 保留原 `state_dict` 结构、没有新 gate 或缩放参数；相同权重与输入下，Normal 前向与现有 PGCA 在数值容差内一致。

可选补充：如需进一步区分模态份额与候选内选择，可按 [Uniform 分析](refactor-p0-uniform.md) 做一次 Modality-Preserving Uniform，仅在每个 head 内均分有效拼音候选，保留原拼音总权重与字形权重。它不作为第二轮开训前硬性阻塞项，本方案不追加其他必做诊断。

## 2. 第二轮启动与实验记录要求

这里需要根据最新仓库再纠正一个路径变化：当前已拆成三个独立脚本，旧的 `phase2.sh` 和 `common.sh` 不再存在。

正确入口为：

```
scripts/train/ms_swift/cpt/
├── phase1.sh
├── phase2_a.sh
├── phase2_b.sh
├── plugin.py
└── prepare_data.py
```

以下是第二轮参数覆盖示例，前提是模型构建和 P0–P2 必要验收已经完成：

```
# Phase1
WEIGHT_DECAY=0.01 GRAD_ACCUMULATION=32 \
bash scripts/train/ms_swift/cpt/phase1.sh

# Phase2-A
WEIGHT_DECAY=0.01 GRAD_ACCUMULATION=8 \
bash scripts/train/ms_swift/cpt/phase2_a.sh

# Phase2-B
WEIGHT_DECAY=0.01 \
GRAD_ACCUMULATION=32 \
WARMUP_RATIO=0.01 \
PHASE2_B_LR=1e-5 \
PHASE2_B_BRANCH_LR=3e-5 \
bash scripts/train/ms_swift/cpt/phase2_b.sh
```

这只是关键超参数示例，不是现在即可执行的完整第二轮命令。正式启动时使用独立初始 `MODEL` 与第二轮 `TRAIN_ROOT`／`OUT`，核对 A 的 `ALIGN_OUT`、B 的 `CPT_ROOT`／`ADAPTATION_OUT`，防止误读第一轮的 `next_stage_model.txt`。模型名称仍沿用 M-ALIGN、M-CPT，用轮次目录隔离，不额外堆叠模型后缀。

不改 tokenizer、词表和语料。现有 Char 构建产物可复用；原编码缓存只要通过 tokenizer、长度、seed、预算、world size 和每卡 batch 校验即可继续使用，不强制重导出。Phase2-A/B 共用相同数据、种子、Token 预算和缓存；只改变梯度累积、LR、WD 或辅助权重不要求重建缓存。

### 全程保留的最小实验档案

建议统一建立：

```
outputs/experiments/rnd2/
├── experiment_manifest.json
├── p0_pgca_probe/
│   ├── probe_manifest.json
│   ├── probe_summary.json
│   ├── probe_samples.jsonl
│   └── gradient_alignment.json
├── p0_uniform_pgca/
│   └── uniform_summary.json
├── p1_auxiliary/
│   └── loss_validation.json
├── p2_fusion/
│   └── fusion_validation.json
└── training/
    ├── Qwen3-1.7B-Base-PGCA-ALIGN/
    └── Qwen3-1.7B-Base-PGCA-CPT/
        ├── adaptation/
        └── joint/
```

`experiment_manifest.json` 固定记录以下信息，不允许运行后悄悄变更：

| 类别    | 必须记录的字段                                             |
| ----- | --------------------------------------------------- |
| 代码与模型 | 可取得的 Git commit、模型 config、初始权重来源、Tokenizer SHA256、音形索引指纹；不计算源码哈希，不以 Git 为运行前提 |
| 数据    | Phase1/Phase2 数据版本、验证集指纹、A/B Token 区间、随机种子          |
| 架构    | PGCA 层号、上下文编码器参数、Normal Attention、候选校正未启用、残差系数 1 |
| 损失    | 完整公式、权重、辅助 warm-up、标签 schema                        |
| 优化    | 各参数组 LR、实际有效 Token Batch、WD、scheduler、warm-up、梯度裁剪  |
| 结果    | 各阶段 checkpoint、训练 Token 数、验证 CLM／辅助指标、P0 诊断结果       |

训练中保留现有 loss 和 accuracy 日志，并补充分项加权比例及分模块联合梯度范数；三任务拆分梯度与余弦仍放在固定点离线诊断，不增加每个训练 step 的额外反向。每次保存 checkpoint 不必记录完整 Attention 张量，只在下述固定 Token 预算点保存分层聚合统计。

## 3. 最终执行顺序与停止条件

1. P0＋Uniform 已完成，保留原始报告与 checkpoint 副本；不重复跑整套诊断，不把可选补测列为必做。
2. P1 最小代码与本地验证已完成：新构建默认 `0.03/0.03`，mask、损失、梯度日志及普通 DDP 归一化通过回归；真实模型仍需下述实卡验收。
3. P2：保留现有 Normal PGCA，完成 mask、因果、BF16 和前向一致性回归；不实施候选校正或残差缩放。
4. 准备独立的第二轮初始 PGCA，显式写入新辅助权重；优先复用未训练初始权重以避免额外初始化差异，需重建时从同一 Char 产物出发并固定/记录初始化种子。不能拿第一轮 ALIGN/A/B 权重或诊断副本作为第二轮起点。核验 Schema、零初始化状态、辅助计数归零及阶段开关，完成小规模三阶段冒烟后冻结 manifest。
5. 按 Phase1 → A → B 完整训练，预算沿用 Phase1 原计划、Phase2-A 100M、Phase2-B 其余 9.9B 有效目标。以累计 Phase2（A+B）有效目标为统一进度，在 1B、3B 和最终点做固定验证及 PGCA 旁路评估；与第一轮比较时额外保留相同累计 Token 预算点，不按 step 号比较不同梯度累积的训练。
6. 训练后按固定口径做 PPL、通用 benchmark 与 CSC/CGEC/ACT 下游对照；先做已有入口支持的关键比较。等预算 Char-only 对照仍是证明架构净收益的独立需求，不把 P0 或当前早期 checkpoint 当作这项对照已经完成。

只设置两类必须满足的条件： 开训前，所有数值、mask、因果和参数冻结测试通过；训练中，固定验证集无持续异常、特征分支有实际梯度、checkpoint 与数据版本完整可追溯。若 P0 发现严重结构性错误，应先修复再开训；若只表现为收益较弱，则不扩大实验规模，按固定方案进入第二轮。

第一轮目前暂停在诊断 checkpoint：建议先保留原完整 checkpoint（含 optimizer、scheduler、trainer/RNG 状态）、当时脚本/插件和参数、日志、ALIGN/A 产物、for_exp 副本及两份诊断结果。当前累计 Phase2 约 1.219B 有效目标，尚非跑满 10B 的最终 CPT 模型；无需为了启动第二轮先补完第一轮。若以后需要续训，按第一轮原配置恢复（本轮为 8 卡、每卡 1、梯度累积 8、WD=0、LR warm-up=3%、辅助权重 0.05/0.02），不要混用第二轮参数，也不要仅用缺少训练状态的 for_exp 目录 resume。

整个方案将第二轮的主要变量控制为：更大的有效 Batch、更短的 B warm-up、WD=0.01 和辅助权重 0.03/0.03。保留学习型 Cross-Attention、并行无 gate 拓扑和 4 层因果编码器；候选数量校正、固定残差缩放和扩大编码器均不实施。几项训练设置仍同时变化，因此收益不能单独归因于其中某个超参数。
