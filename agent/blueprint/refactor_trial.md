# CC-LLM 第二轮实施方案

已经确认：第二轮从初始 Char-PGCA 模型重新执行 Phase1 → Phase2-A → Phase2-B，不复用第一轮训练权重。保留现有并行残差拓扑：

$$S_l=H_l+\operatorname{SelfAttn}(\operatorname{Norm}(H_l)) +\operatorname{PGCA}(\operatorname{Norm}(H_l),F\_{\mathrm{ctx}})$$

之后执行常规的 `Post-RMSNorm → MLP → 残差相加`。不恢复可学习 gate，不增加 Transformer 层数。

## 0. 最新日志核查与第二轮超参数

记录截止于 step 10,430，最后一次完整验证为 step 10,000。

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

以下将独立于训练启动的工作压缩为三个包。P0、P1、P2 都需要在第二轮正式开始前形成确定版本，但只有 P0 是必须先完成的诊断；P1、P2 的行为修改应受 P0 结果约束。

| 优先级 | 工作包            | 主要目的                          | 预计计算开销    |
| --- | -------------- | ----------------------------- | --------- |
| P0  | PGCA 机制诊断与记录系统 | 确认第一轮音形分支究竟帮助还是干扰主干           | 一次小规模离线评估 |
| P1  | 辅助任务改进         | 降低字音、字形任务失衡，并验证其梯度是否有效进入 PGCA | 少量梯度测试    |
| P2  | PGCA 融合结构改进    | 处理拼音候选数量偏置，监控并约束分支幅度          | 前向一致性测试   |

执行和记录基准统一使用最新 [CC-LLM 仓库](https://github.com/Nephamaster/CC-LLM)。

## P0｜PGCA 机制诊断与实验记录

最高优先级

目标：用第一轮现有 checkpoint 判断 PGCA 有没有贡献。只做一次诊断，不重新训练。

### 实施步骤

1. 新增 `scripts/diagnostics/pgca_probe.py`，加载第一轮 Phase2-B 的 `checkpoint-11500`（按本轮确认更新；上文 step 10,000 表格仍为历史日志），固定 `eval()`、`use_cache=False`、相同输入和标签。
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

实施入口及精确统计口径见 [P0 诊断说明](../../scripts/diagnostics/README.md)。该工具完全独立于正在运行的训练，使用 checkpoint 配套代码；100 条初稿位于 `scripts/diagnostics/polyphones_review.tsv`，`reviewed=false` 不进入正式准确率。梯度使用固定 `train_b[0]`（可显式选行），以 eval 模式分别对未加权三任务求导，沿用 CLM token 分母和三项字形 CE 平均，不执行参数更新。本轮不修改 P1/P2 或在训练进程中插入日志。

## P1｜辅助任务改进

目标：降低字音任务对联合损失的过度主导，同时保留 Phase2-A 对新分支的训练压力。

### 实施步骤

第一步：保留当前三任务定义，不重新设计预测头。

目前的下一汉字合法拼音集合损失、三个字形属性 CE、无效标签 mask 和 next-token shift 没有发现需要立即推翻的问题。第一版不增加 G2P 模型或新的辅助数据集。

第二步：调整两个损失权重。

将模型配置修改为：

$$\boxed{ L\_{\mathrm{total}} =L\_{\mathrm{CLM}} +\alpha(t) \left(0.03L\_{\mathrm{phon}}+0.03L\_{\mathrm{glyph}}\right) }$$

其中：

$$L\_{\mathrm{glyph}} =\frac{L\_{\mathrm{stroke}}+L\_{\mathrm{radical}}+L\_{\mathrm{structure}}}{3}.$$

按第一轮 Phase2-A 末尾的损失量级估算，原本约 80:20 的字音／字形加权贡献会调整为约 61:39，整体辅助损失占 CLM 的比例也会从约 10% 降至约 8%。

这是基于第一轮实际损失的工程起点，并不代表梯度贡献已被精确平衡。Phase2-A 的 10M-token 辅助权重 warm-up 保持不变。

第三步：调整配置入口与记录。

修改：

- `src/configuration_qwen3_pgca.py`：设置新的默认辅助权重；
- `src/pgca/build_pgca.py`：增加明确的权重构建参数，并写入最终 config；
- `scripts/train/ms_swift/cpt/plugin.py`：保留原计算公式，新增每项加权损失比例以及分模块梯度记录。

建议在 Phase2-B 持续记录 `weighted_phon_to_clm`、`weighted_glyph_to_clm`，而不只记录两者之和。

第四步：保留 Phase2-A 的冻结策略。

Phase2-A 冻结主干，使 CLM 和辅助任务的梯度都需要通过新增分支适配；Phase2-B 再全部解冻。现有 Auxiliary Heads 仍接在最终 Hidden States 上，这不保证 B 阶段一定依赖 PGCA。因此，不建议现在再新增“直接从 PGCA 输出预测下一字”的分类头：PGCA 输出主要携带当前位置的音形信息，该新任务存在监督与表示能力不匹配的风险，应先完成 P0。

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

## P2｜PGCA 融合结构改进

目标：解决拼音候选数量影响模态权重的问题，同时建立残差幅度的可观测性。保持原有两条 Attention 分支并行。

### 实施步骤

第一步：增加候选数量校正开关。

当前 PGCA 在所有有效拼音候选和字形 slot 上一起 softmax。可以在 [`src/pgca/attention.py`](https://github.com/Nephamaster/CC-LLM/blob/main/src/pgca/attention.py?utm_source=chatgpt.com) 中加入：

$$s'\_{p}=s_p-\log\big(\max(1,N_p)\big), \qquad s'\_g=s_g$$

其中 \\(s_p,s_g\\) 分别是拼音与字形 slot 的注意力 logit，\\(N_p\\) 是该 token 的有效拼音候选数。

在所有 slot 原始分数相等时，校正后拼音模态与字形模态各获得 50% 总权重。它是一个固定的数学校正，不是可学习 gate，不会添加参数。

建议新增配置 `pgca_balance_modalities`，默认关闭；仅当 P0 显示候选数量造成系统性模态偏置时，在第二轮开启。这样可以避免无证据地同时修改损失、优化参数和融合行为。

第二步：增加分支残差幅度日志。

在 `Qwen3PGCADecoderLayer` 中，对实际参与相加的 PGCA 输出计算范数，但不要将它改成新的训练状态。

记录：

$$r_l=\frac{\\|F_l\\|\_2}{\\|A_l\\|\_2+\epsilon}.$$

若 P0 显示 PGCA 明显损害 CLM，且部分层 \\(r_l\\) 异常偏高，再考虑增加固定 `pgca_residual_scale`，例如从 0.5 做一次验证。没有这样的证据，就保持直接相加。

第三步：保留当前 4 层上下文编码器。

不新增独立的音形 Transformer、不扩大 hidden size，也不调整六个 PGCA 插入位置。第二轮首先要判断现有结构是否产生增益，而不是再次改变模型容量。

### 最少测试与记录

| 测试            | 验收要求                     |
| ------------- | ------------------------ |
| 1／4／16 个拼音候选  | 等分数下校正后的拼音／字形模态总权重各约 0.5 |
| 无拼音、只有字形      | 字形权重为 1                  |
| 非汉字、全无效 slots | PGCA 输出严格为 0             |
| 因果前缀不变性       | 改变后缀不影响前缀输出              |
| BF16 与反向传播    | 输出和梯度有限，PGCA 正常更新        |

验收： 保留原 `state_dict` 结构、没有新 gate 参数；旧配置关闭校正时结果与现有 PGCA 在数值容差内一致。

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

这只是关键超参数示例。正式启动时，必须指定第二轮独立的 `MODEL`、`OUT`／阶段衔接根目录，以及与新模型一致的 `CACHE_DIR`，防止误读第一轮的 `next_stage_model.txt`。Phase2-A 和 B 使用相同的数据、种子、Token 预算和缓存，才能保证两阶段监督目标不重叠；只改变梯度累积不要求改变每卡训练 batch。

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
├── p1_auxiliary/
│   └── loss_validation.json
├── p2_fusion/
│   └── fusion_validation.json
└── training/
    ├── phase1/
    ├── phase2_a/
    └── phase2_b/
```

`experiment_manifest.json` 固定记录以下信息，不允许运行后悄悄变更：

| 类别    | 必须记录的字段                                             |
| ----- | --------------------------------------------------- |
| 代码与模型 | Git commit、模型 config、初始权重来源、Tokenizer SHA256、音形索引指纹 |
| 数据    | Phase1/Phase2 数据版本、验证集指纹、A/B Token 区间、随机种子          |
| 架构    | PGCA 层号、上下文编码器参数、模态校正开关、残差缩放                        |
| 损失    | 完整公式、权重、辅助 warm-up、标签 schema                        |
| 优化    | 各参数组 LR、实际有效 Token Batch、WD、scheduler、warm-up、梯度裁剪  |
| 结果    | 各阶段 checkpoint、训练 Token 数、验证 CLM／辅助指标、P0 诊断结果       |

训练中保留现有 loss 和 accuracy 日志，并补充 P0 规定的分模块梯度。每次保存 checkpoint 不必记录完整 Attention 张量：只要每约 1B Phase2 tokens 对固定诊断集运行一次，保存分层聚合统计即可。

## 3. 最终执行顺序与停止条件

1. P0：先诊断第一轮 checkpoint-11500。 输出一份 PGCA On/Off、Attention 和梯度报告；由结果决定 P2 的模态校正是否开启。
2. P1：完成辅助损失调整。 固定 `0.03/0.03`，验证 mask、损失和分布式归一化。
3. P2：完成融合开关及测试。 不改残差拓扑、不加层；是否启用候选校正由 P0 决定。
4. 冻结第二轮模型与代码版本。 重新构建初始 Char-PGCA，核验 Schema、权重和训练阶段开关，生成 manifest。
5. 按 Phase1 → A → B 完整训练。 在 B 的 1B、3B 和最终训练点做固定验证和 PGCA 旁路评估，避免只用 CPT Loss 判断效果。

只设置两类必须满足的条件： 开训前，所有数值、mask、因果和参数冻结测试通过；训练中，固定验证集无持续异常、特征分支有实际梯度、checkpoint 与数据版本完整可追溯。若 P0 发现严重结构性错误，应先修复再开训；若只表现为收益较弱，则不扩大实验规模，按固定方案进入第二轮。

目前不缺新的训练日志。执行 P0 时由服务器提供第一轮 checkpoint-11500 的实际模型权重、配套运行代码及现有编码缓存；不必上传到这里，可以在服务器有空闲 GPU 时运行诊断。先保留固定 checkpoint 副本，防止继续训练时被保存数量上限清理；多音字初稿另需用户审核。

整个方案将第二轮的主要变量控制为：更大的有效 Batch、更短的 B warm-up、WD=0.01、较均衡的辅助损失，以及由 P0 决定是否开启的候选数量校正。保留原并行 PGCA 拓扑和 4 层因果编码器，使后续结果仍有明确的解释基础。
