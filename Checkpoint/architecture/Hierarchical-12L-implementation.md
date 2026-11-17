# Hierarchical-12L 工程报告

实施基线：master `518a578` 的工作区。保留本轮开始时已有的两份 smoke 分析记录及 architecture 目录用户资料；未切换分支，未提交、合并或推送。本报告对应可选架构实现，不表示模型性能已提升。

## 实现结论

已实现可选 Hierarchical-12L：三个参数独立、各四层的 CausalFeatureEncoder 串行计算 M1 → M2 → M3，再按 PGCA 深度路由。默认仍为 Shared，旧参数名和旧配置缺省行为保留。

三级起点按用户补充规则从实际 `pgca_layers` 派生：

```text
first = min(pgca_layers)
middle = (first + max(pgca_layers)) // 2
last = max(pgca_layers)
```

当 PGCA 层为 9–16，起点为 `[9,12,16]`：9–11 使用 M1，12–15 使用 M2，16 使用 M3。不是旧草案中的 `[9,13,16]`。已有 checkpoint 的 PGCA 层位不会被移动。

若不足三个互异起点，或稀疏 `pgca_layers` 不包含该中点，构建明确拒绝，不擅自取邻近层。配置保存 `pgca_context_mode` 和 `pgca_context_stage_starts`；显式边界仅用于确认公式结果，不接受与公式冲突的值。Hierarchical 要求 use_pgca/context_enabled 且每个编码器四层，Shared 原有可配置层数保持兼容。

## 结构、迁移与训练

`FeatureMemoryBuilder.context_encoder` 继续作为 C1，保留旧 state_dict 前缀；`context_refiners.0/1` 分别是新增 C2/C3。三者复用同一个编码器实现，不复制 Transformer 实现。C2 接收 M1，C3 接收 M2，无 detach。Shared 返回单个 Tensor，Hierarchical 返回三个 Memory 的 tuple，形状均为 B×T×S×D。Decoder 循环只选择 Memory，原有 Self-Attention 与 PGCA 并行直接相加的融合、CLM/音形辅助损失公式及分母没有改变。

显式升级入口先正常检查完整 Shared-4L 源 checkpoint，再保留主干、PGCA、Feature Embedding、C1、Feature Index、Token Type Index，新增 C2/C3。只有局部 FeatureMemoryBuilder 迁移允许精确列举的 C2/C3 新键缺失；正常加载 Hierarchical 时缺失或形状错误仍拒绝，不靠全局 strict=False 掩盖损坏。

C2/C3 的 output_proj 零初始化，使二者初始为恒等残差。其余参数正常初始化、均可训练；C1 和特征分支已有 dtype/权重保持不变。源目录不修改，输出另存。Tokenizer 文件原样复制；旧 trainer_state/optimizer 恢复信息不作为新结构的恢复状态发布。升级产物是新的模型训练起点，不能当作原优化器状态的 resume。

从 Char 新建时三个编码器正常初始化；现有 zero-init-output 仍只用于 PGCA 输出投影。构建报告记录总参数量、完整分支参数量以及 C1/C2/C3 各自参数量。升级报告为 `hierarchical_upgrade_report.json`，正常构建报告仍为 `pgca_migration_report.json`。

Phase1 原有正则冻结整个 feature_memory_builder，包含 C1/C2/C3，PGCA 旁路；Phase2-A 的原有整分支正则训练全部编码器；Phase2-B 优化器按 feature_memory_builder 前缀为三个编码器选择分支学习率。梯度日志保留 context_encoder_grad_norm（C1），新增 context_encoder_c2_grad_norm、context_encoder_c3_grad_norm。LoRA 保持 all-linear、辅助头冻结，并逐一检查三个编码器都已纳入 LoRA。

后续另增 Phase2 的独立 `ENABLE_AUXILIARY_TASKS` 开关，默认 true，保持上述原设计；false 时冻结辅助头并跳过其计算，PGCA 与三级编码器仍由 CLM 训练，无需修改或重建结构。用法与恢复限制见 [CPT 说明](../../scripts/train/cpt/README.md)，该开关的独立检查见 [任务记录](../2026-11-17-auxiliary-task-switch.md)。

| 修改文件 | 关键接口/职责 |
|---|---|
| src/configuration_qwen3_pgca.py、src/pgca/config.py | 模式、自动三点边界、配置合法性与序列化 |
| src/embedding/feature_memory.py、context_encoder.py | 串行编码、三份 Memory、独立 feature_caches 及生命周期 |
| src/modeling_qwen3_pgca.py | `_prepare_feature_memory`、Decoder 路由、Cache 同步与必需权重检查 |
| src/pgca/build_pgca.py、upgrade_context.py、validate_pgca.py | 新建、显式升级、自包含导出、参数报告、缓存数值验证 |
| scripts/model_build/build_pgca.sh | CONTEXT_MODE 与 PGCA_LAYERS 透传 |
| scripts/train/cpt/plugin.py | 新编码器梯度统计；沿用原有整分支冻结和学习率判断 |
| scripts/train/downstream/plugin.py、dev_check.py | 三个编码器的 LoRA 覆盖与三组 KV 长度验收；General-SFT 同样复用 |
| scripts/model_build/test_hierarchical.py | 可重复的 CPU 结构/数值/迁移/LoRA 回归与实际 Swift 阶段短跑 |

## Cache 组织与边界

```text
PGCACompositeCache
├── backbone DynamicCache
└── feature_caches
    ├── C1 DynamicCache（内部 0–3 层）
    ├── C2 DynamicCache（内部 0–3 层）
    └── C3 DynamicCache（内部 0–3 层）
```

Shared 只有一个 feature cache；旧 feature_cache 访问器仍指向第一个。模型在增量计算前后检查各 Cache 长度及阶段数；拒绝复用同一个 Cache 对象。每阶段只追加当前 Token/增量块，M2/M3 当前输入来自前一级当前输出。

reorder、repeat、select、crop、reset 同时作用于主干和所有音形 Cache。crop 保留“正数表示保留总长度、负数表示删除尾部数量”的接口，内部转换为上游各版本共同支持的负数删除方式，避免 Transformers 5.22 取消正数 crop 后破坏旧 Shared 用法。

仅沿用现有 DynamicCache、unpacked 2D padding mask 范围。非空 backbone-only Cache、阶段数错误、Cache 不同步或复用、训练启用 context cache、外部 Memory 绕过上下文、4D/字典特征 Mask 均明确拒绝。StaticCache 等非 DynamicCache 输入不冒充已支持。没有实现新的主干语义条件化或门控。

模型验证器增加 full/prefill+decode 对照和各阶段 Cache 长度记录。对 ALIGN 中保存的 pgca_enabled=false，会临时启用分支审计 Cache 后恢复原设置，报告记录这一点，避免只检查被旁路的主干。

## 可执行命令

以下在项目根目录执行，路径均需换成实际模型。模式与边界写入 checkpoint 后，后续训练不需要反复传入结构开关。

### 从已有 Char 构建

```bash
CHAR_MODEL=/absolute/path/to/Char
HIER_MODEL=/absolute/path/to/PGCA-Hierarchical-12L

.venv/train/bin/python -B -m src.pgca.build_pgca \
  --model-path "$CHAR_MODEL" --output-path "$HIER_MODEL" \
  --context-encoder --context-mode hierarchical \
  --auxiliary-heads --zero-init-output
```

默认 PGCA 层位仍按主干中间区域选择；需要指定插入区域时加 `--pgca-layers 9,10,11,12,13,14,15,16`（要求主干实际包含这些层）。无需手工指定边界；可加 `--context-stage-starts 9,12,16` 确认，错误边界报错。

如果连 Char 也需重建，原总入口支持：

```bash
BASE_MODEL=/absolute/path/to/Qwen3-Base \
BASE_MODEL_NAME=Qwen3-Base \
CONTEXT_MODE=hierarchical \
PGCA_MODEL="$PWD/models/Qwen3-Base-PGCA-Hierarchical-12L" \
bash scripts/model_build/run.sh
```

PGCA_LAYERS 可覆盖上述 shell 入口的插入区域，未设置则自动计算。Shared 对照不设置 CONTEXT_MODE，或显式设 shared。

### 从 Shared-4L 完整 checkpoint 升级

```bash
SHARED_MODEL=/absolute/path/to/Shared-checkpoint
HIER_MODEL=/absolute/path/to/PGCA-Hierarchical-12L

.venv/train/bin/python -B -m src.pgca.upgrade_context \
  --model-path "$SHARED_MODEL" --output-path "$HIER_MODEL"
```

源必须是完整、启用四层 context 的 Shared PGCA，不接受未合并 Adapter。升级不会改变源 PGCA 层位，不会修改源 checkpoint。

### 校验自包含模型

```bash
.venv/train/bin/python -B -m src.pgca.validate_pgca \
  --model-path "$HIER_MODEL" --text '银行运行中国ABC'
```

报告位于模型的 reports/pgca_validation_report.json，包含三个 Memory 形状、编码器参数量、Cache 长度、最大误差和 Top-1 对照。

### Phase1、Phase2-A、Phase2-B

从 Char 新建的 Hierarchical 可按原链路从头训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_NAME=Qwen3-Base-PGCA-Hierarchical-12L
export TRAIN_ROOT="$PWD/outputs/train/hierarchical"
export TRAIN_BATCH_SIZE=1 EVAL_BATCH_SIZE=1

MODEL="$HIER_MODEL" bash scripts/train/cpt/phase1.sh
bash scripts/train/cpt/phase2_a.sh
bash scripts/train/cpt/phase2_b.sh
```

三个阶段的完成指针自动衔接，学习率、Token 预算、长度和梯度累积仍按原脚本配置。卡数和 batch 以实际资源确定，不把上例当作所有规模的显存保证。从已训练 Shared 升级时，应从实验计划对应的阶段用 MODEL 指定新模型开始，不重复执行已完成阶段，也不继承旧优化器 resume。

General-SFT/专项 LoRA 使用原入口，MODEL 指向新结构的完整 checkpoint；没有更改指令成品数据、模板、EOS 或损失口径。

### 自动测试与小规模冒烟

```bash
.venv/train/bin/python -m pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple

# 快速 CPU 单元回归；较慢的真实 Swift 短跑默认跳过。
.venv/train/bin/python -B -m pytest -q -p no:cacheprovider \
  scripts/model_build/test_hierarchical.py

# 同时执行实际 Phase1、Phase2-A、Phase2-B，各四步，使用临时合成模型和数据。
CC_LLM_RUN_SWIFT_SMOKE=1 \
.venv/train/bin/python -B -m pytest -q -p no:cacheprovider \
  scripts/model_build/test_hierarchical.py
```

CPU 短跑调用真实阶段 shell、ms-swift、损失/优化器/回调、数据编码和保存加载，不替换模型前反向。测试目录由 pytest 管理，不读取或改写项目成品数据；Python 3.14 下仅在测试子进程使用临时 sitecustomize 设置 fork，以匹配项目支持的 Linux/Python 3.12 工作进程语义。

具备完整 Base 权重、数据与 GPU 时，可运行真实集中冒烟：

```bash
CONTEXT_MODE=hierarchical CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/smoke/run.sh /absolute/path/to/Qwen3-Base
```

这条实卡命令本轮未执行，不把之前 Shared 的 smoke v1/v2 作为新结构验收。

## 实际测试结果与证据

测试环境：Python 3.14.6、Torch 2.15.0+cpu、Transformers 5.22.2、ms-swift 4.6.1、PEFT 0.20.0。环境仅在 /tmp/cc-hierarchical-20261117/env，未改项目依赖版本或用户现有环境；这些是记录的验证基线，不是新版本锁。

| 测试项 | 执行情况 | 结果/证据 |
|---|---|---|
| Shared 旧配置与原有上下文测试 | 已执行 | 读取 main 中原有 13 项测试，包含子测试；通过，旧 C1 名称与单 Memory 路径保留 |
| 三个四层模块、串行输入与路由 | 已执行 | Hook 捕获 C2 输入=M1、C3 输入=M2，9–11/12–15/16 路由正确，参数不共享 |
| 非汉字、缺失拼音、Padding、因果性 | 已执行 | 无效槽保持零，未来 Token 不改变前缀；2D Mask 正常，4D/字典 Mask 明确拒绝 |
| 初始化与梯度 | 已执行 | 升级 M1=M2=M3；零投影起始内部梯度可为零，更新后全部编码器获得梯度并输出不同 Memory |
| Checkpoint 升级、Tokenizer/索引、Save/Load | 已执行 | tied/untied 均通过，初始 Logits 最大差为 0；AutoConfig/AutoModel 与 portable runtime 校验通过；缺失/错形状 C2/C3 拒绝 |
| Cache On/Off | 已执行 | 单 Token/多 Token 增量、左 Padding 批次、重排/重复/选择、正负裁剪后续解码、reset、Greedy 和 Beam Search 通过 |
| Phase1 冻结与真实 CPU 训练 | 已执行 | 四步完成，三个 context 梯度均为零，验证 CLM loss 有限 |
| Phase2-A 冻结与真实 CPU 训练 | 已执行 | 四步完成，C1/C2/C3 在 warmup/零输出延迟后均出现非零梯度 |
| Phase2-B 学习率组与真实 CPU 训练 | 已执行 | 四步完成，全部新模块分到分支学习率组，三组 context 梯度非零，验证 loss 有限 |
| General-SFT/下游 all-linear LoRA | 已执行 CPU 模型级检查 | 三个编码器每个 Linear 均被覆盖，辅助头冻结，前反向、Adapter 保存/重新加载输出一致；未宣称正式 SFT 生成质量通过 |
| 真实规模 GPU 训练/推理与全流程 | 未执行 | 本轮仅有 CPU 验证环境；需用户实卡冒烟，不宣称性能/质量或显存通过 |

数值探针采用 tiny 配置（hidden=16，context hidden=8，每级四层）：升级前后 Logits 最大绝对差 0；Hierarchical FP32 full vs cached 最大差 `3.725290298461914e-08`、Top-1 一致率 100%；CPU BF16 此样例最大差 0、Top-1 一致率 100%。Prefill 长度 3、逐 Token 到 6 时，主干与三个 feature Cache 均逐步为 3/4/5/6。BF16 样例不等于真实 GPU BF16 已验收。

同一 tiny 配置每级编码器 3,312 参数，三级共 9,936；完整模型 29,732、完整分支 14,580 参数。真实规模请读取各自构建报告，不外推本表参数量。

本架构改造轮（独立辅助开关引入之前）的完整执行结果为 **48 passed、6 个 unittest 子测试通过**，包含当时新增的 35 个参数化用例（其中一个用例执行三阶段真实 CPU 短跑）与原有 13 项 Shared 回归。默认快速执行不启用 Swift 短跑时，该单个慢用例明确标为 skipped。三个 warning 为上游 Python 3.14 JIT 提示与合成 LoRA 基座路径提示，没有忽略测试异常。后续辅助开关用例已加入同一个测试文件，当前用例数量会相应增加。

实际完整测试日志与数值结果在 `/tmp/cc-hierarchical-20261117/`：`complete.log`、`unit-complete.log`、`final-validation.log`、`numerics.json`，以及 pytest 目录内的三阶段训练日志/trainer_state。原有 Shared 测试从 main 复制至临时目录执行，没有修改其断言。临时 CPU 短跑最初因 Python 3.14 forkserver 无法反序列化 Swift 动态插件而失败；仅测试进程切换 fork 后链路通过，未顺带改生产运行时。

## 后续对照与限制

对照应从同一个 Shared checkpoint 出发，一份保留 Shared、一份显式升级，保持 PGCA 层位、Tokenizer、数据顺序、训练 Token、优化器起点与评测协议一致。另记录吞吐、峰值显存和三份 Memory/KV 的额外开销，不假设沿用 Shared batch 一定可行。本轮不宣称 Hierarchical 更优。

没有扩展 packed/4D mask、静态 Cache 或新的主干语义条件化，也没有提供 vLLM 原生 PGCA 内核。正常 PGCA 仍走 HF 后端。实际 Python 3.12 + NVIDIA GPU、大规模、多卡训练和部署后端仍待用户验收；源码与 CPU 证据已交付。
