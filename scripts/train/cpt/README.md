# CC-LLM Phase1 / Phase2 训练

本目录固定由 CC-LLM 管理，不移动到 ms-swift 仓库。训练环境位于 `scripts/train/env/`，通过普通安装提供 ms-swift；插件调用模型 checkpoint 自带的 `multitask_loss_sums`，不依赖外部框架源码目录。

目标接口：ms-swift `4.7.0.dev0`，交付时本地源码 HEAD 为 `ce28e0288657d6472a439ea219f1bcc77310e764`（其 `swift/` 与审查起点 `1516d78d627381c2072be80037a7a64c3811480d` 无差异）。隔离验证环境使用 PyTorch `2.10.0+cpu`、Transformers `5.12.1`。这不是服务器 CUDA/显存验证；不要为了照抄测试版本在 GPU 环境安装 CPU wheel。

先在 CC-LLM 根目录准备环境（GPU 环境先安装匹配 CUDA 的 PyTorch）：

```bash
python3.11 -m venv scripts/train/env
scripts/train/env/bin/python -m pip install -r scripts/train/requirements.txt
scripts/train/env/bin/python -m pip check
```

## 1. 先构建新模型

前提：`models/Qwen3-1.7B-Base-Char` 包含原有语义 tokenizer/权重、修复后的 schema v2 音形资源、匹配的 `embedding_config.json`。schema 版本记录在 manifest，不写成模型名称后缀。不需要重新生成语义词表，也不能把新音形索引直接替换进旧 PGCA 权重；旧目录存在时请准备独立输入并显式指定路径，不自动覆盖。

在 CC-LLM 仓库根目录执行，输出目录必须尚不存在：

```bash
scripts/train/env/bin/python -B -m src.pgca.build_pgca \
  --model-path models/Qwen3-1.7B-Base-Char \
  --output-path models/Qwen3-1.7B-Base-PGCA \
  --pgca-layers 11,12,13,14,15,16 \
  --context-encoder --auxiliary-heads --zero-init-output \
  --aux-phonetic-weight 0.03 --aux-glyph-weight 0.03

scripts/train/env/bin/python -B -m src.pgca.validate_pgca \
  --model-path models/Qwen3-1.7B-Base-PGCA
```

核对构建报告的 `context_parameter_count`、`auxiliary_parameter_count`、`pgca_initialization.passed`，以及验证报告 `passed=true`。实际候选宽度来自 manifest，修复后的默认值为 16＋1 slots；辅助类别从合法 ID 3 开始映射，排除 pad/none/unknown。模型保存类别名称、资源哈希和 token 类型索引指纹。

P1 更新后，新构建模型的辅助权重默认是字音 `0.03`、字形 `0.03`，构建参数会显式写入 `config.json` 和构建报告。使用 `scripts/build_pgca.sh` 时，可在脚本顶部或通过 `AUX_PHONETIC_WEIGHT`、`AUX_GLYPH_WEIGHT` 调整；10M-token 辅助 warm-up 不变。已有 checkpoint 中的显式 `0.05/0.02` 保持原值，加载和 resume 不会自动迁移权重；源码默认值变化也不会更新磁盘上的旧 config。准备第二轮模型请使用独立目录，不就地覆盖第一轮模型。

## 2. 一个阶段一个脚本

三个脚本完全独立：各自顶部包含该阶段的全部参数，下面直接准备数据并启动训练，不再 source 公共脚本。旧 `common.sh`、合并入口 `phase2.sh` 已移除；`plugin.py` 和 `prepare_data.py` 仍负责原有模型插件及数据处理，不存放脚本超参数。

| 脚本 | 可训练参数 | 目标与默认学习率 | 长度 | 默认输入 |
| --- | --- | --- | --- | --- |
| [phase1.sh](phase1.sh) | 主干及语义输入输出层；PGCA 旁路冻结 | 纯 CLM，`PHASE1_LR=1e-5` | 2048 | 初始 PGCA 模型 |
| [phase2_a.sh](phase2_a.sh) | 音形 embedding、上下文编码器、PGCA、辅助头 | 三任务，`PHASE2_A_LR=5e-5` | 4096 | Phase1 完成指针 |
| [phase2_b.sh](phase2_b.sh) | 全参数 | 三任务，主干 `PHASE2_B_LR=1e-5`，分支 `PHASE2_B_BRANCH_LR=3e-5` | 4096 | Phase2-A 完成指针 |

阶段切换新建优化器和调度器，不是 resume。A 选择最后 checkpoint，Phase1/B 按 `eval_clm_loss` 选优。A/B 默认共用 10B 上限，A 使用前 100M 有效目标，B 使用其余部分；辅助权重在前 10M 有效目标内升至模型 config 保存的值（新构建默认 `0.03/0.03`，首轮旧模型为 `0.05/0.02`）。标签、shift、mask、三个优化损失的 CLM 有效 token 分母及字形三项平均均未改变。

### 修改参数

直接打开要运行的脚本，修改顶部相应参数。GPU、batch、梯度累积、缓存开关、优化器、保存间隔等都在同一个文件，不需要找另一个 common。环境变量仍优先于文件默认值。

当前工作区配置原样迁入三个脚本：**每卡训练 batch=1、验证 batch=1、梯度累积=32、数据处理进程=4**。以后修改某个脚本只影响该阶段。

例如，在 `phase1.sh` 中把 `PHASE1_LR="${PHASE1_LR:-1e-5}"` 改为 `PHASE1_LR="${PHASE1_LR:-2e-5}"`，或者只对一次运行覆盖：

```bash
PHASE1_LR=2e-5 bash scripts/train/cpt/phase1.sh
```

关键参数：

- `PYTHON_CMD`：默认 `scripts/train/env/bin/python`；同一解释器负责导出、数据准备和 Swift 启动。
- `MODEL_NAME`：基础模型 M，默认 `Qwen3-1.7B-Base-PGCA`；`MODEL` 是本阶段的输入模型路径，显式值优先。
- `ALIGN_OUT`：仅 A 使用，默认读取 `outputs/train/M-ALIGN/next_stage_model.txt`。
- `CPT_ROOT`：A/B 的产物根目录，默认 `outputs/train/M-CPT`。A 的 `OUT` 默认是其 `adaptation/`，B 的 `OUT` 默认是其 `joint/`；B 完成后向 `CPT_ROOT` 发布最终指针。
- `ADAPTATION_OUT`：仅 B 使用，默认 `$CPT_ROOT/adaptation`，从该目录的完成指针选择 A 的 checkpoint。
- `OUT`：当前阶段的输出根目录。新训练由 ms-swift 自动创建版本子目录；断点恢复继续原版本。
- `DATA_DIR`：包含 `train/*.jsonl` 和 `validation/*.jsonl` 的目录；`CACHE_DIR` 是数据准备目录。
- `PHASE1_TOKENS`：上限 1B；`PHASE2_TOKENS`：A+B 总上限 10B。数据不足时按实际可用量继续，0 表示全部；`ADAPTATION_TOKENS` 默认 100M，必须小于 Phase2 实际总量。
- `MAX_LENGTH`、卡数、batch、种子和预算都参与缓存适配检查；`LOAD_FROM_CACHE_FILE` 默认 true，仅控制原生预处理缓存复用，不是模型 KV cache，也不强制重建已有准备产物。
- `WARMUP_RATIO` 是学习率 warmup；字音/字形权重和辅助 warmup 仍来自模型 config 的 `pgca_aux_*`。
- 三个脚本统一使用 `RESUME_FROM_CHECKPOINT` 指定本阶段断点，不再使用 `RUN_STAGE`、`RESUME_A`、`RESUME_B`。

**A/B 必须保持相同的语料、tokenizer、MAX_LENGTH、DATA_SEED、PHASE2_TOKENS 和 ADAPTATION_TOKENS**，以维持监督区间不重叠。默认共用同一 `CACHE_DIR`；若需要不同卡数或训练 batch，应分别准备匹配的数据视图并设置不同 `CACHE_DIR`，可复用同一份 `.raw`，不修改原始数据。

### 启动与自动衔接

在仓库根目录按顺序执行；每个命令只运行一个阶段：

```bash
bash scripts/train/cpt/phase1.sh
bash scripts/train/cpt/phase2_a.sh
bash scripts/train/cpt/phase2_b.sh
```

也可以显式固定输入 checkpoint：

```bash
MODEL=/实际路径/Phase1/checkpoint-N bash scripts/train/cpt/phase2_a.sh
MODEL=/实际路径/Phase2-A/checkpoint-N bash scripts/train/cpt/phase2_b.sh
```

输出保持原布局：Phase1 为 `outputs/train/M-ALIGN/vN-时间戳/`；A/B 分别为 `outputs/train/M-CPT/adaptation/vN-时间戳/`、`joint/vN-时间戳/`，日志在对应版本的 `runs/` 下。已有 checkpoint 和默认目录下的完成指针可继续使用，不需要迁移。

每个版本及阶段根目录都保留 `next_stage_model.txt`。只读取最近完成训练发布的指针，不按最大版本号猜测未完成的 run。B 完成后继续发布 `outputs/train/M-CPT/next_stage_model.txt`，下游微调默认入口不变。并发实验使用不同 `CPT_ROOT`/`OUT`，避免共享完成指针。

默认脚本仍是完整 PGCA 方案；消融方案应显式安排，不能把单个完整方案当作机制验证。
## 3. 数据准备与 token 预算

首次运行脚本会调用官方 `swift export`，用本插件的模板执行无 chat template 的预训练编码，再调用 `prepare_data.py`。不重采语料，不执行逐文 G2P，不改原始 JSONL。

这里有两个不同层面：ms-swift 的预处理缓存复用由 `LOAD_FROM_CACHE_FILE` 控制；`prepare_data.py` 的额外工作是有效 token 预算、A/B 分段、无效监督过滤和 DDP 补齐，产物仍是 Hugging Face Dataset。后者不是另写一套 tokenizer 缓存。

已有 `CACHE_DIR/preparation_report.json` 时，脚本继续校验并复用准备好的数据，不重新调用导出。因此 `LOAD_FROM_CACHE_FILE=false` **不等于强制重建 `CACHE_DIR`**；需要重新准备时，指定一个新的 `CACHE_DIR`。已有产物不会自动删除或覆盖。例如：

```bash
LOAD_FROM_CACHE_FILE=true bash scripts/train/cpt/phase1.sh
```

准备过程：

1. 在模板切分后过滤零有效目标分片；再次检查缓存 labels/input_ids 和序列长度。
2. 用 Arrow/NumPy 分批计数 `labels[1:] != -100`，固定 seed 打乱训练样本；验证集不打乱。
3. A/B 使用不重叠的监督目标区间。跨分界的同一输入片段可能出现在两段中，但 labels 使用互补 mask，不重复训练目标，也不丢掉边界目标。
4. 训练样本数补齐到 `world_size × per_device_train_batch_size` 的倍数。补齐行复制输入但全标签置 -100，携带 `is_padding_dummy=true`，贡献图连接的零损失；避免 DDP 自动重复真实样本。改变卡数/batch 时必须重新准备视图。
5. 保存 `preparation_report.json`：请求上限 `requested_train_targets`、实际量 `train_targets`、短缺量 `shortfall_targets`，以及丢弃行数、A/B 区间、补齐行数、tokenizer 指纹、长度及 batch 几何。缓存复用按请求上限与可用量的较小值校验，不再要求不足额的数据满足原上限。

缓存以 `--cached_dataset` / `--cached_val_dataset` 加载，必须保持 `truncation_strategy=split`。当前 Swift 对非 split 缓存会走 LazyLLMDataset，不能把编码后的行重新当作原始 messages。

原始官方缓存保存在 `${CACHE_DIR}.raw`，分段缓存和中间 Arrow 文件保存在 `CACHE_DIR`。**需预留额外磁盘空间**，不能把它们视为仅有几个字节的索引文件。已有兼容官方缓存可直接调用：

如果旧版本因预算不足停在 `prepare_data.py`，此时 `.raw` 通常已成功导出，分段目录尚未创建；可直接运行下面的准备命令（填写本阶段的实际路径、长度、卡数和 batch），完成后再启动原训练脚本，不必重新导出。直接重跑整段脚本仍可能触发已有 raw 目录的保护检查。

```bash
scripts/train/env/bin/python -B scripts/train/cpt/prepare_data.py \
  --input /已有官方缓存根目录 \
  --output /新的分段缓存目录 \
  --model /本轮模型目录 --max-length 4096 \
  --world-size 8 --batch-size 1 \
  --adaptation-tokens 100000000 --total-tokens 10000000000
```

2048 缓存不能伪装成 4096 长上下文训练数据。新模型必须复用相同语义 tokenizer；改变 tokenizer 后不能复用旧 token IDs。

## 4. 断点恢复

三个阶段各自在自己的脚本中配置 `RESUME_FROM_CHECKPOINT`，或通过环境变量传入：

```bash
RESUME_FROM_CHECKPOINT=/本轮ALIGN/v0-时间戳/checkpoint-N \
bash scripts/train/cpt/phase1.sh

RESUME_FROM_CHECKPOINT=/本轮CPT/adaptation/v0-时间戳/checkpoint-N \
bash scripts/train/cpt/phase2_a.sh

RESUME_FROM_CHECKPOINT=/本轮CPT/joint/v0-时间戳/checkpoint-N \
bash scripts/train/cpt/phase2_b.sh
```

恢复时保持该阶段原 `OUT` 根目录、缓存、模型/数据版本、卡数、batch、累积步数、预算、种子和优化器配置。脚本定位断点所在的版本目录，关闭自动加版本并继续写入。A 恢复完成后不会自行启动 B；需要运行 B 脚本。`cc_llm_state.json` 和模型 buffer 继续保存辅助训练进度，不重置辅助权重 warmup。
## 5. 验证指标与工程边界

- 以 `eval_clm_loss` 选优，PPL 只能用 `exp(eval_clm_loss)`。`eval_total_loss` 是含辅助任务的联合指标。
- `eval_loss` 使用与 `eval_total_loss` 相同的全局有效 token 加权统计，在框架记录日志和返回验证指标前覆盖默认聚合值；不随卡数缩放，也不对不同长度 batch 的均值再做简单平均。Phase1 三项 loss 相等；Phase2 的 `eval_loss` 包含辅助损失，不能用于 PPL。此修正不改变训练 loss/梯度或已有 checkpoint，旧日志不会自动修正。
- `eval_phon_loss`、`eval_glyph_loss` 使用 CLM token 分母；`*_supervised_loss`、`*_accuracy` 使用各自有效监督分母；同时记录 `*_tokens`。
- 验证指标先通过 Accelerate 去掉 DDP 尾部重复样本，再按 token 累计。优化损失只由 Swift 乘一次 world size，不在插件里重复放大。
- 训练和验证新增 `weighted_phon_to_clm`、`weighted_glyph_to_clm`，保留原 `weighted_aux_to_clm_ratio`。各项比例为日志窗口内实际加权（训练时包含辅助 warm-up ramp）的 loss 总和 / CLM loss 总和，不是逐 batch 比例的平均；验证字段带 `eval_` 前缀。无辅助监督但有 CLM 目标时为 0；整个窗口没有 CLM loss 分母时为 null，不伪造比例。
- 保留 `backbone_grad_norm`、`branch_grad_norm`，新增 `feature_embedding_grad_norm`、`context_encoder_grad_norm`、`pgca_{q,k,v,o}_proj_grad_norm`、`pgca_norms_grad_norm`、`auxiliary_phonetic_grad_norm`、`auxiliary_glyph_grad_norm`。记录已有联合损失的梯度，每个参数只计算一次梯度平方和，再汇总到总体/细分组；不增加独立任务反向、optimizer step 或梯度修改。
- 当前 Trainer 在裁剪后、optimizer step 前触发该回调。因此上述范数是累积和同步后的联合梯度范数，按日志窗口对优化步取均值；不同于框架通常记录的裁剪前 `grad_norm`，也不是 Adam 更新量。该口径针对当前单卡/普通 DDP，不宣称是 FSDP/ZeRO 分片的全局梯度范数。冻结或没有梯度的组显式记录 0；PGCA 零输出初始化后，第一步内部 Q/K/V/上下文梯度为零是预期现象。三任务梯度余弦继续由离线诊断提供，不在每步训练中额外计算。
- 首版仅验证 HF DynamicCache 路径，支持 prefill、增量解码和 beam 重排。不支持 packing、padding-free、sequence parallel、Liger loss、logits-to-keep、StaticCache 或 vLLM；不能把旧 vLLM 后端当作已经支持新编码器。
- 正式 GPU 训练前应先用少量独立 JSONL 和显式小 token 预算做冒烟，检查真实显存、吞吐、三阶段冻结状态及有效验证指标。本地 CPU 测试不替代该步骤。

此前的独立 smoke 入口及说明已删除；正式训练使用上面的三个阶段脚本。不要将历史冒烟 checkpoint 用作正式第二轮起点。
