# CC-LLM：Hierarchical-12L 级联音形上下文编码器改造

实现与实际测试结果见 [工程报告](Hierarchical-12L-implementation.md)。Shared 为默认模式，Hierarchical 需显式启用。

## 0. 任务目标

当前 CC-LLM 的 PGCA 架构仅有一个四层 `CausalFeatureEncoder`，在主干 Decoder 循环之前将静态音形特征编码成一份上下文化 Memory，供所有 PGCA 层共享。

现在需要新增一种可选的 **Hierarchical-12L** 架构：

- 将现有四层因果编码器扩展为三级串行结构。
- 每一级均复用现有 `CausalFeatureEncoder` 的完整结构，包含四层因果 Transformer 和候选级残差融合。
- 第二级接收第一级的输出，第三级接收第二级的输出。
- 三级输出分别供不同深度区域的 PGCA 使用。
- 保留原始 Shared-4L 架构作为对照，不能破坏现有模型加载、训练与评估能力。

**不要将三个编码器实现为并行地接收静态音形特征，也不要修改 PGCA 为每层一个独立编码器。**

本次只实现 Hierarchical-12L，不扩展为主干语义 Hidden State 条件化、Gate 机制或其他新型融合架构。

## 1. 架构定义

### 1.1 当前 Shared-4L

静态音形 Embedding 输出：

\[
F\in\mathbb R^{B\times T\times S\times D}
\]

其中：

- B：Batch Size
- T：序列长度
- S：每个 Token 的音形候选槽数
- D：主干隐藏维度

以现有 1.7B 配置为例，S=17，D=2048。

现有架构：

\[
M=C(F)
\]

\[
P_l=\operatorname{PGCA}_l(H_l,M)
\]

所有启用 PGCA 的主干层共享同一份 M，但每层 PGCA 的 Q/K/V/O 投影保持独立。

### 1.2 新增 Hierarchical-12L

新架构：

\[
M_1=C_1(F)
\]

\[
M_2=C_2(M_1)
\]

\[
M_3=C_3(M_2)
\]

其中：

- C1、C2、C3 是三个参数独立的 `CausalFeatureEncoder`。
- 每个编码器包含四层因果 Transformer。
- 各编码器沿用现有特征摘要、Token Type、RoPE、候选级融合、Mask 和残差更新机制。
- 不共享三个编码器的 Transformer 参数。
- 不在级联过程中使用 `detach()` 或其他截断梯度的操作。
- 每个编码器的输入输出均保持 `[B,T,S,D]`。
- 三个编码器均不得访问未来 Token。

对于 PGCA 配置：

```text
pgca_layers = [9,10,11,12,13,14,15,16]
```

先采用以下分配规则（0-based）：

| 主干层 | 使用的 Memory |
|---|---|
| 9、10、11 | M1 |
| 12、13、14、15 | M2 |
| 16 | M3 |

相当于：

```text
Static Phonetic-Glyph Embedding
              |
              v
       C1: 4-layer Encoder
              |
              +--------> M1 --------> PGCA L9-L11
              |
              v
       C2: 4-layer Encoder
              |
              +--------> M2 --------> PGCA L12-L15
              |
              v
       C3: 4-layer Encoder
              |
              +--------> M3 --------> PGCA L16
```

三阶段边界必须从实际 `pgca_layers` 推导，不能写死层号：首点为最小 PGCA 索引，尾点为最大 PGCA 索引，中点为 `(首点 + 尾点) // 2`。调整 PGCA 插入区域会相应改变边界。`pgca_context_stage_starts` 保存派生结果；显式传入此字段或 CLI 参数时，只接受与该公式一致的值。若三点不互异，或稀疏 PGCA 配置中不包含该中点，明确拒绝 Hierarchical 模式，不擅自移动 PGCA 层。

注意：上述 9–16 层只是本次目标实验配置。实际已有 checkpoint 的 `pgca_layers` 可能不同，必须从 `config.json` 读取，不得擅自修改其 PGCA 插入位置。

建议新增配置：

```python
pgca_context_mode = "shared"       # shared / hierarchical
pgca_context_stage_starts = [9, 12, 16]
```

可调整命名，但语义必须明确，且必须支持保存和重新加载。

Hierarchical 模式根据 `pgca_context_stage_starts` 将 `pgca_layers` 分配给三个阶段：每个 PGCA 层使用不晚于自身层号的最后一个阶段输出。

需要验证：

- 阶段起点严格递增、唯一。
- 起点必须属于 `pgca_layers`。
- 第一个阶段起点必须覆盖最早的 PGCA 层。
- 所有 PGCA 层必须有确定的 Memory 来源。
- Hierarchical 模式要求 `use_pgca=true`、`pgca_context_enabled=true`。
- 如果配置不合法，构建时明确报错，不能静默回退。
- 旧配置不包含新字段时，默认使用原有 Shared-4L，不改变旧行为。

## 2. 实施前源码审查

请先检查下列文件及其调用关系：

```text
src/configuration_qwen3_pgca.py
src/embedding/context_encoder.py
src/embedding/feature_memory.py
src/modeling_qwen3_pgca.py
src/pgca/attention.py
src/pgca/build_pgca.py
src/pgca/validate_pgca.py

scripts/model_build/run.sh
scripts/model_build/build_pgca.sh

scripts/train/cpt/phase1.sh
scripts/train/cpt/phase2_a.sh
scripts/train/cpt/phase2_b.sh
scripts/train/cpt/plugin.py

scripts/train/instruction/
scripts/train/downstream/
scripts/evaluation/
scripts/smoke/
```

重点确认：

1. 当前 `FeatureMemoryBuilder` 如何构造和返回 Memory。
2. `Qwen3PGCAModel.forward()` 在何处生成 Memory，以及如何传递给 Decoder 层。
3. `Qwen3PGCADecoderLayer` 如何调用 PGCA。
4. `PGCACompositeCache` 如何管理主干和音形 KV Cache。
5. PGCA 构建程序如何初始化权重、复制运行时源码并生成自包含模型。
6. Phase1/Phase2 的参数冻结、不同学习率分组与梯度监控。
7. General-SFT 和 Downstream LoRA 是否会正确纳入新增编码器参数。
8. 模型 `save_pretrained()`、`from_pretrained()`、`AutoModelForCausalLM` 的兼容要求。

先确认现有实现，再进行最小必要改造。不要基于旧版本目录或推测修改。

## 3. 模型结构改造

### 3.1 保留现有编码器实现

优先复用现有 `CausalFeatureEncoder`，不要复制四层 Transformer 的实现代码，也不要重新发明一套 Encoder。

当前 `CausalFeatureEncoder` 已完成：

- 音形候选摘要
- 四层因果 Transformer
- Token 类型 Embedding
- 拼音缺失状态处理
- 候选级上下文融合
- 残差更新
- 特征 Mask
- KV Cache

新增一个负责级联组织与 Memory 分配的薄封装，或者在 `FeatureMemoryBuilder` 中完成组织。根据现有职责边界决定最终实现位置。

要求：

```python
memory_0, mask = feature_embedding(feature_ids)

memory_1, mask = encoder_1(memory_0, mask, ...)
memory_2, mask = encoder_2(memory_1, mask, ...)
memory_3, mask = encoder_3(memory_2, mask, ...)
```

不得写成：

```python
memory_1 = encoder_1(memory_0)
memory_2 = encoder_2(memory_0)
memory_3 = encoder_3(memory_0)
```

也不得直接把三个 Encoder 的输出相加或平均后交给所有 PGCA。

### 3.2 保存三阶段 Memory

Hierarchical 模式必须在一次前向中得到：

```python
stage_memories = (M1, M2, M3)
feature_mask = mask
```

每个 PGCA 层按照配置选取自己的 Memory。

例如：

```python
memory = stage_memories[stage_index_for_layer(layer_idx)]
```

不能将全部 PGCA 都传入 M3。

请考虑现有 `_prepare_feature_memory()` 和 Decoder `forward()` 接口如何以最小改动兼容两种模式。

Shared 模式仍然保持单份 Memory 的既有逻辑，避免增加不必要的数据结构转换和开销。

### 3.3 明确执行顺序

由于三个编码器只依赖静态音形特征和前一级 Memory，而不依赖 Qwen3 当前层的语义 Hidden State，所以允许在主干 Decoder 循环之前计算完 M1、M2、M3。

不要求真的在执行到 Qwen3 第 9、12、16 层时才运行对应编码器。

但是必须保证：

- 编码器级联方向正确。
- 每个主干 PGCA 层使用配置指定的 Memory。
- 不跨样本串用 Memory。
- 不产生未来 Token 泄漏。
- 不因为 Memory 提前计算改变因果语义。

## 4. 初始化与旧权重迁移

这是本次改造的关键要求。

需要同时支持：

### 4.1 从 Char 模型构建新的 Hierarchical 模型

支持直接从 Char 基座构建完整 Hierarchical-12L 模型。

需要为 C1、C2、C3 分别初始化参数，并保留现有 PGCA `zero-init-output` 等构建约定。

新增命令行参数，建议类似：

```bash
python -m src.pgca.build_pgca \
  --model-path models/Qwen3-1.7B-Base-Char \
  --output-path models/Qwen3-1.7B-Base-PGCA-Hierarchical-12L \
  --pgca-layers 9,10,11,12,13,14,15,16 \
  --context-encoder \
  --context-mode hierarchical \
  --context-stage-starts 9,12,16 \
  --auxiliary-heads \
  --zero-init-output
```

上述 `--context-mode` 和 `--context-stage-starts` 为本次预期新增的接口，不是声称当前已经存在。

### 4.2 从已有 Shared-4L checkpoint 升级

额外提供明确的 checkpoint 转换入口，支持将已训练的 Shared-4L 完整 checkpoint 升级为 Hierarchical-12L。

要求：

1. 保留全部主干权重。
2. 保留原始 PGCA 各层参数。
3. 保留音形 Embedding、Feature Index、Token Type Index。
4. 保留原有四层编码器作为 C1，并完整复制其参数。
5. 新增 C2、C3。
6. 不修改原 checkpoint 目录，结果另存。
7. 记录源 checkpoint、目标结构、参数迁移及初始化报告。

**新增 C2、C3 应采用初始恒等残差策略。**

现有编码器输出形式为：

\[
M_{\mathrm{out}}=M_{\mathrm{in}}+\Delta M
\]

因此可以将新增 C2、C3 的 `output_proj` 零初始化，使：

\[
C_2(M_1)=M_1
\]

\[
C_3(M_2)=M_2
\]

初始化完成时：

\[
M_1=M_2=M_3
\]

这是期望的转换不变量。

在相同输入、相同 PGCA 层位、Eval 模式、无 Dropout 干扰的条件下，升级前后模型输出 Logits 应达到严格数值一致或明确的浮点容差。

注意：

- C1 使用原 checkpoint 权重。
- C2/C3 内部其他参数可以按现有初始化规则初始化。
- 不能将 C2/C3 永久冻结。
- 零初始化只用于起始状态，后续必须能够通过反向传播更新。
- 不得为了让 checkpoint 强制加载成功而全面设置 `strict=False` 并忽略缺失键。

旧 checkpoint 加载与新结构升级必须区分：

- 正常加载旧 Shared-4L：严格兼容旧行为。
- 显式升级：只允许新增 C2/C3 对应的已知参数缺失，初始化后保存完整的新 checkpoint。
- 正常加载新 Hierarchical checkpoint：新编码器参数不得缺失或形状不匹配。

请检查现有 `from_pretrained()` 的必需权重检查逻辑，确保不会误将升级过程中预期新增的参数缺失当成损坏，也不能让正常加载静默忽略真实的权重损坏。

## 5. KV Cache 与自回归推理

这是最高风险的实现部分，不能只让 `use_cache=False` 的前向通过。

当前 `CausalFeatureEncoder` 的每个 Transformer Block 使用自己的 `layer_idx` 更新 DynamicCache。

三个编码器的内部层号都会是 0–3。

**禁止让三个编码器复用同一个 DynamicCache 的相同 Layer Index。**

必须为 C1、C2、C3 提供逻辑独立的 KV Cache。

例如：

```text
CompositeCache
├── backbone_cache
└── feature_caches
    ├── C1_cache
    ├── C2_cache
    └── C3_cache
```

可以根据当前 Transformers `DynamicCache` 接口选取更合适的具体结构，但必须满足以下约束。

### 5.1 Prefill

对于一次长度为 T 的 Prefill：

- 主干 Cache 长度为 T。
- C1、C2、C3 各自的 Cache 长度也为 T。
- 三组 Cache 不发生覆盖。
- 各 Encoder 使用一致的 `position_ids` 和有效因果 Mask。

### 5.2 Decode

每生成一个新 Token：

- 主干和三个音形编码器分别追加当前 Token 的 KV。
- 三个音形编码器的有效上下文长度与主干保持一致。
- M2 的当前 Token 输入来自当前 Token 的 M1。
- M3 的当前 Token 输入来自当前 Token 的 M2。
- 不重新计算完整前缀，除非明确采用非缓存路径进行一致性测试。

### 5.3 Cache 生命周期

检查并适配：

- `reorder_cache`
- `batch_repeat_interleave`
- `batch_select_indices`
- `crop`
- `reset`
- 已有 Cache 的继续解码
- Padding 和 Position IDs
- Beam Search 所需的缓存重排（现有实现支持的范围内）

Shared-4L 旧 Cache 逻辑不得被破坏。

如果某个上游接口存在无法安全支持的情况，应明确拒绝并报告原因，不能静默输出错误结果。

## 6. 训练兼容性

必须审查并验证现有训练流程。

### 6.1 Phase1

沿用原设计：

- 主干训练。
- PGCA 和所有音形编码器冻结。
- 不启用音形辅助损失。

C1/C2/C3 均属于需要冻结的新增分支。

### 6.2 Phase2-A

沿用原设计：

- 冻结主干。
- 训练音形 Embedding、PGCA、全部三个编码器和辅助预测头。

不能因为现有正则只匹配 `context_encoder` 而遗漏 C2/C3。

### 6.3 Phase2-B

沿用原设计：

- 主干和完整音形分支联合训练。
- 新增编码器参数采用音形分支学习率。
- 不误分到主干学习率组。
- CLM 和音形辅助损失保持原有计算口径。

重点检查：

```text
scripts/train/cpt/plugin.py
```

中的：

```python
is_branch_parameter()
branch_gradient_group()
```

以及训练参数冻结和可训练性校验。

建议增加 C1、C2、C3 的独立梯度范数统计，以便后续分析级联训练中的梯度流动情况。

不得改动已有 CLM/Phonetic/Glyph 损失公式和 Token 分母。

### 6.4 General-SFT 与 Downstream LoRA

检查当前：

```text
scripts/train/instruction/
scripts/train/downstream/
```

确保新结构可以由 ms-swift 正确加载。

如果启用 all-linear LoRA：

- 需要覆盖三个编码器中的对应 Linear 参数。
- 原有 PGCA 层继续参与 LoRA。
- 辅助头继续遵循现有关闭和冻结策略。
- 可训练参数校验不能只检查 C1 而漏掉 C2/C3。

不能为了适配本次改造，顺带改变现有指令数据、SFT Loss、EOS 或其他无关训练行为。

## 7. 模型构建、序列化与运行时兼容

需要保证构建出来的 Hierarchical-12L 模型仍是自包含的 Hugging Face checkpoint。

必须验证：

- `config.json` 正确记录模式与阶段边界。
- 保存并重新加载后结构不变。
- 所有新增编码器权重完整保存。
- Tokenizer、Special Tokens、Feature Index 不被改变。
- `auto_map` 与 `trust_remote_code=True` 加载正常。
- 构建时复制的运行时源码包含最新改造。
- 模型重新加载后的输出与保存前一致。
- 现有 Shared-4L checkpoint 继续加载正常。

重点检查：

```text
src/pgca/build_pgca.py
src/pgca/validate_pgca.py
```

以及其运行时源码打包机制。

新增模型的参数量、分支参数量和三个编码器各自参数量必须写入构建报告。

## 8. 测试要求

请增加可重复执行的自动化测试。优先使用小型合成 Qwen3/PGCA 配置在 CPU 上验证，真实 1.7B checkpoint 的测试作为具备权重和 GPU 时的补充。

### 8.1 结构测试

验证：

- Hierarchical 模式包含三个完整四层编码器。
- C1、C2、C3 的参数相互独立。
- 输入输出形状正确。
- 非汉字、缺失拼音及 Padding Mask 正确处理。
- 第 9–11 层使用 M1。
- 第 12–15 层使用 M2。
- 第 16 层使用 M3。
- 无 PGCA 的层不引入额外融合。
- 非法阶段配置能够拒绝加载。

不要仅仅测试最终输出形状，要通过 Hook、显式捕获或可靠的等效方法确认每个 PGCA 实际使用了正确阶段的 Memory。

### 8.2 级联与梯度测试

验证：

- M2 的输入确实是 M1。
- M3 的输入确实是 M2。
- 不存在 `detach()`。
- C1/C2/C3 在适当的非退化训练状态下均能获得梯度。
- Phase2-A 的冻结范围正确。
- Phase2-B 的学习率分组正确。
- 不产生 NaN/Inf。

注意初始零投影可能导致部分内部参数首步梯度为零。测试应区分预期初始化行为和永久梯度断流，不应要求所有内部参数在第一个训练 Step 就一定非零。

### 8.3 Checkpoint 迁移测试

验证：

1. Shared-4L 升级为 Hierarchical-12L。
2. 升级时 C1 权重精确保留。
3. C2/C3 初始近似恒等。
4. 未训练的新模型与原模型 Logits 一致。
5. 保存后再次加载仍然一致。
6. 训练更新后三级 Memory 能产生差异。
7. 对原 Shared-4L checkpoint 的加载没有回归。

### 8.4 KV Cache 测试

对同一组真实 Token 输入，分别执行：

- 完整序列 `use_cache=False`
- Prefill + 逐 Token Decode `use_cache=True`

比较：

- 所有预测位置的 Logits；
- Top-1 一致率；
- 最大绝对误差；
- 三个音形 Cache 的长度；
- 主干 Cache 的长度；
- Greedy 自由生成结果。

至少包含：

- 单序列；
- 多 Token 增量解码；
- 汉字与非汉字混合；
- 多音字；
- Padding；
- Cache 重排及裁剪。

根据实际精度设置合理的数值容差，但不能以放宽容差掩盖系统性差异。

### 8.5 训练和验证冒烟

至少完成：

- Shared-4L 原有测试；
- Hierarchical-12L 从头构建与加载；
- Hierarchical-12L Phase1 短跑；
- Hierarchical-12L Phase2-A/B 各少量 Step；
- 下游模型/LoRA 加载与前向；
- HF Greedy 推理；
- 构建与验证脚本。

测试资源不足时，明确区分：

- 已执行并通过；
- 已执行但失败；
- 因缺少模型权重、GPU 或数据无法执行。

不能把未执行的真实训练测试描述为通过。

## 11. 交付内容

完成改造后提供一份正式工程报告，建议结构如下。

### 11.1 实现结论

- 是否完成 Hierarchical-12L；
- 新旧模式是否都可用；
- 旧 checkpoint 是否兼容；
- KV Cache 是否通过验证；
- 当前剩余的限制。

### 11.2 架构与关键实现

说明：

- C1 → C2 → C3 的数据流；
- 各 PGCA 层的 Memory 路由；
- 旧权重迁移与恒等初始化；
- Cache 组织；
- 参数冻结与优化器分组；
- 修改的文件和关键接口。

重点解释设计决策，不要仅罗列代码改动。

### 11.3 可执行命令

必须给出：

1. 从 Char 构建 Hierarchical-12L 的完整命令。
2. 从 Shared-4L checkpoint 升级的完整命令。
3. 模型验证命令。
4. Phase1、Phase2-A、Phase2-B 的使用方式。
5. 已实现的单元测试和冒烟执行命令。

路径使用示例或环境变量，不虚构本机存在的模型和数据路径。

### 11.4 测试结果

逐项报告：

| 测试项 | 是否执行 | 结果 | 证据 |
|---|---|---|---|
| Shared-4L 兼容 | | | |
| 三阶段 Memory 路由 | | | |
| 级联梯度 | | | |
| Checkpoint 升级等价性 | | | |
| Save/Load | | | |
| Cache On/Off | | | |
| Phase1 | | | |
| Phase2-A | | | |
| Phase2-B | | | |
| LoRA 加载 | | | |

提供具体测试命令、运行结果和必要的数值误差。

### 11.5 后续实验建议

仅简要指出需要进行的 Shared-4L 与 Hierarchical-12L 对照实验，包括同一起点、训练预算、模型评价和效率对比。

不要在没有实测数据的情况下宣称 Hierarchical-12L 更优。

## 12. 最终验收条件

只有全部满足以下要求，才能宣布源码改造完成：

- [ ] 已实现三个四层编码器的真正串行级联。
- [ ] M1、M2、M3 根据配置供不同 PGCA 区域使用。
- [ ] 旧 Shared-4L 架构与 checkpoint 兼容。
- [ ] 新增配置可保存、加载并经过合法性验证。
- [ ] 支持从 Char 构建新的 Hierarchical-12L。
- [ ] 支持从已有 Shared-4L checkpoint 显式升级。
- [ ] 升级初始状态与原模型 Logits 一致。
- [ ] 三个编码器在训练时均可按预期更新。
- [ ] Phase1、Phase2-A/B 参数冻结和学习率分组正确。
- [ ] General-SFT 与下游 LoRA 可识别全部新增模块。
- [ ] 主干及三个音形 KV Cache 在增量生成时保持同步。
- [ ] Cache On/Off 数值一致性得到验证。
- [ ] HF 模型自包含导出、加载及模型校验通过。
- [ ] 相关测试与文档同步更新。
- [ ] 未修改无关训练和评测行为。
- [ ] 工程报告包含实际验证证据和剩余限制。

请开始检查源码并实施，完成后按上述交付结构汇报。