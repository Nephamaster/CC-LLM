# PGCA Backbone Implementation Plan

## 1. 目标与边界

本阶段实现 Backbone: PGCA 音形交叉注意力分支，将已完成的 `FeatureMemoryBuilder` 接入 Qwen3 decoder block。

核心目标：

- 保持 Qwen3 原始 causal self-attention、MLP、RoPE、KV cache 和 generation 流程不变。
- 在指定中间层加入 position-wise cross-attention：每个文本 token 只查询自身对应的 feature memory。
- PGCA 与原 self-attention 并行计算：二者使用同一个 `RMSNorm(H)` 输入，输出在 residual 路径汇合后再进入 MLP。
- PGCA 输出通过可学习 gate 融入 residual，gate 初始化为 0，使初始行为接近 Char baseline。
- 默认不使用图形图像编码，仅使用当前 embedding 阶段产出的拼音、声母、韵母、声调、笔画数、部首笔画、结构特征。

非目标：

- 不重训 tokenizer。
- 不修改 semantic embedding / lm_head 迁移逻辑。
- 不实现 glyph image encoder。
- 不把 feature token 纳入 `lm_head` 或普通文本序列。

## 2. 依据

本地当前模型与产物：

- `models/Qwen3-1.7B-Base-Char/config.json`
  - `vocab_size=154019`
  - `hidden_size=2048`
  - `num_hidden_layers=28`
  - `num_attention_heads=16`
  - `num_key_value_heads=8`
  - `head_dim=128`
  - `max_position_embeddings=32768`
- `models/Qwen3-1.7B-Base-Char/embedding_config.json`
  - `d_model=2048`
  - `d_feat=256`
  - `max_pinyin_per_char=8`
  - `num_feature_slots=9`
  - `use_glyph_image=false`
- `FeatureMemoryBuilder.forward(input_ids)` 已返回：
  - `feature_memory`: `(batch, seq, 9, 2048)`
  - `feature_mask`: `(batch, seq, 9)`

外部资料依据：

- Qwen3-1.7B-Base 官方模型卡说明该模型为 causal LM，28 层，GQA 为 16 个 Q heads / 8 个 KV heads，上下文长度 32768。
- Hugging Face Qwen3 文档说明 `Qwen3Config` 使用 decoder-only 参数体系，包含 GQA、RMSNorm、RoPE、sliding window、cache 等接口。
- ChineseBERT 等工作表明汉字 glyph 与 pinyin 信息能增强中文表征；本项目只采用其“音形信息作为辅助表征”的思想，不照搬其 embedding fusion 方式。
- PaLM 类 parallel layer、Flamingo gated cross-attention 等结构说明外部 memory 分支可以作为 gated residual branch 注入 decoder block。PGCA 的 feature memory 是 per-position 局部查询，不跨 token，因此不破坏 causal LM 假设。

## 3. 模块设计

新增目录：

```text
src/pgca/
  __init__.py
  config.py
  attention.py
  build_pgca.py
  validate_pgca.py
```

新增 Qwen3 PGCA 文件：

```text
src/configuration_qwen3_pgca.py
src/modeling_qwen3_pgca.py
```

实现基准：

- `src/modeling_qwen3_hf.py`
- `src/configuration_qwen3_hf.py`
- `src/modular_qwen3.py`

### 3.1 `PGCAConfig`

文件：`src/pgca/config.py`

建议字段：

```python
@dataclass
class PGCAConfig:
    use_pgca: bool = True
    pgca_layers: list[int] | None = None
    pgca_num_attention_heads: int = 16
    pgca_num_key_value_heads: int = 8
    pgca_head_dim: int = 128
    pgca_gate_init: float = 0.0
    pgca_dropout: float = 0.0
    pgca_feature_slots: int = 9
    pgca_feature_hidden_size: int = 2048
    pgca_feature_embedding_dim: int = 256
    pgca_max_pinyin_per_char: int = 8
    pgca_feature_vocab_sizes: dict[str, int] | None = None
    pgca_use_glyph_image: bool = False
```

默认层选择：

- Qwen3-1.7B-Base 共 28 层。
- 未传 `--pgca-layers` 时，代码默认插入中间 1/3 的 9 层：`[9, 10, 11, 12, 13, 14, 15, 16, 17]`。
- 本轮实验通过命令显式使用 6 层：`[11, 12, 13, 14, 15, 16]`。

配置落点：

- 基于 `src/configuration_qwen3_hf.py` 新增 `src/configuration_qwen3_pgca.py`。

### 3.2 `PGCACrossAttention`

文件：`src/pgca/attention.py`

输入：

```python
hidden_states: FloatTensor          # RMSNorm(H), (batch, seq, hidden)
feature_memory: FloatTensor         # (batch, seq, slots, hidden)
feature_mask: BoolTensor            # (batch, seq, slots)
```

计算：

```text
Q = q_norm(q_proj(hidden_states)).view(batch, seq, q_heads, head_dim)
K = k_norm(k_proj(feature_memory)).view(batch, seq, slots, kv_heads, head_dim)
V = v_proj(feature_memory).view(batch, seq, slots, kv_heads, head_dim)

repeat K/V from kv_heads to q_heads
score[b, s, h, m] = dot(Q[b, s, h], K[b, s, h, m]) / sqrt(head_dim)
score masked by feature_mask
Z[b, s, h] = softmax(score) @ V[b, s, h]
output = gate * o_proj(Z.reshape(batch, seq, hidden))
```

并行融合：

```text
normed = RMSNorm(H)
self_attn_out = SelfAttention(normed)
pgca_out = PGCA(normed, feature_memory, feature_mask)

H_mid = H + self_attn_out + pgca_out
H_next = H_mid + MLP(RMSNorm(H_mid))
```

实现要点：

- 使用 Qwen3 同样的 `head_dim=128`、GQA 结构、`Qwen3RMSNorm` head-wise q/k norm。
- PGCA 不使用 RoPE，因为 feature slots 不是时间序列位置。
- 对非汉字 token，`feature_mask` 全 0；PGCA 输出应为 0，避免影响原始非汉字建模。
- 对全 mask 行，masked score 使用当前 dtype 的有限最小值，softmax 后再将 mask 位置清零，避免 NaN。
- `pgca_gate` 使用标量参数，初始为 0，使初始行为接近 Char baseline。

### 3.3 Qwen3 接入

实现方案：基于三个 HF 官方代码副本新增独立的 PGCA 模型类型。

1. `src/configuration_qwen3_pgca.py`
   - 定义 `Qwen3PGCAConfig(PreTrainedConfig)`。
   - 使用唯一的 `model_type="qwen3_pgca"`，避免被 Transformers 静默识别为原生 Qwen3。
   - 保留 Qwen3 原始字段，并加入全部 `pgca_*` 和 feature vocab 配置。
   - 输出配置写入 `architectures=["Qwen3PGCAForCausalLM"]` 与完整 `auto_map`。

2. `src/modeling_qwen3_pgca.py`
   - 定义 `Qwen3PGCAPreTrainedModel`、`Qwen3PGCAModel` 和 `Qwen3PGCAForCausalLM`。
   - `config_class=Qwen3PGCAConfig`，`base_model_prefix="model"`，支持 gradient checkpointing 和 Transformers attention backend。

3. `Qwen3PGCADecoderLayer`
   - 仅当 `layer_idx in config.pgca_layers` 时创建 `self.pgca_attn`。
   - self-attention 和 PGCA 使用同一个 `input_layernorm(hidden_states)` 结果并行计算：

```python
residual = hidden_states
normed_hidden_states = self.input_layernorm(hidden_states)
self_attn_out, _ = self.self_attn(normed_hidden_states, ...)
pgca_out = self.pgca_attn(normed_hidden_states, feature_memory, feature_mask)
hidden_states = residual + self_attn_out + pgca_out
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
```

4. `Qwen3PGCAModel.__init__`
   - 只根据 config 构造 `PhoneticGlyphFeatureEmbedding` 和空的 `FeatureMemoryBuilder`。
   - 不读取 `config.name_or_path`、`embedding_config.json` 或 `features/feature_index.pt`。
   - feature index 作为 persistent buffer 从模型权重恢复。

5. `Qwen3PGCAModel.forward`
   - feature 输入优先级：显式 `feature_memory/feature_mask`，其次 `feature_ids`，最后根据 `input_ids` 自动查表。
   - generation decode 阶段只查当前 step token，不需要额外 feature cache。

6. `Qwen3PGCAForCausalLM.forward`
   - 透传 `feature_memory/feature_mask/feature_ids`。
   - 保持 `logits_to_keep`、loss 和 `past_key_values` 原逻辑。

7. 严格加载
   - `from_pretrained()` 完成后检查 feature index 的 `missing_keys`、ready 状态、行数和有效汉字特征。
   - 完整性检查只发生在加载阶段；forward 内不执行 `.item()` 或数据依赖 Python 分支，以兼容 vLLM/`torch.compile`。

## 4. 权重与配置迁移

执行：

```bash
python -m src.pgca.build_pgca \
  --model-path models/Qwen3-1.7B-Base-Char \
  --output-path models/Qwen3-1.7B-Base-Char-PGCA \
  --pgca-layers 11,12,13,14,15,16
```

流程：

1. 复制 Char 模型到 PGCA 输出目录，并读取迁移输入 `embedding_config.json` 与 `features/feature_index.pt`。
2. 将 embedding 配置合并进 `Qwen3PGCAConfig`，写入：
   - `model_type="qwen3_pgca"`
   - `architectures=["Qwen3PGCAForCausalLM"]`
   - `auto_map`
   - PGCA 层、头维度、feature vocab sizes 和 feature index SHA256。
3. 从 Char checkpoint 加载原主干参数；本次迁移显式允许 PGCA 新参数和 feature buffers 缺失。
4. 写入并验证 feature index，初始化全部 PGCA 新参数，并将所有 gate 重置为 `pgca_gate_init=0.0`。
5. `save_pretrained()` 后复制扁平化 remote code，确保不存在 `src.*` 导入。
6. 删除最终仓库中的 `embedding_config.json`、`features/` 及旧 runtime 目录；feature index 已作为 persistent buffer 进入模型权重。
7. 生成 `pgca_migration_report.json`，记录配置映射、feature index SHA256、加载信息和 PGCA 初始化检查。

构建脚本不自动运行验证。构建完成后单独执行：

```bash
python -m src.pgca.validate_pgca \
  --model-path models/Qwen3-1.7B-Base-Char-PGCA
```

验证结果写入 `models/Qwen3-1.7B-Base-Char-PGCA/reports/pgca_validation_report.json`。

## 5. 验证设计

`src/pgca/validate_pgca.py` 当前验证：

1. 模型仓库
   - `model_type="qwen3_pgca"`
   - `architectures` 与 `auto_map` 正确
   - remote-code 文件完整
   - 最终仓库不再依赖外部 `embedding_config.json` 和 `features/feature_index.pt`

2. 配置与模块
   - PGCA 层均在合法范围内，实际插入层与配置一致
   - hidden/head/feature slot 维度一致
   - gate 与 `pgca_gate_init` 一致
   - PGCA 参数不存在 meta、NaN 或 Inf

3. feature memory
   - persistent feature buffers 已进入 checkpoint
   - feature index ready，行数等于 `vocab_size`
   - 输出形状为 `(batch, seq, 9, 2048)` 和 `(batch, seq, 9)`
   - 测试文本包含汉字并能产生有效 feature slot

4. forward
   - 默认使用 `中国ABC`，也可通过 `--text` 指定输入
   - logits shape 为 `(batch, seq, 154019)`
   - loss 为有限值

当前自动验证不比较 Char/PGCA logits，也不单独测试 KV cache。gate 为 0 的初始等价性可作为额外实验检查，但不应写成当前 validation report 已覆盖的指标。

## 6. 训练控制建议

阶段性参数冻结建议：

- Phase 1：保持 PGCA disabled 或 gate=0 且不训练 PGCA，只训练 `E_sem/lm_head`。
- Phase 2：启用 PGCA，先只训练：
  - `feature_embedding`
  - `pgca_attn`
  - `pgca_gate`
  - 可选 top/mid 层 norm
- 稳定后再全参 CPT。

学习率建议：

- PGCA 新参数学习率可高于 backbone，例如 backbone lr 的 2 到 5 倍。
- `pgca_gate` 可单独较大学习率，但要监控 gate 是否快速放大导致 loss spike。

## 7. 风险与处理

- Transformers/ms-swift 加载：最终目录是自包含 remote-code 模型仓库，使用 `model_type="qwen3_pgca"`、`auto_map` 和 `trust_remote_code=True` 加载，避免落入原生 Qwen3。
- feature index 完整性：加载完成后严格检查必需 buffers；缺失时直接失败，不能静默退化为全零 PGCA。
- vLLM 编译：forward 路径不读取 `feature_index_ready.item()`，避免 `torch.compile` 的数据依赖控制流错误。
- generation：PGCA 是 position-wise，仅需当前 token 的 feature lookup，不增加历史 feature cache。
- 全 mask softmax：masked score 使用有限最小值，softmax 后再次清零 mask，确保非汉字位置输出为零且不产生 NaN。
- 输出目录固定为 `models/Qwen3-1.7B-Base-Char-PGCA`，避免覆盖 Char baseline。

## 8. 实施顺序

1. 实现 `src/pgca/config.py` 和 `src/pgca/attention.py`。
2. 新增独立的 `src/configuration_qwen3_pgca.py` 与 `src/modeling_qwen3_pgca.py`。
3. 在 decoder layer 中并行接入 PGCA，并让 model/CausalLM forward 透传 feature 参数。
4. 将 embedding 配置合并进模型 config，并把 feature index 注册为 persistent buffers。
5. 实现 `src/pgca/build_pgca.py`，完成权重迁移、初始化、remote-code 打包和旧外部依赖清理。
6. 实现 `src/pgca/validate_pgca.py`，构建完成后独立执行验证。
7. 生成 `pgca_migration_report.json` 与 `reports/pgca_validation_report.json`。

## 9. 预期交付产物

源码：

```text
src/pgca/__init__.py
src/pgca/config.py
src/pgca/attention.py
src/pgca/build_pgca.py
src/pgca/validate_pgca.py
src/configuration_qwen3_pgca.py
src/modeling_qwen3_pgca.py
```

最终模型仓库：

```text
models/Qwen3-1.7B-Base-Char-PGCA/
  __init__.py
  config.json
  generation_config.json
  tokenizer files
  configuration_qwen3_pgca.py
  modeling_qwen3_pgca.py
  embedding_config.py
  feature_embedding.py
  feature_memory.py
  pgca_attention.py
  pgca_config.py
  model*.safetensors
  model.safetensors.index.json       # 分片时存在
  pgca_migration_report.json
  reports/pgca_validation_report.json
```

`embedding_config.json` 和 `features/feature_index.pt` 只作为迁移输入，不保留在最终 PGCA 仓库中。

最小成功标准：

- PGCA 模型能通过 Transformers `AutoModelForCausalLM` 和 remote code 从完整 checkpoint 加载。
- 指定层包含 PGCA 参数，gate 初始为 0。
- feature index 已进入模型权重，缺失时加载直接失败。
- 中文/英文混合输入 forward 通过，loss 有限。
- `bos/eos/pad` 均在合法范围内。
- logits 最后一维为 `154019`。
