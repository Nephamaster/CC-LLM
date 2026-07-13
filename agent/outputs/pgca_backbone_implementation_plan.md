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
  - `vocab_size=147688`
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
src/configuration_qwen3.py
src/modeling_qwen3.py
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
    pgca_build_features_in_model: bool = True
```

默认层选择：

- Qwen3-1.7B-Base 共 28 层。
- 默认插入中间 1/3 的 6 层：`[11, 12, 13, 14, 15, 16]`。
- 用户需要更轻量实验时可改为 `[10, 14, 18, 22]`。

配置落点：

- 基于 `src/configuration_qwen3_hf.py` 新增 `src/configuration_qwen3.py`。

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
- 对全 mask 行，不能让 softmax 产生 NaN；实现时先把 masked score 置为大负值，softmax 后再乘 mask，并对分母做 clamp。
- `pgca_gate` 使用标量参数，初始为 0，使初始行为接近 Char baseline。

### 3.3 Qwen3 接入

实现方案：基于三个 HF 官方代码副本新增 PGCA 版本。

1. `src/configuration_qwen3.py`
   - 从 `src/configuration_qwen3_hf.py` 派生。
   - 保留 Qwen3 原始字段。
   - 新增 `pgca_*` 字段。
   - `model_type` 可保持 `qwen3`，但 `architectures` 在输出配置中写为 `Qwen3PGCAForCausalLM`。

2. `src/modeling_qwen3.py`
   - 从 `src/modeling_qwen3_hf.py` 派生。
   - 类名建议：
     - `Qwen3PGCAPreTrainedModel`
     - `Qwen3PGCAModel`
     - `Qwen3PGCAForCausalLM`

3. `Qwen3DecoderLayer.__init__`
   - 若 `layer_idx in config.pgca_layers`，创建 `self.pgca_attn = PGCACrossAttention(config)`。
   - 否则为 `None`。

4. `Qwen3DecoderLayer.forward`
   - 新增参数：
     - `feature_memory: Optional[torch.Tensor] = None`
     - `feature_mask: Optional[torch.Tensor] = None`
   - self-attention 和 PGCA 使用同一个 `input_layernorm(hidden_states)` 结果并行计算：

```python
residual = hidden_states
normed_hidden_states = self.input_layernorm(hidden_states)

self_attn_out, _ = self.self_attn(
    hidden_states=normed_hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,
    use_cache=use_cache,
    cache_position=cache_position,
    position_embeddings=position_embeddings,
    **kwargs,
)

pgca_out = 0
if self.pgca_attn is not None and feature_memory is not None:
    pgca_out = self.pgca_attn(normed_hidden_states, feature_memory, feature_mask)

hidden_states = residual + self_attn_out + pgca_out
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

5. `Qwen3PGCAModel.__init__`
   - 若 `config.pgca_build_features_in_model=True`，根据模型目录加载：
     - `embedding_config.json`
     - `features/feature_index.pt`
   - 构造：
     - `PhoneticGlyphFeatureEmbedding`
     - `FeatureMemoryBuilder`

6. `Qwen3PGCAModel.forward`
   - 新增参数：
     - `feature_memory=None`
     - `feature_mask=None`
     - `feature_ids=None`
   - 优先级：
     - 显式传入 `feature_memory/feature_mask`：直接使用。
     - 否则若 `input_ids` 存在且 `feature_memory_builder` 存在：自动构造。
     - 否则 PGCA 分支跳过。
   - generation decode 阶段通常 `input_ids` 为当前 step token，自动 lookup 可正常工作，不需要额外 KV cache。

7. `Qwen3PGCAForCausalLM.forward`
   - 透传 `feature_memory/feature_mask/feature_ids`。
   - 保持 `logits_to_keep`、loss、past_key_values 原逻辑不变。

## 4. 权重与配置迁移

新增脚本：

```text
src/pgca/build_pgca.py
```

功能：

```bash
python -m src.pgca.build_pgca \
  --model-path models/Qwen3-1.7B-Base-Char \
  --output-path models/Qwen3-1.7B-Base-Char-PGCA \
  --pgca-layers 11,12,13,14,15,16
```

流程：

1. 复制 char 模型目录到 PGCA 输出目录。
2. 更新 `config.json`：
   - `architectures=["Qwen3PGCAForCausalLM"]`
   - `use_pgca=true`
   - `pgca_layers=[11,12,13,14,15,16]`
   - `pgca_num_attention_heads=16`
   - `pgca_num_key_value_heads=8`
   - `pgca_head_dim=128`
   - `pgca_feature_slots=9`
   - `pgca_feature_hidden_size=2048`
   - `pgca_gate_init=0.0`
3. 加载 char 权重到 PGCA 模型，允许 missing PGCA 权重。
4. 新增 PGCA 参数按 `initializer_range=0.02` 初始化，gate 为 0。
5. 保存模型和报告：
   - `pgca_migration_report.json`
   - `reports/pgca_validation_report.json`

报告核心字段：

- `base_model_path`
- `output_model_path`
- `pgca_layers`
- `loaded_base_parameter_count`
- `missing_pgca_parameter_count`
- `unexpected_keys`
- `pgca_gate_values`
- `feature_memory_shape`
- `forward_passed`

## 5. 验证设计

新增：

```text
src/pgca/validate_pgca.py
```

最低验证项：

1. 配置一致性
   - `hidden_size == pgca_feature_hidden_size == 2048`
   - `num_attention_heads == pgca_num_attention_heads == 16`
   - `num_key_value_heads == pgca_num_key_value_heads == 8`
   - `head_dim == pgca_head_dim == 128`
   - `pgca_layers` 均在 `[0, 27]`

2. 参数存在性
   - 仅指定层包含 `pgca_attn`
   - 非 PGCA 层不创建 PGCA 参数
   - gate 初始值为 0

3. feature memory
   - `FeatureMemoryBuilder(input_ids)` 输出 `(batch, seq, 9, 2048)` 和 `(batch, seq, 9)`
   - 汉字 token 至少有一个 feature slot 有效
   - 非汉字 token feature mask 全 0

4. forward
   - 中文、英文、混合输入能 forward
   - logits shape 为 `(batch, seq, 147688)`
   - `past_key_values` 可返回
   - loss 计算可运行

5. 初始等价性
   - gate 为 0 时，同一输入下 PGCA 模型和 Char 模型 logits 应接近。
   - 该检查要求 PGCA 与 self-attention 并行但 gate 为 0，保证新增分支不会改变初始主干输出。
   - 允许极小数值误差；推荐阈值 `max_abs_diff < 1e-5`，若 dtype 为 bf16 可放宽到 `1e-2`。

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

- `AutoModelForCausalLM` 自动加载：若要通过 Transformers auto class 加载，需要注册本地 modeling 或使用 `trust_remote_code` 路径；本阶段优先保证 `from src.modeling_qwen3 import Qwen3PGCAForCausalLM` 可用。
- generation 时 feature 自动构造：decode step 只有新 token 的 `input_ids`，PGCA 是 position-wise，不依赖历史 feature，因此不需要 feature cache。
- 全 mask softmax：必须显式处理，否则非汉字 token 会产生 NaN。
- 输出目录：PGCA 建议输出到 `models/Qwen3-1.7B-Base-Char-PGCA`，避免覆盖已验证的 Char baseline。

## 8. 实施顺序

1. 新增 `src/pgca/config.py` 和 `src/pgca/attention.py`。
2. 基于 `src/configuration_qwen3_hf.py` 新增 `src/configuration_qwen3.py`，只加入 `pgca_*` 字段。
3. 基于 `src/modeling_qwen3_hf.py` 和 `src/modular_qwen3.py` 新增 `src/modeling_qwen3.py`：
   - decoder layer 并行接入 PGCA
   - model forward 自动构造或接收 feature memory
   - CausalLM forward 透传 feature 参数
4. 新增 `src/pgca/build_pgca.py` 执行门面。
5. 新增 `src/pgca/validate_pgca.py`。
6. 生成 `pgca_migration_report.json` 与 `reports/pgca_validation_report.json`。

## 9. 预期交付产物

代码：

```text
src/pgca/__init__.py
src/pgca/config.py
src/pgca/attention.py
src/pgca/build_pgca.py
src/pgca/validate_pgca.py
src/configuration_qwen3.py
src/modeling_qwen3.py
```

执行后模型目录：

```text
models/Qwen3-1.7B-Base-Char-PGCA/
  config.json
  generation_config.json
  embedding_config.json
  pgca_migration_report.json
  reports/pgca_validation_report.json
  features/
  tokenizer files
  model weights
```

最小成功标准：

- PGCA 模型能从 Char 权重加载。
- 指定层包含 PGCA 参数，gate 初始为 0。
- 中文/英文/混合输入 forward 通过。
- `bos/eos/pad` 仍不越界。
- logits 维度为 `147688`。
- gate=0 时输出与 Char baseline 接近。
