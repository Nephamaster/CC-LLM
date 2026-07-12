# Embedding 设计模块实现方案

## 1. 目标

本阶段实现 markdown 中的第 3 部分 Embedding 设计，面向 `Qwen3-1.7B-Base` 的 Char tokenizer 版本。

目标分两层：

1. 语义 embedding：构建新词表下的 `E_sem` 和 `lm_head`，最大限度继承 Qwen3 原语义空间。
2. 音形 feature embedding：根据 `feature_index.pt/jsonl` 构造每个 token 位置的 feature memory，为后续 PGCA 使用。

本阶段不实现 PGCA block 本体，但要把 feature memory 的张量接口设计到位。

## 2. 资料依据

- Qwen3 Technical Report 说明 Qwen3 是 dense/MoE 混合规模家族，覆盖 0.6B 到 235B，并强调多语言能力和开源可复现；本实验选用 dense 的 `Qwen3-1.7B-Base`。参考：https://arxiv.org/abs/2505.09388
- `Qwen3-1.7B-Base` 当前本地迁移报告显示：`hidden_size=2048`，原模型 `old_vocab_size=151936`，新词表 `new_vocab_size=147688`，`tie_word_embeddings=true`，输入输出 embedding 需要保持 tied。
- 词表扩展初始化研究表明，新 token 不宜随意随机初始化；用旧 tokenizer 分解后的已有 embedding 组合初始化，是比随机初始化更稳妥的基线。参考：
  - Vocabulary expansion empirical comparison：https://arxiv.org/abs/2407.05841
  - OFA unseen subword embedding initialization：https://arxiv.org/abs/2311.08849
  - Grounded token initialization 对“所有新 token 用同一个均值”提出风险诊断：https://arxiv.org/abs/2604.02324

对本项目的结论：新增汉字 token 应优先使用“旧 tokenizer 编码结果的 embedding 均值”，而不是全局均值或随机初始化；feature embedding 则可以随机初始化，因为它是新增结构分支的私有参数，不直接进入原 LM 语义空间。

## 3. 当前前置产物

已完成并应作为输入：

```text
models/Qwen3-1.7B-Base-Char/
  vocab.json
  tokenizer.json
  new2old_token_id.json
  new_token_init_token_ids.json
  semantic_vocab_manifest.json
  features/
    feature_vocabs/*.json
    char_features.json
    char_feature_index.jsonl
    feature_index.pt
    feature_index_manifest.json
```

当前 tokenizer/feature validation 已通过，关键值：

```text
new_vocab_size = 147688
hanzi_token_count = 21816
hidden_size = 2048
max_pinyin_per_char = 8
```

注意：迁移前必须修复 `bos_token_id` 同步逻辑。Qwen3 tokenizer 的 `bos_token` 是 `None`，迁移时应把 `model.config.bos_token_id` 和 `generation_config.bos_token_id` 显式置为 `None`，不能保留底座旧值 `151643`。

## 4. 模块划分

建议新增：

```text
src/embedding/
  __init__.py
  semantic_embedding.py
  feature_embedding.py
  feature_memory.py
  config.py
  validate_embedding.py
```

现有 `src/vocab/migrate_embeddings.py` 保留为词表阶段的权重迁移入口，但本阶段应把可复用逻辑抽到 `src/embedding/semantic_embedding.py`，避免 embedding 逻辑继续堆在 vocab 模块中。

## 5. 语义 Embedding 设计

### 5.1 输入

```text
base_model_path = /share/project/wuhaiming/data/models/Qwen3-1.7B-Base/
char_model_path = models/Qwen3-1.7B-Base-Char
new2old_token_id.json
new_token_init_token_ids.json
```

### 5.2 初始化规则

对每个新 token id：

1. 如果 `new_id in new2old_token_id`：
   - `E_sem[new_id] = E_old[old_id]`
   - `lm_head[new_id] = lm_head_old[old_id]`
2. 如果 `new_id in new_token_init_token_ids`：
   - `E_sem[new_id] = mean(E_old[old_ids])`
   - `lm_head[new_id] = mean(lm_head_old[old_ids])`
3. 禁止无记录随机初始化：
   - `new2old ∪ init_ids` 必须覆盖 `[0, new_vocab_size)`。

### 5.3 tied embedding

Qwen3-1.7B-Base 使用 tied word embeddings。迁移后必须：

```python
model.set_input_embeddings(new_input)
model.set_output_embeddings(new_output)
model.config.vocab_size = new_vocab_size
model.tie_weights()
```

并验证：

```python
model.get_input_embeddings().weight.data_ptr()
== model.get_output_embeddings().weight.data_ptr()
```

### 5.4 special token 同步

必须改为“完全同步 tokenizer”，包括 `None`：

```python
for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
    value = getattr(tokenizer, name, None)
    setattr(model.config, name, value)
    setattr(model.generation_config, name, value)
```

这会使：

```text
pad_token_id = 147662
eos_token_id = 147662
bos_token_id = None
```

## 6. 音形 Feature Embedding 设计

### 6.1 输入张量

来自 `feature_index.pt`：

```python
{
  "is_hanzi": BoolTensor[vocab_size],
  "pinyin_ids": LongTensor[vocab_size, max_pinyin_per_char],
  "shengmu_ids": LongTensor[vocab_size, max_pinyin_per_char],
  "yunmu_ids": LongTensor[vocab_size, max_pinyin_per_char],
  "tone_ids": LongTensor[vocab_size, max_pinyin_per_char],
  "pinyin_mask": BoolTensor[vocab_size, max_pinyin_per_char],
  "stroke_count_ids": LongTensor[vocab_size],
  "radical_stroke_ids": LongTensor[vocab_size],
  "structure_ids": LongTensor[vocab_size]
}
```

运行时根据 `input_ids` 查表：

```python
batch_feature_ids = {k: table[k][input_ids] for k in feature_keys}
```

### 6.2 Feature embedding 参数

建议新增 `PhoneticGlyphFeatureEmbedding`：

```python
E_pinyin:        [|V_pinyin|, d_feat]
E_shengmu:       [|V_shengmu|, d_feat]
E_yunmu:         [|V_yunmu|, d_feat]
E_tone:          [|V_tone|, d_feat]
E_stroke_count:  [|V_stroke_count|, d_feat]
E_radical:       [|V_radical_stroke|, d_feat]
E_structure:     [|V_structure|, d_feat]
```

初版推荐：

```text
d_model = 2048
d_feat = 256
feature_memory_dim = 2048
max_pinyin_per_char = 8
```

原因：

- `d_feat=256` 控制新增参数量。
- 最终通过 MLP/projector 投影到 `d_model=2048`，便于后续 PGCA 的 K/V 使用。

### 6.3 拼音侧表示

每个候选读音构造一个 memory slot：

```python
e_py_j = MLP_py(
  concat(
    E_pinyin[pinyin_id_j],
    E_shengmu[shengmu_id_j],
    E_yunmu[yunmu_id_j],
    E_tone[tone_id_j],
  )
)
```

输出：

```text
py_memory: [batch, seq, max_pinyin_per_char, d_model]
py_mask:   [batch, seq, max_pinyin_per_char]
```

多音字保留多个候选，后续 PGCA 通过 query 动态选择。

### 6.4 字形侧表示

当前阶段不做图像 glyph encoder，只做结构化字形：

```python
e_shape = MLP_shape(
  concat(
    E_stroke_count[stroke_count_id],
    E_radical[radical_stroke_id],
    E_structure[structure_id],
  )
)
```

输出一个 memory slot：

```text
shape_memory: [batch, seq, 1, d_model]
shape_mask:   [batch, seq, 1]
```

### 6.5 最终 feature memory

拼接：

```python
M_i = [py_memory_i, shape_memory_i]
```

张量：

```text
feature_memory: [batch, seq, max_pinyin_per_char + 1, d_model]
feature_mask:   [batch, seq, max_pinyin_per_char + 1]
```

非汉字 token：

- `is_hanzi=False`
- `feature_mask=False`
- feature memory 可为零向量
- 后续 PGCA 对这些位置应跳过或得到零 residual

## 7. 初始化策略

### 7.1 语义 embedding

使用旧 embedding 复制/均值初始化，不做随机。

### 7.2 feature embedding

随机初始化即可，但要稳定：

```text
normal_(mean=0, std=0.02)
```

或沿用 Qwen3 config 的 `initializer_range`。如果 config 存在：

```python
std = model.config.initializer_range
```

### 7.3 projector/gate

feature projector 正常初始化；PGCA gate 虽不在本阶段实现，但方案预留：

```text
pgca_gate_init = 0.0
```

这样后续模型初始行为接近 Char-only 模型。

## 8. 配置设计

新增 `EmbeddingFeatureConfig`：

```python
{
  "semantic_vocab_size": 147688,
  "d_model": 2048,
  "d_feat": 256,
  "max_pinyin_per_char": 8,
  "num_feature_slots": 9,
  "feature_vocab_sizes": {
    "pinyin": ...,
    "shengmu": ...,
    "yunmu": ...,
    "tone": ...,
    "stroke_count": ...,
    "radical_stroke": ...,
    "structure": ...
  },
  "use_glyph_image": false
}
```

保存到：

```text
models/Qwen3-1.7B-Base-Char/embedding_config.json
```

## 9. 验证指标

语义侧：

- `new_vocab_size == len(tokenizer)`
- `new2old ∪ init_ids` 覆盖所有 token id
- input/output embedding shape 为 `[147688, 2048]`
- tied embedding 为 true
- `bos_token_id is None`
- `pad/eos_token_id == tokenizer.pad/eos_token_id`
- forward smoke test loss 非 NaN/Inf

feature 侧：

- `feature_memory.shape == [batch, seq, 9, 2048]`
- `feature_mask.shape == [batch, seq, 9]`
- 汉字 token 至少有一个有效 feature slot
- 非汉字 token mask 全 false
- 多音字如“行”应有多个拼音 slot
- 不创建 `lm_head` 对应的 feature 参数

## 10. CLI 建议

新增：

```bash
python -m src.embedding.semantic_embedding migrate \
  --base-model-path /share/project/wuhaiming/data/models/Qwen3-1.7B-Base/ \
  --char-model-path models/Qwen3-1.7B-Base-Char

python -m src.embedding.feature_embedding build \
  --char-model-path models/Qwen3-1.7B-Base-Char

python -m src.embedding.validate_embedding \
  --char-model-path models/Qwen3-1.7B-Base-Char
```

为了兼容已有流程，`src.vocab.build_vocab migrate` 可以继续存在，但内部应调用新的 `src.embedding.semantic_embedding`。

## 11. 实现顺序

1. 修复现有 `migrate_embeddings._sync_special_token_ids`，显式清空 `bos_token_id`。
2. 抽出 `SemanticEmbeddingMigrator` 到 `src/embedding/semantic_embedding.py`。
3. 新增 `EmbeddingFeatureConfig`。
4. 实现 `PhoneticGlyphFeatureEmbedding`。
5. 实现 `FeatureMemoryBuilder`，输入 `input_ids`，输出 `feature_memory/feature_mask`。
6. 新增 `validate_embedding.py`。
7. 将 `src.vocab.build_vocab migrate` 迁移为调用新模块。

## 12. 风险与边界

- feature embedding 是新增随机参数，必须在 Phase 1/2 中训练；不能指望迁移后立即带来收益。
- 拼音候选是静态候选集合，不做上下文消歧；动态选择应交给后续 PGCA。
- 当前不做 glyph image，因此 `glyph_id/glyph_vec` 不进入实现。
- 特征词表与语义词表要解耦：feature embedding 不进入 `lm_head`，也不参与 decode。
- 如果未来扩展 `rare_high_freq.txt` 或重建 tokenizer，必须同步重建 `feature_index.pt` 和 embedding config。
