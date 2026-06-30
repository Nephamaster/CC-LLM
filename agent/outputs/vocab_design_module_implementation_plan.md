# 词表设计模块实现方案

生成日期：2026-06-30

## 1. 目标与范围

本阶段实现 `Character-Level Chinese Large Language Model.md` 中的词表设计模块，服务于 Qwen3-1.7B-Base 的最小可行实验。

范围包括：

- 构建语义词表 `V_sem = V_hanzi_single ∪ V_nonhan_original`。
- 裁剪 Qwen3 原 tokenizer 中的多汉字 token 和对应 BPE merge。
- 确保所有目标汉字以单字 token 进入新词表。
- 生成新旧 token id 映射与新增汉字 token 的 embedding 初始化来源。
- 构建音形特征词表 `V_feat`，本阶段只做拼音、声母、韵母、声调、笔画数、部首/笔画索引、结构类型；暂不做图形编码。
- 生成与 `input_ids` 对齐的 `feature_ids` 数据规范，为后续 PGCA 提供输入。

本阶段不做：

- PGCA 模型结构改造。
- 字形图片渲染、CNN/ViT glyph encoder。
- 大规模 CPT/SFT 训练。

## 2. 资料依据

### 2.1 C-LLM 论文与代码

C-LLM 指出中文拼写纠错需要字符级长度约束和音近约束，混合字符-词 tokenization 会导致字符对齐不清。其方法是修改 BPE 词表和 merge 规则，确保中文字符逐字编码，并继续预训练来恢复新词表下的语言建模能力。

关键结论：

- 字符级 tokenization 能建立明确的字符对齐。
- 直接替换词表后 PPL 会明显恶化，必须做 continued pre-training。
- C-LLM 算法核心是删除 `len(decoded_token) > 1 and is_chinese_string(decoded_token)` 的词表项，并删除会形成多汉字片段的 merge。
- C-LLM 报告新词表约为原词表的 89.2%，并在 7B/14B 模型上通过继续预训练恢复能力。

参考：

- https://arxiv.org/abs/2406.16536
- https://ar5iv.org/html/2406.16536v2
- https://github.com/ktlKTL/C-LLM

### 2.2 本项目已有脚本

当前 `src/charize` 已经比 C-LLM 原始脚本更工程化：

- `src/charize/tokenizer_prune_qwen.py`
  - 支持 fast tokenizer 的 `tokenizer.json`。
  - 支持 byte-level BPE 的 byte encoder/decoder。
  - 删除纯多汉字 token。
  - 保护单汉字 token 依赖的中间 byte/BPE token。
  - 对缺失的单汉字 token 补充 BPE 路径，并生成 `new_token_init_token_ids.json`。
- `src/charize/pruner.py`
  - 读取 `new2old_token_id.json`。
  - 对保留 token 复制原 embedding/lm_head。
  - 对新增 token 使用旧 tokenizer 编码结果的 embedding 均值初始化。

需要扩展的缺口：

- 当前候选汉字来自被删除多汉字 token，不能覆盖 markdown 要求的《通用规范汉字表》、CJK 基本区、常用繁体、姓名地名古文罕见字等范围。
- 当前 tokenizer 仍是“改 BPE 文件”路线，没有输出 `feature_ids`。
- 当前没有音形特征词表与 `char -> feature ids` 索引。
- 当前缺少系统化验收脚本和产物 manifest。

### 2.3 Qwen3-1.7B-Base 约束

Qwen3-1.7B-Base 官方 model card/config 信息：

- 架构：`Qwen3ForCausalLM`
- 参数量：1.7B，总非 embedding 参数约 1.4B
- 层数：28
- hidden size：2048
- attention heads：16
- KV heads：8
- context length：32768
- vocab size：151936
- tied word embeddings：true
- dtype：bfloat16
- tokenizer class：`Qwen2Tokenizer`
- special token id 从 `151643` 起，`<|endoftext|>` 同时作为 eos/pad。
- 官方要求 `transformers >= 4.51.0`，否则可能出现 `KeyError: 'qwen3'`。

参考：

- https://huggingface.co/Qwen/Qwen3-1.7B-Base
- https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/tokenizer_config.json

### 2.4 汉字与音形数据依据

- Unicode Unihan 数据库提供 Han ideograph 的读音、部首笔画、总笔画、异体关系等结构化属性，适合作为覆盖集和笔画/部首索引的权威来源。
- `pypinyin` 支持多音字、声母、韵母、声调等拼音风格，可用于快速构建初版读音特征；其单字拼音数据来自 `pinyin-data`，声母韵母依据《汉语拼音方案》。

参考：

- https://www.unicode.org/reports/tr38/
- https://pypi.org/project/pypinyin/

## 3. 模块设计

建议新增 `src/vocab` 包，不继续把所有逻辑堆在 `src/charize`。`src/charize` 保留为参考和兼容脚本，新模块按产物边界拆分。

```text
src/vocab/
  __init__.py
  unicode_ranges.py
  hanzi_set.py
  bpe_state.py
  semantic_vocab.py
  feature_vocab.py
  feature_index.py
  qwen3_char_tokenizer.py
  migrate_embeddings.py
  validate_vocab.py
  build_vocab.py
```

### 3.1 `unicode_ranges.py`

职责：

- 定义 CJK 汉字判断函数。
- 区分“本阶段纳入语义词表的汉字范围”和“只用于识别/过滤的广义 CJK 范围”。

建议：

- `is_cjk_hanzi(char)` 覆盖 CJK Unified Ideographs 基本区、扩展 A/B/C/D/E/F/G/H、兼容汉字。
- `is_vocab_hanzi(char)` 初版默认纳入：
  - CJK Unified Ideographs 基本区：`U+4E00..U+9FFF`
  - CJK Unified Ideographs Extension A：`U+3400..U+4DBF`
  - CJK Compatibility Ideographs：`U+F900..U+FAFF`
  - 配置文件显式追加的姓名、地名、古文、异体字高频字。

取舍：

- markdown 写到“CJK 基本区 + 高频罕见字”，不建议一开始把所有扩展 B-H 全量加入语义词表。扩展 B-H 会显著增加词表和 embedding 参数，且大量字符缺少稳定音形特征。初版把扩展区作为可选 `--include-cjk-ext`。

### 3.2 `hanzi_set.py`

职责：

- 构建 `V_hanzi_single` 候选集。
- 输出覆盖集审计报告。

输入来源：

- 内置 Unicode range。
- `resources/hanzi/tghz2013.txt`：通用规范汉字表 8105 字，如本地未提供则允许用户指定路径。
- `resources/hanzi/common_traditional.txt`：常用繁体字表。
- `resources/hanzi/rare_high_freq.txt`：姓名、地名、古文和异体高频罕见字。
- 原 tokenizer 中被删除多汉字 token 拆出的汉字。

输出：

- `hanzi_set.txt`：每行一个汉字。
- `hanzi_set.meta.json`：

```json
{
  "num_chars": 0,
  "sources": {
    "cjk_basic": 20992,
    "cjk_ext_a": 6592,
    "tghz2013": 8105,
    "traditional": 0,
    "rare_high_freq": 0,
    "from_pruned_tokens": 0
  }
}
```

### 3.3 `bpe_state.py`

职责：

- 从 Qwen3 tokenizer 提取 BPE `vocab`、`merges`、`tokenizer.json`。
- 兼容 `Qwen2Tokenizer` slow/fast 形态。
- 统一 byte-level token decode/encode。

继承当前 `src/charize/tokenizer_prune_qwen.py` 的以下逻辑：

- `bytes_to_unicode`
- `get_byte_decoder`
- `get_byte_encoder`
- `decode_bpe_piece`
- `extract_bpe_state`
- `normalize_vocab`
- `normalize_merges`

需要调整：

- 不在工具函数里 `print` 大量删除项，统一用 logger 和 summary。
- 所有函数加类型标注。
- 对 `errors="replace"` 的 decode 结果额外记录不可逆 token，避免误判。

### 3.4 `semantic_vocab.py`

职责：

- 实现 `V_sem = V_hanzi_single ∪ V_nonhan_original`。
- 删除所有“含两个及以上汉字”的原 token。
- 删除所有“包含汉字且不是单汉字”的混合 token。
- 保留 special token、byte token、非汉字 token、单汉字 token。
- 对 `hanzi_set` 中原词表没有的汉字补充单字 BPE token。

删除规则建议比 C-LLM 更严格：

```python
hanzi_count = count_hanzi(decoded_token)
remove = hanzi_count >= 1 and decoded_token not in hanzi_set_single_char_form
keep = hanzi_count == 0 or decoded_token in hanzi_set_single_char_form or is_special_token
```

原因：

- C-LLM 只删除“纯多汉字 token”。本研究要求“所有中文汉字以单字形式进入词表”，因此类似 `abc中国`、`中国.`、`的</w>` 这类含汉字混合 token 也应删除，避免编码时跨汉字合并。

单汉字路径补充：

- 如果汉字原本已有完整单字 token，直接保留并记录 `new2old_token_id`。
- 如果没有完整单字 token，则按 UTF-8 byte piece 构造 BPE 路径，补充中间 token 和最终单字 token。
- 新增最终单字 token 的 `new_token_init_token_ids` 使用 `old_tokenizer.encode(char, add_special_tokens=False)`。
- 新增中间 byte/BPE token 如原词表存在则复制，否则使用同一汉字旧编码均值初始化。

merge 规则：

- 删除任何会生成“含汉字且非单汉字”的 merge。
- 保护生成单汉字 token 必需的 byte merge。
- 新增汉字路径 merge 提升到 merge 列表前部，保证单字 token 形成优先于跨字合并。
- 过滤 dangling merge。

输出：

- `vocab.json`
- `merges.txt`
- `tokenizer.json`
- `tokenizer_config.json`
- `new2old_token_id.json`
- `new_token_init_token_ids.json`
- `semantic_vocab_manifest.json`

### 3.5 `feature_vocab.py`

职责：

- 根据 `hanzi_set` 构建音形特征子词表。
- 保留 `<pad>`、`<unk>`、`<none>`。

子词表：

- `pinyin_vocab.json`：如 `zhong1`、`hang2`，轻声统一为 `5`。
- `shengmu_vocab.json`：包含空声母 `""`，如 `b p m f ... zh ch sh r z c s`。
- `yunmu_vocab.json`：如 `a ai an ang ... üe`。内部建议统一用 `v` 存储 `ü`，避免 Unicode 标准化差异。
- `tone_vocab.json`：`0/1/2/3/4/5`，其中 `0` 表示无读音或未知。
- `stroke_count_vocab.json`：整数笔画数，未知为 `0`。
- `radical_stroke_vocab.json`：Unihan `kRSUnicode` 原始值或解析后的 `radical.residual`。
- `structure_vocab.json`：初版来自本地字形结构表；缺失填 `<unk>`。

拼音生成策略：

- 使用 `pypinyin.pinyin(char, heteronym=True, style=Style.TONE3, neutral_tone_with_five=True, strict=True)` 得到候选拼音。
- 再用 `pypinyin.contrib.tone_convert.to_initials/to_finals` 拆分声母韵母。
- 对 `ü` 做统一规范：存储层用 `v`，展示层再转回 `ü`。
- 多音字保留候选集合，不做静态消歧。

笔画/部首策略：

- 优先从 Unihan 解析：
  - `kTotalStrokes` -> `stroke_count`
  - `kRSUnicode` -> `radical_stroke`
  - `kMandarin` 可作为 pypinyin 失败时的补充读音来源
- 如果本地没有 Unihan 数据，本阶段代码应允许 `--allow-missing-features`，缺失项填 `<unk>`，但验收报告必须列出缺失率。

结构类型策略：

- 初版定义结构表输入格式：

```tsv
char    structure
明      left_right
品      top_middle_bottom
国      full_surround
日      single
```

- 暂不从图形自动推断结构。
- 缺失结构填 `<unk>`。

输出：

- `feature_vocabs/*.json`
- `feature_vocab_manifest.json`

### 3.6 `feature_index.py`

职责：

- 构建每个汉字 token id 对应的 feature id。
- 支持多音字候选。

建议输出 `char_feature_index.jsonl`：

```json
{"char":"行","token_id":12345,"pinyin_ids":[10,42],"shengmu_ids":[8,6],"yunmu_ids":[31,12],"tone_ids":[2,2],"stroke_count_id":7,"radical_stroke_id":88,"structure_id":2}
```

再额外输出适合训练加载的 dense tensor 文件：

- `feature_index.pt`

推荐张量结构：

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

非汉字 token：

- `is_hanzi=False`
- 所有 feature id 填 `<none>` 或 `0`
- mask 全 false

### 3.7 `qwen3_char_tokenizer.py`

职责：

- 包装新 tokenizer，输出 `input_ids` 和对齐 `feature_ids`。
- 保留原 tokenizer 的 special token、chat template、decode 行为。

编码流程：

1. 规范化输入：统一换行，移除非法控制字符。
2. 从左到右扫描。
3. 优先匹配 special token。
4. 当前字符为目标汉字：输出单字 token id，并查 `feature_index`。
5. 连续非汉字 span：调用底层 tokenizer 编码。
6. 如果非汉字 span 编码后仍出现含汉字 token，视为构建错误并抛出异常。

解码流程：

- 只依赖 `input_ids`，调用底层 decode。
- `feature_ids` 不参与 decode。

注意：

- 如果语义词表已经正确裁剪，理论上可以直接用 Hugging Face tokenizer 编码全量文本；但为了稳定输出 `feature_ids` 并防止混合 token 漏网，建议实现扫描式 wrapper。

### 3.8 `migrate_embeddings.py`

职责：

- 从 Qwen3-1.7B-Base 迁移 embedding/lm_head 到新语义词表。

继承当前 `src/charize/pruner.py`：

- 保留 token：复制 input embedding 和 output embedding。
- 新增 token：取旧 tokenizer 编码 token ids 的均值初始化。
- Qwen3 配置 `tie_word_embeddings=true`，但实现时仍应兼容 `get_output_embeddings()`，最终保存前调用 `model.tie_weights()` 或检查输入输出权重共享状态。

输出：

- 新模型目录。
- `embedding_migration_report.json`，包含复制数、新增均值初始化数、随机初始化数、dtype、shape、tie 检查结果。

### 3.9 `validate_vocab.py`

职责：

- 自动验收词表模块。

必须通过的检查：

- 所有 `hanzi_set.txt` 中的字符编码后都是单个 token。
- 任意编码结果中不存在 decoded token 含两个及以上汉字。
- 不存在 decoded token 同时含汉字和非汉字。
- special token id 与 tokenizer config/model config 一致。
- `decode(encode(text))` 对核心样例可逆。
- 中英混排、URL、代码、数学公式不报错。
- `feature_ids` 与 `input_ids` 长度完全一致。
- 汉字位置 `is_hanzi=True` 且至少有一组有效拼音候选或 `<unk>` 记录。
- 非汉字位置 feature 全 `<none>`。
- 新 embedding shape 等于新 vocab size。

建议样例：

```text
这是一个中文分词测试。
行行重行行，银行行长行不行？
繁體字與简体字混排。
Python 3.11: print("你好, Qwen3!")
URL: https://example.com/中文?a=1
数学：∑_{i=1}^n i = n(n+1)/2
罕见字：𠮷、㐂、堃、垚、喆
```

输出：

- `validation_report.json`
- `tokenization_examples.md`

## 4. CLI 设计

统一入口：

```bash
conda activate work
python -m src.vocab_design.build_vocab \
  --base-model Qwen/Qwen3-1.7B-Base \
  --output-dir outputs/qwen3-1.7b-char-vocab \
  --hanzi-sources resources/hanzi/tghz2013.txt resources/hanzi/common_traditional.txt resources/hanzi/rare_high_freq.txt \
  --unihan-zip resources/unihan/Unihan.zip \
  --structure-table resources/hanzi/structure.tsv \
  --include-cjk-basic \
  --include-cjk-ext-a \
  --allow-missing-features
```

建议 CLI 子命令：

```bash
python -m src.vocab_design.build_vocab build-hanzi-set ...
python -m src.vocab_design.build_vocab build-semantic-vocab ...
python -m src.vocab_design.build_vocab build-feature-vocab ...
python -m src.vocab_design.build_vocab migrate-embeddings ...
python -m src.vocab_design.build_vocab validate ...
python -m src.vocab_design.build_vocab all ...
```

## 5. 产物目录

```text
outputs/qwen3-1.7b-char-vocab/
  tokenizer/
    vocab.json
    merges.txt
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    new2old_token_id.json
    new_token_init_token_ids.json
    semantic_vocab_manifest.json
  features/
    hanzi_set.txt
    hanzi_set.meta.json
    feature_vocabs/
      pinyin_vocab.json
      shengmu_vocab.json
      yunmu_vocab.json
      tone_vocab.json
      stroke_count_vocab.json
      radical_stroke_vocab.json
      structure_vocab.json
    char_feature_index.jsonl
    feature_index.pt
    feature_vocab_manifest.json
  model/
    config.json
    model.safetensors
    generation_config.json
  reports/
    embedding_migration_report.json
    validation_report.json
    tokenization_examples.md
```

## 6. 实现顺序

1. 抽取并清理 `src/charize/tokenizer_prune_qwen.py` 中的 BPE 基础函数到 `bpe_state.py`。
2. 实现 `unicode_ranges.py` 和 `hanzi_set.py`，先生成覆盖集与审计报告。
3. 实现 `semantic_vocab.py`，完成更严格的汉字 token 裁剪和单字补全。
4. 实现 `validate_vocab.py` 的 tokenizer 验收，先不涉及音形。
5. 实现 `feature_vocab.py` 和 `feature_index.py`，生成 `V_feat` 与 dense tensor。
6. 实现 `qwen3_char_tokenizer.py` wrapper，打通 `input_ids + feature_ids`。
7. 将 `src/charize/pruner.py` 迁移为 `migrate_embeddings.py`，适配 Qwen3 tied embeddings。
8. 实现总入口 `build_vocab.py` 和完整报告。

## 7. 风险与处理

- 风险：全量 CJK 基本区 + 扩展 A 会增加约 2.7 万汉字 token，Qwen3 原本已有大量中文 token，但新增汉字仍会增加 embedding 参数。
  - 处理：默认基本区 + 扩展 A + 显式高频罕见字；扩展 B-H 只作为可选实验。
- 风险：Qwen3 tokenizer 的 byte-level BPE 中间 token 如果误删，会导致单字不可达。
  - 处理：沿用并加强当前脚本的 dependency protection 和 promoted merge。
- 风险：只删除纯多汉字 token 不足以满足研究目标。
  - 处理：删除任何含汉字且非单汉字的 token，包括中英混合 token。
- 风险：pypinyin 多音字候选不等于上下文真实读音。
  - 处理：本阶段保留候选集合，不做消歧；后续由 PGCA query 在候选 memory 中选择。
- 风险：结构类型数据源不完整。
  - 处理：结构缺失用 `<unk>`，报告缺失率；先不阻塞 tokenizer 实验。
- 风险：新 tokenizer 使中文序列变长。
  - 处理：验收报告必须统计中文字符/token 比、平均 token 增长率，后续训练报告单独追踪效率。

## 8. 阶段验收标准

本阶段完成后应满足：

- `hanzi_set` 中 100% 字符可被编码为单个 token。
- 随机中文样本文本中，不出现多汉字 token。
- 中英混排、代码、URL、数学符号保持可编码和可解码。
- special token 与 Qwen3 原始 tokenizer 行为一致。
- `feature_index.pt` 覆盖所有汉字 token，非汉字 token 有明确空特征。
- `new2old_token_id.json + new_token_init_token_ids.json` 覆盖新词表全部 token id。
- 迁移后模型可运行一次 forward，loss 非 NaN/Inf。

## 9. 与现有 `src/charize` 的关系

短期可以复用 `src/charize` 的核心算法，但不建议直接继续在该目录叠加音形词表逻辑。推荐：

- `src/charize` 保留为历史参考。
- 新实现放入 `src/vocab_design`。
- 当前 `tokenizer_prune_qwen.py` 中成熟的 byte-level BPE 保护逻辑迁移过去。
- 当前 `pruner.py` 的 embedding 迁移逻辑迁移过去，并补充 Qwen3 tied embedding 检查。

这样词表设计模块边界更清晰，也方便后续 PGCA、训练脚本和评估脚本复用。
