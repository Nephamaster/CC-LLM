# 构建词表所需资源清单

生成日期：2026-06-30

本文明确“词表设计模块”落地时需要准备的资源、格式、用途和缺失降级策略。当前实验对象为 `Qwen/Qwen3-1.7B-Base`，本阶段音形词表不做图形编码。

## 1. 必需资源

### 1.1 底座模型与 tokenizer

用途：

- 读取原始 BPE `vocab`、`merges`、special tokens、chat/template 配置。
- 生成 `V_nonhan_original`。
- 生成 `new2old_token_id.json` 和新增 token 的初始化来源。
- 迁移 embedding/lm_head。

资源：

```text
Qwen/Qwen3-1.7B-Base
```

可用形式：

- Hugging Face 模型名，自动从缓存或网络加载。
- 本地模型目录，例如：

```text
models/Qwen3-1.7B-Base/
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  special_tokens_map.json
  model.safetensors
```

最低要求：

- `transformers >= 4.51.0`
- `torch`
- `tokenizers`

### 1.2 汉字覆盖集数据

用途：

- 决定哪些汉字必须以单字 token 形式进入 `V_hanzi_single`。
- 作为后续音形特征索引的主键集合。

必需来源：

1. CJK Unified Ideographs 基本区

```text
U+4E00..U+9FFF
```

这部分可以由代码直接按 Unicode 范围生成，不需要外部文件。

2. 《通用规范汉字表》

建议文件：

```text
resources/hanzi/tghz2013.txt
```

建议格式：

```text
一
乙
二
十
...
```

或 TSV：

```text
char    level
一      1
乙      1
...
```

如果未提供，代码仍可运行，但验收报告必须标记 `tghz2013` 覆盖源缺失。

3. 常用繁体字表

建议文件：

```text
resources/hanzi/common_traditional.txt
```

建议格式：

```text
體
與
臺
後
...
```

如果未提供，初版仍可靠 CJK 基本区覆盖大量繁体字，但无法显式审计“常用繁体”覆盖。

4. 高频罕见字表

建议文件：

```text
resources/hanzi/rare_high_freq.txt
```

覆盖范围：

- 姓名常用罕见字
- 地名常用罕见字
- 古文常见字
- 异体字或兼容字中高频字

建议格式：

```text
𠮷
㐂
堃
垚
喆
昇
祎
...
```

注意：

- 扩展 B-H 不建议默认全量加入，因为词表和 embedding 成本明显增加，且很多字缺少稳定音形特征。
- 如果确实需要，建议用独立开关：`--include-cjk-ext-b`、`--include-cjk-ext-c` 等。

### 1.3 汉字判定规则

用途：

- 裁剪所有含汉字但不是单汉字的 token。
- 防止中英混合 token、标点混合 token 夹带汉字。

这部分不需要外部文件，代码内置 Unicode 范围即可。

需要覆盖：

```text
CJK Unified Ideographs            U+4E00..U+9FFF
CJK Extension A                   U+3400..U+4DBF
CJK Compatibility Ideographs      U+F900..U+FAFF
CJK Extension B-H                 可识别，但默认不一定纳入词表
```

## 2. 音形特征资源

### 2.1 拼音数据

用途：

- 生成 `pinyin_id`
- 生成 `shengmu_id`
- 生成 `yunmu_id`
- 生成 `tone_id`
- 为多音字保存候选读音集合

推荐依赖：

```text
pypinyin
```

推荐调用策略：

- `heteronym=True`：保留多音字候选。
- `Style.TONE3`：声调用数字表示，如 `xing2`。
- `neutral_tone_with_five=True`：轻声统一为 `5`。
- `strict=True`：按标准声韵母拆分。

输出规范：

```json
{
  "char": "行",
  "pinyin": ["xing2", "hang2"],
  "shengmu": ["x", "h"],
  "yunmu": ["ing", "ang"],
  "tone": [2, 2]
}
```

降级策略：

- `pypinyin` 查不到的字填 `<unk>`，并写入缺失报告。
- 如提供 Unihan，可用 `kMandarin` 作补充读音来源。

### 2.2 Unihan 数据

用途：

- 补充汉字属性。
- 提供部首笔画、总笔画、普通话读音等权威字段。

建议文件：

```text
resources/unihan/Unihan.zip
```

至少需要解析字段：

```text
kMandarin
kTotalStrokes
kRSUnicode
```

建议输出：

```json
{
  "char": "明",
  "kMandarin": "míng",
  "kTotalStrokes": 8,
  "kRSUnicode": "72.4"
}
```

降级策略：

- 没有 Unihan 时，拼音仍可由 `pypinyin` 生成。
- 笔画数、部首笔画填 `<unk>`。
- 验收报告中统计缺失率，缺失率不能静默忽略。

### 2.3 字形结构表

用途：

- 生成 `structure_id`。
- 区分左右、上下、包围、独体等结构。

建议文件：

```text
resources/hanzi/structure.tsv
```

建议格式：

```text
char    structure
明      left_right
品      top_middle_bottom
国      full_surround
日      single
问      upper_surround
这      lower_left_surround
```

建议结构枚举：

```text
single
left_right
left_middle_right
top_bottom
top_middle_bottom
full_surround
upper_surround
lower_surround
left_surround
right_surround
upper_left_surround
upper_right_surround
lower_left_surround
lower_right_surround
overlaid
unknown
```

降级策略：

- 缺失结构填 `<unk>`。
- 本阶段不从字体图像自动推断结构。

### 2.4 笔顺或笔画序列数据

markdown 中写到 `stroke_seq_id` 是笔顺或笔画序列。当前最小版本可以先实现“笔画数/部首笔画”，不强制做完整笔顺序列。

如果要做完整笔顺，建议资源：

```text
resources/hanzi/stroke_order.jsonl
```

建议格式：

```json
{"char":"一","strokes":["H"]}
{"char":"十","strokes":["H","S"]}
{"char":"明","strokes":["S","HZ","H","H","P","HZ","H","H"]}
```

笔画类型建议先归一到有限集合，例如：

```text
H    横
S    竖
P    撇
N    捺/点
T    提
HZ   横折及折类
G    钩类
UNK  未知
```

降级策略：

- 没有完整笔顺数据时，只生成 `stroke_count_id` 和 `radical_stroke_id`。
- `stroke_seq_id` 暂不启用或统一填 `<none>`。

## 3. 可选增强资源

### 3.1 简繁转换表

用途：

- 审计简体、繁体覆盖。
- 以后可做简繁对齐评估。

建议依赖或文件：

```text
opencc
resources/hanzi/simplified_traditional.tsv
```

本阶段非必需。

### 3.2 多音字上下文数据

用途：

- 后续训练 PGCA 动态选择读音。
- 本阶段构建词表只保留候选读音集合，不需要上下文标注。

建议后续文件：

```text
resources/phonetic/polyphone_context.jsonl
```

### 3.3 形近字/音近字混淆集

用途：

- 后续 CSC、OCR/ASR 噪声增强与评估。
- 不参与当前词表构建主流程。

建议后续文件：

```text
resources/confusion/shape_confusion.jsonl
resources/confusion/phonetic_confusion.jsonl
```

### 3.4 Glyph 图像或字体资源

当前阶段明确不需要。

后续如果做图形编码，才需要：

```text
resources/fonts/*.ttf
resources/glyph_images/
```

## 4. 构建产物需要的中间资源

词表构建脚本会生成以下中间文件，不需要手动准备：

```text
hanzi_set.txt
hanzi_set.meta.json
new2old_token_id.json
new_token_init_token_ids.json
feature_vocabs/*.json
char_feature_index.jsonl
feature_index.pt
semantic_vocab_manifest.json
feature_vocab_manifest.json
validation_report.json
```

## 5. 推荐目录结构

```text
resources/
  hanzi/
    tghz2013.txt
    common_traditional.txt
    rare_high_freq.txt
    structure.tsv
    stroke_order.jsonl          # 可选
    simplified_traditional.tsv  # 可选
  unihan/
    Unihan.zip
  phonetic/
    polyphone_context.jsonl     # 后续可选
  confusion/
    shape_confusion.jsonl       # 后续可选
    phonetic_confusion.jsonl    # 后续可选
models/
  Qwen3-1.7B-Base/              # 可选，本地模型目录
outputs/
  qwen3-1.7b-char-vocab/
```

## 6. 最小可运行资源集

如果只想先把 tokenizer 和 `feature_ids` 跑通，最小资源是：

```text
1. Qwen/Qwen3-1.7B-Base tokenizer 和模型权重
2. Python 环境：work
3. torch
4. transformers >= 4.51.0
5. tokenizers
6. pypinyin
7. 代码内置 CJK 基本区范围
```

这时可以生成：

- 单汉字化 tokenizer。
- `new2old_token_id.json`。
- `new_token_init_token_ids.json`。
- 拼音/声母/韵母/声调特征。
- 缺失结构、笔画、部首的 `<unk>` 占位。

但不建议把这个版本作为最终词表，因为缺少《通用规范汉字表》、常用繁体、高频罕见字和 Unihan 审计。

## 7. 推荐正式资源集

正式构建建议准备：

```text
1. Qwen/Qwen3-1.7B-Base 本地完整模型目录或可访问的 HF 缓存
2. resources/hanzi/tghz2013.txt
3. resources/hanzi/common_traditional.txt
4. resources/hanzi/rare_high_freq.txt
5. resources/unihan/Unihan.zip
6. resources/hanzi/structure.tsv
7. pypinyin
8. transformers >= 4.51.0
9. torch + tokenizers + safetensors
```

正式版本验收时必须报告：

- 汉字总覆盖数。
- 各来源贡献数量。
- 新词表大小。
- 删除的含汉字复合 token 数。
- 新增单汉字 token 数。
- 拼音缺失率。
- 笔画/部首缺失率。
- 结构类型缺失率。
- 中文样例 token 增长率。

## 8. 当前 markdown 观察

我重新读取了 `Character-Level Chinese Large Language Model.md`。当前 `git diff -- "Character-Level Chinese Large Language Model.md"` 没有输出，说明相对当前 Git 基线没有可见差异。

需要注意的一点是：markdown 的 `1.2 音形词表` 仍包含 `glyph_id / glyph_vec` 的长期设计描述；但你的本轮要求和 markdown 的最小可行版本都明确当前阶段“不使用图像 glyph encoder”。因此实现时按当前阶段执行：

- 不准备字体或 glyph 图片资源。
- 不生成 `glyph_id/glyph_vec`。
- 音形词表只覆盖拼音、声母、韵母、声调、笔画/部首、结构类型。
