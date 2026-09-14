# Training - Phase 0: Tokenizer&Embedding Validation

## 1. 目标

确保新 tokenizer 可用，新 embedding/lm_head 初始化合理。 

主要任务：用少量中文、英文、代码、中英混排文本做 forward 测试，观察 loss 是否异常。 

必须完成以下指标： 
- 中文文本必须完全单汉字化。
- 任意文本必须可逆编码。
- special token 和 chat template 行为正常。
- 新模型 forward 不报错，初始 loss 不应极端异常。

## 2. 数据构建

现在要构建的是 **Tokenizer + 权重迁移诊断数据集**，不是训练集。数据量为 **500 条文本**，覆盖各种 token 类型和边界场景。

将验证数据拆成 2 类文件：

```text
data/validation/tokenizer_validation.jsonl
data/validation/forward_validation.jsonl
```

`tokenizer_validation.jsonl` 用来检查单汉字化、可逆编码、special token、chat template。

`forward_validation.jsonl` 用来跑模型 forward，观察 loss、logits、NaN/Inf、不同文本类型下的 loss 是否异常。

### tokenizer_validation.jsonl 构建内容

这个文件重点不是自然语料，而是覆盖边界情况。

建议包含 8 类。

| 类别                            | 数量 | 目的                               |
| ----------------------------- | -: | -------------------------------- |
| 现代简体中文                        | 30 | 检查常规中文是否完全单汉字化                   |
| 多汉字词 / 成语 / 专名                | 30 | 检查“中华人民共和国”“人工智能”等不会被合并成多字 token |
| 繁体中文                          | 20 | 检查繁体字是否单字化                       |
| 罕见字 / 姓名地名用字                  | 20 | 检查低频汉字、扩展汉字、byte fallback        |
| 中英混排                          | 30 | 检查汉字单字化，同时英文保留正常编码               |
| 数字 / 符号 / 数学公式                | 20 | 检查非汉字 token 是否正常                 |
| 代码 / JSON / Markdown          | 30 | 检查代码符号、缩进、换行可逆                   |
| special token / chat template | 20 | 检查 eos、bos、对话模板、特殊符号行为           |

## 3. 数据来源

### 仓库内已有资源：用于 tokenizer 结构验证

用仓库里已经落地的字表和配置文件，首先验证这些产物自身是否一致。

具体来源：

```text
resources/hanzi/hanzi_set.txt
resources/hanzi/tghz2013.txt
resources/hanzi/common_traditional.txt
resources/raw/STCharacters.txt
resources/raw/Unihan/*
resources/raw/cjkvi-ids-master/*
models/Qwen3-1.7B-Base-Char/features/char_features.json
models/Qwen3-1.7B-Base-Char/new_token_init_token_ids.json
models/Qwen3-1.7B-Base-Char/tokenizer_config.json
models/Qwen3-1.7B-Base-Char/generation_config.json
models/Qwen3-1.7B-Base-Char/chat_template.jinja
```

用途：

```text
hanzi_set.txt              检查所有目标汉字是否单 token
tghz2013.txt               检查常用简体字
common_traditional.txt     检查常用繁体字
STCharacters.txt           检查简繁映射相关字符
Unihan / cjkvi-ids         检查罕见字、部首、结构、字形特征覆盖
tokenizer_config           检查 special token
generation_config          检查 eos、pad、stop 行为
chat_template              检查对话模板是否能正常 apply_chat_template
```

这类数据主要用于 `tokenizer_validation`，不是 forward loss 主体。

### 仓库内文档和代码：用于真实项目文本验证

仓库本身就有中文说明、英文路径、代码片段、Markdown 标题、列表、配置字段，非常适合做中英混排和技术文本验证。

具体来源：

```text
README.md
Character-Level Chinese Large Language Model.md
所有 .md 文件
所有 .py 文件中的注释、函数名、配置路径
所有 .json 配置文件
```

用途：

```text
中英混排 forward loss
Markdown 可逆编码
代码片段 forward loss
路径、JSON、配置字段是否正常
```

这类数据最贴近当前项目，建议占验证集的 **30%–40%**。

### 人工构造诊断文本：用于边界情况验证

已由人工构造完成，见 `data/validation/manual_boundary.jsonl`

覆盖内容：

```text
多汉字词：中华人民共和国、自然语言处理、大语言模型、人工智能
成语：莫名其妙、津津有味、井井有条
形近字：己/已/巳，未/末，人/入，土/士
音近字：在/再，的/地/得，作/做
多音字：银行/行走，重量/重新，音乐/快乐
繁体：這裡、學習、語言、體驗
罕见字：龘、靐、麤、淼、垚、燚
emoji / 日文 / 韩文 / 数学符号
```

用途：

```text
中文必须完全单汉字化
特殊 Unicode 必须可逆编码
罕见字要么单 token，要么 byte fallback 正常
不会出现多汉字 token
decode 后文本一致
```

这类数据占验证集 **10%** 左右。

---

## 4. 少量公开自然语料：用于 forward loss 是否异常

forward loss 不能只用人工文本，否则分布太假，需要少量自然文本。

当前来源：

- CCI3.0-HQ 小样本: `/share/project/wuhaiming/data/dataset/CCI3-HQ/data/part_000000.jsonl`
    数据示例：
    ```json
    {
        "id": "02301a3477ca2b5434ab29dfc32f95d853abc",
        "text": "《农村财政与财务》杂志创办于1996，是中国农村财政研究会主管的国家重点学术期刊，国家级期刊，影响因子0.163，现被万方收录(中)等权威机构收录，主要方向：研究报告、文献综述、简报、专题研究\n《农村财政与财务》以宣传党和国家财政政策、推动税收体制改革、研究财税理论、指导基层财政和涉农工作,传播理财知识为宗旨,融政策性、指导性、权威性、实用性和知识性为一体。\n...",
        "score": 2.3
    }
    ```

- 英文 FineWeb-Edu 小样本: `/share/project/wuhaiming/data/dataset/fineweb-edu/sample/10BT/000_00000.parquet`
    数据示例：
    ```bash
    print(data.head())
                                                 text                                               id             dump                                                url                                          file_path language  language_score  token_count     score  int_score
    The Independent Jane\nFor all the love, romanc...  <urn:uuid:0d8a309d-25c5-405d-a08a-c11239f0d717>  CC-MAIN-2013-20      http://austenauthors.net/the-independent-jane  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/se...       en        0.974320          845  2.750000          3
    ```

建议用途：

```text
中文自然段落 loss
英文自然段落 loss
代码 loss
数学公式 / LaTeX loss
中英混排 loss
```

这类数据建议占验证集 **50%-60%**。

### 最终来源配比

验证集总量 **500 条左右**。

```text
A. 仓库字表 / feature / config 生成样本       10%
B. 仓库 markdown / README / 代码抽取样本      35%
C. 人工构造 tokenizer 边界样本               10%
D. 公开自然语料小样本                        45%
```