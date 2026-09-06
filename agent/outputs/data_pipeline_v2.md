可以。Phase2 我建议正式采用一套“DataTrove 主导 + 项目自定义中文规则”的生产级 pipeline，而不是继续扩充你当前那套 SQLite/Python 文档循环。

核心目标是：20B-token 全参继续预训练数据，构建过程可并行、可断点、可复用、可审计，而且 Phase1/Phase2 共用同一份源数据缓存。

第三方库我建议这样定：

- `datatrove`：主框架，负责 reader/writer、并行、sharding、sampling、stats、MinHash、tokenization、Slurm/Ray executor。它本身就是为大规模 LLM 预训练数据处理设计的，支持 local/Slurm/Ray，并提供 FineWeb 复现、MinHash、token estimation、100B-token subset/mixing 示例。
- `pyarrow`：Parquet 中间数据格式。
- `tokenizers`：最终高吞吐精确 tokenization。
- `fasttext` / DataTrove `LanguageFilter`：语言识别。DataTrove 已提供基于 FastText/GlotLID 的语言过滤模块。
- `datasketch`：如果 DataTrove MinHash 接口满足需求就不单独直接调用；只作为必要时补充。
- `transformers`：质量模型推理、最终 tokenizer/model 兼容。
- `numpy` / `orjson` / `xxhash`：高速统计、JSON、稳定哈希。
- `ftfy`：只用于明显编码异常修复，不做激进 Unicode normalization。
- 不建议同时再引入 Dolma 作为第二套主框架。Dolma 也能完成 dedup/mixing/tokenization，而且支持 Bloom-filter 去重与 HF tokenizer，但同时维护 DataTrove + Dolma 没有收益。

我会最终选：**DataTrove 做唯一数据底座。**

---

## 一、Phase2 最终数据分布

先把内容分布固定下来：

| 类别 | 比例 | 20B tokens |
|---|---:|---:|
| 中文高质量通用 | 50% | 10B |
| 中文知识密集 | 15% | 3B |
| 英文/多语/中英混排 | 15% | 3B |
| 数学/代码/科学 | 15% | 3B |
| 汉字长尾/繁体/古文/音形增强 | 5% | 1B |

“长文本”不再单独占一个互斥 bucket，而是作为横向属性控制。例如最终 20B 中可以要求：

```text
long_doc_tokens >= 10%
```

但这些 2B 长文本仍然属于：

```text
Chinese knowledge
English
Science
Classical
...
```

中的某一类。

---

# 二、具体数据来源

我建议 Phase2 直接定成下面这些，不继续无边界扩来源。

### 中文通用 10B

主要：

```text
CCI3.0-HQ
OpenCSG FineWeb-Edu-Chinese V2
WanJuan1.0 高质量子集
```

建议内部：

```text
CCI3.0-HQ                  5B
FineWeb-Edu-Chinese        3.5B
WanJuan                    1.5B
```

如果 CCI/FineWeb-Edu 自身规模足够，WanJuan只是 reserve。

---

### 中文知识密集 3B

```text
Chinese Cosmopedia
中文 Wikipedia
FineWeb-zhtw
Wikisource
ect-krp
```

建议：

```text
Cosmopedia        1.2B
Wikipedia         0.7B
FineWeb-zhtw      0.6B
Wikisource        0.4B
ect-krp           <=0.1B
```

其中 `ect-krp` 不强行补 quota。实际 unique token 不足就全量使用一次，剩余返回 Wikisource。

---

### 英文/多语/中英混排 3B

```text
FineWeb-Edu English       1.6B
FineWeb2 multilingual     0.4B
中英技术混排               1.0B
```

中英混排不要再爬 GitHub 几百个仓库作为主来源，直接从：

```text
CCI3-HQ
FineWeb Chinese
FineWeb-Edu Chinese
The Stack v2 README/docs
```

中自动筛选。

---

### 数学/代码/科学 3B

```text
The Stack v2       1.4B
OpenWebMath        0.8B
S2ORC/OA science   0.8B
```

如果你暂时不想再增加数据下载，也可以 Phase2 第一版：

```text
The Stack v2
+
已有 science/math corpus
```

先跑，但论文正式实验建议把数学和科学与普通代码分开。

---

### 汉字长尾 1B

这不是独立新数据源。

从前面中文 corpus 中挖：

```text
繁体
古汉语
罕见字
异体字
姓名地名
多音字高密度文档
音近/形近相关自然上下文
```

这一 bucket 的核心原则仍然是：

> 选择包含目标汉字的真实文档，而不是人工拼“龘靐麤”。

---

# 三、完整目录设计

建议彻底固定：

```text
data/
├── raw/
│   ├── cci3_hq/
│   ├── fineweb_zh/
│   ├── fineweb_edu_zh/
│   ├── fineweb_edu_en/
│   ├── wanjuan/
│   ├── cosmopedia/
│   ├── fineweb_zhtw/
│   ├── wikipedia/
│   ├── wikisource/
│   ├── ect_krp/
│   ├── the_stack_v2/
│   ├── openwebmath/
│   └── s2orc/
│
├── cache/
├── scored/
├── candidates/
├── dedup/
├── selected/
├── tokenized/
├── packed/
├── stats/
└── manifests/
```

Phase1/Phase2 不各自重新做一套 raw。

---

# 四、Stage 0：Source ingestion / Cache

工具：

```text
DataTrove
PyArrow
```

任务：

```text
raw JSONL / Parquet / tar
          ↓
统一 Document
          ↓
Parquet cache
```

统一字段：

```python
Document(
    id=doc_id,
    text=text,
    metadata={
        "source": ...,
        "subset": ...,
        "license": ...,
        "url": ...,
        "language_hint": ...
    }
)
```

DataTrove 本身的基本数据对象就是：

```text
text
id
metadata
```

并用 reader → generator → pipeline block → writer 的形式流动。

这里只做：

```text
字段抽取
明显 HTML 清理
非法控制符处理
基础 metadata
```

绝不 tokenize。

---

# 五、Stage 1：Basic normalization

建议自己实现：

```text
CCLLMNormalizeFilter
```

作为 DataTrove custom block。

规则固定：

允许：

```text
CRLF → LF
删除 NUL/control chars
HTML entity decode
明显 boilerplate
连续超量空白
```

禁止：

```text
繁体 → 简体
古文 → 现代文
NFKC 全量 normalize
异体字统一
CJK compatibility 全量折叠
```

因为你研究的就是汉字结构。

最终保存：

```text
normalized_text
```

如果磁盘允许，可以 metadata 保存：

```text
raw_hash
normalized_hash
```

不需要保存 raw text 两份。

---

# 六、Stage 2：语言和类型分类

工具：

```text
DataTrove LanguageFilter
fastText
自定义规则
```

DataTrove 已有 FastText/GlotLID language filter。

每个文档计算：

```text
language
hanzi_ratio
latin_ratio
digit_ratio
code_ratio
```

然后分类成：

```text
zh
en
multilingual
zh_en_mixed
code
math
scientific
traditional
classical
```

注意：

```text
traditional
classical
long_doc
rare_hanzi
```

最好是 tag，不是互斥主类别。

例如：

```json
{
  "domain": "knowledge_zh",
  "traditional": true,
  "classical": false,
  "long_doc": true
}
```

---

# 七、Stage 3：廉价规则质量过滤

这一阶段只允许 CPU-cheap rules。

建议实现一个：

```text
CCLLMQualityFilter
```

检查：

```text
char_count
hanzi_ratio
line_count
mean_line_length
repeat_line_ratio
duplicate_ngram_ratio
punctuation_ratio
digit_ratio
symbol_ratio
URL ratio
boilerplate ratio
```

现代中文 web：

可以过滤：

```text
过短
SEO
菜单页
纯关键词
大量重复行
乱码
```

古文/诗歌单独使用另一套 profile。

不要拿：

```text
平均句长
标点率
```

直接过滤 Wikisource/ect-krp。

也就是说至少定义：

```text
quality_profile = modern_web
quality_profile = classical
quality_profile = code
quality_profile = scientific
```

---

# 八、Stage 4：语义质量评分

Phase2 我建议正式加入。

但不要所有来源都跑昂贵 Transformer classifier。

分层处理。

### A. 已高质量数据

例如：

```text
CCI3.0-HQ
FineWeb-Edu
Cosmopedia
Wikipedia
ect-krp
```

不需要再跑重模型筛掉一遍。

仅记录：

```text
upstream_quality = high
```

---

### B. 普通 web / WanJuan / mixed candidates

跑质量评分。

英文可以直接使用 FineWeb-Edu 官方 classifier，它就是为判断 web page 的 educational value 而训练，并用于构建 FineWeb-Edu。

但这里有一个关键点：

> FineWeb-Edu classifier 是英文 classifier，不应该直接拿来给中文做质量判别。

中文我建议第一版不要训练一个复杂质量模型。

采用：

```text
上游质量分
+
中文 heuristic
+
来源 prior
```

例如：

```text
CCI-HQ          quality_prior=1.0
FineWeb-Edu-ZH  1.0
FineWeb-ZH      0.7
WanJuan         0.7
```

后面有时间再自己建立：

```text
10万中文文档
→ LLM 标注 0~5
→ 训练小型 BERT/FastText quality classifier
```

这可以作为第二版。

---

# 九、Stage 5：Token calibration

这一步非常重要，而且一定在 dedup/tokenization 前。

DataTrove 官方直接有：

```text
examples/estimate_tokens.py
```

用于：

```text
只流式 tokenize 小样本
→ 平均 tokens/doc 收敛
→ 推算总 token
→ 决定 SamplerFilter rate
```

官方文档明确指出这是为了从 multi-trillion-token dataset 中构造固定 token 子集。

每个：

```text
source × category
```

抽：

```text
50K docs
```

计算：

```text
tokens/doc
tokens/char
P50
P90
P99
```

最后得到：

```json
{
  "cci3_hq": {
    "estimated_tokens": 56000000000,
    "tokens_per_char": 1.03
  }
}
```

---

# 十、Stage 6：Candidate sampling

这是整个效率体系的核心。

假设：

```text
CCI target = 5B
available estimated = 60B
```

则：

```text
rate = 5B / 60B * 1.15
```

通过 DataTrove：

```python
SamplerFilter(
    rate=rate,
    seed=42
)
```

即可。

DataTrove 官方 100B mixture 示例本身就是：

```text
TARGET / total_tokens * 1.05
```

进行 oversampling。

你的 pipeline 后面还要 dedup/filter，所以建议：

```text
oversample = 1.15～1.20
```

最终让 20B 训练集先产生：

```text
~23B candidate
```

而不是对整个 200B+ corpus 做昂贵处理。

---

# 十一、Stage 7：Exact Dedup

先做最便宜的：

```text
normalized document hash
```

工具可以：

```text
xxhash
DataTrove exact dedup
```

保留：

```text
source priority
quality score
```

当重复时：

```text
higher quality source wins
```

建议 source priority：

```text
canonical structured source
>
high-quality curated corpus
>
general web
>
synthetic
```

例如：

```text
Wikipedia > random web mirror
ect-krp > web copy of classical text
FineWeb-Edu > generic FineWeb
```

---

# 十二、Stage 8：MinHash near-dedup

直接用：

```text
DataTrove MinHash
```

不要自己再实现大规模 SQLite MinHash。

FineWeb 使用 MinHash 去重；DataTrove 也直接提供完整 MinHash pipeline 示例。

初始配置建议：

```text
5-gram shingles
Jaccard ≈ 0.8
```

具体参数之后可以通过抽样检查调。

输入只剩：

```text
~22B
```

而不是全部原始 corpus。

---

# 十三、Stage 9：Benchmark decontamination

这个必须在最终 training selection 前做。

准备：

```text
data/eval_contamination/
```

至少：

```text
C-Eval
CMMLU
MMLU
ARC
GSM8K
HumanEval/MBPP

SIGHAN
CSCD-NS
LEMON
ECSpell
FCGEC
NaSGEC

ACLUE
AC-EVAL
C3Bench
Fuxi
```

可以使用：

```text
DataTrove decontamination
```

当前 DataTrove甚至单独有 `decont` extra；Dolma 的公开 decontamination 方案则使用文档/段落 Bloom filter。

第一版不用做特别复杂：

```text
exact paragraph
+
13-gram overlap
```

即可。

---

# 十四、Stage 10：长尾汉字挖掘

这是 CC-LLM 自己的 custom block。

先从整个候选中文池统计：

```text
char_frequency
```

生成：

```text
hanzi_frequency.parquet
```

然后为每个文档计算：

```text
rare_hanzi_score
traditional_score
classical_score
polyphone_density
```

例如：

\[
Rare(d)
=
\sum_{c\in d}
\frac{1}{\sqrt{freq(c)+1}}
\]

然后构造：

```text
longtail candidate pool
```

注意：

> 从原类别中移出，不能重复进入 5% 长尾 bucket。

否则相同文档会被训练两遍。

---

# 十五、Stage 11：最终 Mixture Selection

这时才真正锁定 20B。

建议生成一个：

```yaml
mixture.yaml
```

例如：

```yaml
total_tokens: 20000000000

groups:
  zh_general:
    tokens: 10000000000
    sources:
      cci3_hq: 0.50
      fineweb_edu_zh: 0.35
      wanjuan: 0.15

  zh_knowledge:
    tokens: 3000000000

  en_multi_mixed:
    tokens: 3000000000

  math_code_science:
    tokens: 3000000000

  hanzi_longtail:
    tokens: 1000000000
```

再增加横向约束：

```yaml
constraints:
  traditional_min: 0.05
  classical_min: 0.03
  long_document_min: 0.10
  synthetic_max: 0.10
  source_max: 0.30
```

这里特别建议加：

```text
source_max
```

防止某个超大 corpus 占据整个 mixture。

---

# 十六、Stage 12：Exact Tokenization

到这里才真正 tokenize。

现在输入已经约：

```text
20～21B exact selected text
```

使用：

```text
tokenizers
```

或者 DataTrove tokenization block。

不要用 Python：

```python
for text:
    AutoTokenizer(text)
```

推荐 batch。

你现在 tokenizer 已经是标准 HF tokenizer，因此 DataTrove/Dolma 都可以直接消费。

Dolma 的 tokenizer 也支持任意 HF-compatible tokenizer，并且专门针对多文件并行 tokenization 优化。

但既然主框架选 DataTrove，就保持统一。

---

# 十七、Stage 13：长度处理

这一步不要在文本阶段做很多重复 tokenize。

已经有：

```text
input_ids
```

之后直接 token-level。

建议 Phase2：

```text
2K / 4K / 8K
```

而不是所有样本固定 8K。

第一版可以简单：

```text
50%  2K
35%  4K
15%  8K
```

这里是训练 sequence 分布，不是 source mixture。

长文本：

```text
保留原文档内部连续 chunks
```

优先：

```text
document A section1
document A section2
```

而不是随机拼。

---

# 十八、Stage 14：Packing

最终：

```text
doc A
<EOS>
doc B
<EOS>
```

直到：

```text
seq_length
```

packing 完成。

PGCA 不需要额外保存 feature。

所以最终训练样本只需要：

```text
input_ids
attention metadata / boundaries（如果训练框架需要）
```

不用保存：

```text
pinyin_ids
stroke_ids
...
```

模型内部 feature index lookup 即可。

---

# 十九、最终输出格式

推荐：

```text
data/packed/phase2/
├── 2k/
│   ├── shard-00000.parquet
│   └── ...
├── 4k/
├── 8k/
└── manifest.json
```

如果你的 DeepSpeed/Swift trainer 更适合：

```text
.npy / mmap / Arrow
```

可以在最后增加一次格式转换。

前面的 corpus 不动。

---

# 二十、Stage 15：Final QA

训练前必须生成一个完整：

```text
phase2_dataset_report.json
```

至少包含：

### Token

```text
total_tokens
tokens_by_source
tokens_by_domain
```

### Quality

```text
quality_score distribution
removed_by_filter
```

### Dedup

```text
exact duplicates removed
MinHash clusters
dedup removal ratio
```

### Language

```text
zh/en/multilingual
```

### Chinese-specific

```text
simplified
traditional
classical

Hanzi coverage
rare Hanzi coverage
pinyin coverage
polyphone coverage
radical coverage
structure coverage
feature missing rate
```

### Length

```text
document P50/P90/P99
2K/4K/8K distribution
long-document token %
```

### Contamination

```text
benchmark match count
removed count
remaining suspicious
```

只有这些通过以后才能启动 CPT。

---

# 二十一、DataTrove 的代码组织

项目中我建议最终变成：

```text
scripts/data_factory/
├── phase2/
│   ├── config.py
│   ├── sources.py
│   ├── normalize.py
│   ├── quality.py
│   ├── classify.py
│   ├── longtail.py
│   ├── decontaminate.py
│   ├── mixture.py
│   ├── tokenize.py
│   ├── pack.py
│   └── run.py
│
└── common/
    ├── hanzi.py
    ├── hashing.py
    └── stats.py
```

其中 DataTrove 负责：

```text
Reader
Writer
Document
executor
SamplerFilter
LanguageFilter
MinHash
stats
tokenization
```

你自己只写：

```text
CCLLMNormalize
CCLLMQuality
ChineseDomainClassifier
MixedZhEnClassifier
RareHanziScorer
MixturePolicy
```

---

# 二十二、20B 规模如何实际并行

如果服务器有 Slurm：

我建议直接：

```text
SlurmPipelineExecutor
```

DataTrove 官方支持 job arrays，并允许设置 task 数和并发 worker 数。

例如：

```text
cache          128 tasks
filter         128 tasks
sampling       128 tasks
MinHash sig    256 tasks
MinHash bucket 128 tasks
tokenize       128 tasks
```

具体不是越大越好，取决于文件数和存储吞吐。

如果只有一台服务器：

```text
LocalPipelineExecutor
```

例如：

```text
32～64 workers
```

但前提是 NVMe/高速共享盘跟得上。

---

# 二十三、推荐 requirements

建议至少：

```text
datatrove[io,processing,multilingual]
pyarrow
tokenizers
transformers
fasttext-numpy2-wheel
xxhash
orjson
ftfy
numpy
tqdm
```

集群：

```text
datatrove[ray]
```

或者直接用 Slurm executor，不一定需要 Ray。

DataTrove 当前官方把 `io`、`processing`、`multilingual`、`ray`、`decont` 等作为独立 optional extras。

---

# 二十四、最终正式 pipeline

所以你的 Phase2 完整数据管线应该固定为：

```text
             Raw Corpus
                 │
                 ▼
      1. DataTrove source reader
                 │
                 ▼
      2. Normalize / basic clean
                 │
                 ▼
       3. Language + type tagging
                 │
                 ▼
       4. Cheap quality filtering
                 │
                 ▼
       5. Quality score / tagging
                 │
                 ▼
          Parquet Corpus Cache
                 │
                 ▼
       6. Token calibration 50K/source
                 │
                 ▼
      7. Estimated-token sampling
                 │
               ~23B
                 ▼
          8. Exact dedup
                 │
                 ▼
         9. MinHash dedup
                 │
                 ▼
      10. Benchmark decontamination
                 │
                 ▼
   11. Rare/traditional/classical scoring
                 │
                 ▼
      12. Exact mixture selection
                 │
               20B
                 ▼
       13. Batch exact tokenizer
                 │
                 ▼
      14. 2K/4K/8K token slicing
                 │
                 ▼
             15. Packing
                 │
                 ▼
       Pre-tokenized Train Shards
                 │
                 ▼
             16. QA Report
```

这版和你现在 Phase1 数据工厂相比，最重要的三项变化是：

**第一，DataTrove 替代 SQLite 成为主数据执行框架。**

**第二，语义质量筛选和数据分类发生在候选抽样之前，但 MinHash/tokenization 等昂贵步骤发生在候选抽样之后。**

**第三，20B 最终数据只精确 tokenize 一次，PGCA feature 不进入训练数据文件。**

这就是我建议你真正作为论文 Phase2、同时也作为后续长期模型训练基础设施维护的正式方案。