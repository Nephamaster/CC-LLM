# 数据构建方案

## 数据统一格式说明

语义对齐阶段和全参继续预训练阶段的数据组织方式如下：

```jsonl
{"text": "...", "source": "github", "repo": "xxx/yyy", "license": "apache-2.0"}
{"text": "...", "source": "hf_model", "repo": "Qwen/Qwen3-1.7B", "license": "unknown"}
{"text": "...", "source": "Chinese Fineweb Edu Dataset V2.2", "doc_id": "fineweb_edu_chinese_v2.2-4_5-000000-234", "license": "..."}
```

其中:
- `source` 指的是该数据来源，如果是从本地现成数据集获取则填数据集名称；从Github或HF获取的doc等需要标明 "github" 或 "hf_model" 或 "hf_dataset"
- `repo` 仅当进行网站数据爬取时需要填写对应的仓库名称，否则不写该字段
- `doc_id` 需要填写该条数据对应的文件所在的子集，如 CLUE benchmark 的 afqmc 子目录的 train-00000-of-00001.parquet 的第 10 条 doc，则 doc_id 为 clue-afqmc-train-10；Chinese Fineweb Edu Dataset V2.2 的 4_5 子目录的 000000.parquet 的第 234 条 doc，则 doc_id 为 fineweb_edu_chinese_v2.2-4_5-000000-234。即以数据集在服务器的名称、子目录、文件名、条目序号用'-'链接，组成部分中包含'-'的，将其转为'_'
- `license` 按照来源的证书填写，github/hf仓库的就填写仓库证书，数据集填数据集证书
- 对网页、GitHub 和 Hugging Face 数据，额外保留 `url`、文件相对路径 `path`、固定版本 `revision`（commit SHA 或数据快照日期）。这些字段用于复现和履行署名义务，不参与训练文本。
- `license` 必须是可核验的 SPDX/Creative Commons 标识。许可缺失、识别为 `NOASSERTION`/`unknown`、仓库许可证与文件头冲突的数据只能进入隔离区，不能进入最终训练集。
- `doc_id` 在 Phase 1 和 Phase 2 全局唯一。最终数据建议同时写入 `content_hash`，用于跨阶段去重审计。

### 数据存放路径

Phase 1 数据：`data/semantic_alignment/`

Phase 2 数据：`data/continue_pretrain/`

数据处理脚本存放路径：`scripts/data/`

每个阶段目录内部统一划分为：

```text
raw_manifest/    # 原始文件清单、URL、revision、license，不复制无法再分发的原文
normalized/      # 完成解析和基础清洗、尚未全局去重的数据
deduplicated/    # 完成阶段内及跨阶段去重的数据
final/           # 按 token 配比采样并分片后的训练 JSONL
reports/         # 数量、token、来源、许可、过滤和去重统计
```

## 数据去重

Phase 1 和 Phase 2 之间不能重复，各阶段内部也不能重复。去重在文本清洗之后、token 计数和配比采样之前执行，原始数据保持只读。

### 统一流程

1. **规范化副本**：将文本转换为 UTF-8 和 Unicode NFC，统一换行符，删除首尾空白、零宽字符及连续网页导航/页脚；保留大小写、段落、Markdown、代码缩进和繁简差异。规范化仅用于哈希和比较，不能使用会改变字符语义的 NFKC。
2. **文档内去重**：删除同一文档中重复出现的完整段落、导航栏、版权页脚和模板块；过短文本（少于 50 个字符）只做精确去重。
3. **精确去重**：对规范化文本计算 SHA-256。相同哈希只保留一个代表文档，并将哈希、保留 `doc_id`、被删除 `doc_id` 和原因写入去重报告。
4. **近似去重**：
   - 中文及中英混排文本使用 Unicode 字符 5-gram；英文自然文本使用词元 5-gram；计算 128 维 MinHash，通过 LSH 召回候选，估计 Jaccard 相似度不低于 0.80 时归为同一重复簇。
   - 代码和结构化文本按非字母数字边界切分词法 token，使用 token 5-gram、256 维 MinHash；Jaccard 相似度不低于 0.85 时归为重复簇。该阈值和处理方式参考 The Stack 的代码近似去重方案。
5. **代表样本选择**：同一重复簇优先保留许可明确、质量分高、来源可追溯、正文完整的版本；质量相同时保留更早进入管线的版本，保证结果可复现。不能通过复制或重复采样补足某一类别 token 配额。
6. **跨阶段去重**：维护全局 `dedup_registry.parquet`，至少记录 `content_hash`、MinHash 签名、保留 `doc_id`、阶段和类别。先构建并冻结 Phase 1 索引；Phase 2 的每条候选数据先查 Phase 1，再进行 Phase 2 内部去重。
7. **评测集去污染**：最终训练数据还需对项目使用的验证集和公开 benchmark 做相同的精确哈希与近似匹配，命中项不得进入训练集。

实现优先使用 DataTrove 已有的 exact/sentence/MinHash 去重组件；代码数据单独使用词法签名，不能与自然语言共用规范化规则。每次运行输出输入量、精确重复量、近似重复量、跨阶段命中量和各来源保留率。

## 已有数据说明

### CCI3-HQ

- 服务器路径：`/share/project/wuhaiming/data/dataset/CCI3-HQ/data`

- 子文件路径（一共100个文件）：
    ```bash
    part_000000.jsonl
    ...
    part_000099.jsonl
    ```
- 原始数据结构：
    ```jsonl
    {"id": "b91c08f50c45d0f9c163fb2efcccc6c2", "text": "曾巩:为人廉洁奉公,才华横溢,关心民间疾苦曾巩,字子固,是我国北宋时期著名的文学家,政治家和教育家...", "score": 3.7735397815704346}
    ```
    直接提取 `text` 即可

### WanJuan

- 服务器路径：`/share/project/wuhaiming/data/dataset/WanJuan1_dot_0/raw/nlp/CN/`

- 子文件路径（一共100个文件）：
    ```bash
    ChinaNews-cn/part-xxxxxx-yyyyyyyy.jsonl.tar.gz
    Exam-cn/part-xxxxxx-yyyyyyyy.jsonl.tar.gz
    Law-cn/part-xxxxxx-yyyyyyyy.jsonl.tar.gz
    Patent-cn/part-xxxxxx-yyyyyyyy.jsonl.tar.gz
    WebText-cn/part-xxxxxx-yyyyyyyy.jsonl.tar.gz
    ```
    也就是说每个子目录下的文件命名方式如上，文件有若干个

- 原始数据结构：
    - Exam-cn:
        ```jsonl
        {"id":"BkQQU-7xK3YAJdm0cWMc","q_type":"单选题","q_main":"下列属于同种物质的是（）","option_a":"冰和水","option_b":"铁和铁锈","option_c":"镁和氧化镁","option_d":"金刚石和石墨","option_e":"","std_ans":"A","answer":"","answer_detail":"A、冰时由H、O元素组成，水是由H、O元素组成，化学式都是H2O，属于同种物质，故A正确；  \nB、铁中只有铁元素，而铁锈中有铁和O元素，显然组成元素不同>，不属于同种物质，故B错误；  \nC、镁中只有镁元素，氧化镁中有镁和氧元素，组成元素不同，不属于同种物质，故C错误；  \nD、金刚石和石墨都是碳元素组成的，但碳原子的排列方式不同，结构不同，不属>于同种物质，故D错误；  \n故选A","grade":"九年级","major":"化学","keypoint":"化学式的书写及意义"}
        ```
        提取 q_mean 和 answer_detail，组成 text
    - Law-cn:
        ```jsonl
        {"id":"BkQfC-7xK2liq9BBV_mq","content":"陕西省咸阳市杨陵区人民法院\..."}
        ```
        只需提取 content
    - Patent-cn:
        ```jsonl
        {"id":"BkR2V-3xK0fsDjvRg0m5","content":"本实用新型涉及一种天车滑线伸缩缝连接结构，包括有天车滑线A和天车滑线B..."}
        ```
        只需提取 content
    - WebText-cn:
        ```jsonl
        {"id":"BkPn-b3xK3xiB1hk6SFp","content":"# 【消防课程】生活中的安全用电小常识\n为丰富学生安全实训经历，帮助学生形成正确的安全意识，了解基本的安全常识..."}
        ```
        只需提取 content
    - ChinaNews-cn:
        ```jsonl
        {"id":"BkPn-b3xK3xiB1hk6SFp","content":"前几周，姗姗来迟的京东终于推出了自营品牌“京造”，加入了网易严选和米家有品组成的自营..."}
        ```
        只需提取 content

### CLUE benchmark 训练集

- 服务器路径：`/share/project/wuhaiming/data/dataset/clue/`

- 子文件路径：
    ```bash
    afqmc/train-00000-of-00001.parquet
    c3/train-00000-of-00001.parquet
    chid/train-00000-of-00001.parquet
    cluewsc2020/train-00000-of-00001.parquet
    cmnli/train-00000-of-00001.parquet
    cmrc2018/train-00000-of-00001.parquet
    csl/train-00000-of-00001.parquet
    drcd/train-00000-of-00001.parquet
    iflytek/train-00000-of-00001.parquet
    ocnli/train-00000-of-00001.parquet
    tnews/train-00000-of-00001.parquet
    ```

- 原始数据结构
    - afqmc
        ```python
        import pandas as pd

        data = pd.read_parquet('/share/project/wuhaiming/data/dataset/clue/afqmc/train-00000-of-00001.parquet')
        print(data.head())

        ## 列名：sentence1  sentence2   label   idx
        0   蚂蚁借呗等额还款可以换成先息后本吗  借呗有先息到期还本吗    0   0
        ```
    - c3: head 为 id context question choice answer，需要拼接的是context、question和answer
    - chid：
        ```json
        {'idx': 0, 'candidates': array(['传宗接代', '得过且过', '咄咄逼人', '碌碌无为', '软硬兼施', '无所作为', '苦口婆心', '未雨绸缪',
        '和衷共济', '人老珠黄'], dtype=object), 'content': array(['谈到巴萨目前的成就，瓜迪奥拉用了“坚持”两个字来形容。自从上世纪90年代克鲁伊夫带队以来，巴萨就坚持每年都有拉玛西亚球员进入一队的传统。即便是范加尔时代，巴萨强力推出的“巴萨五鹰”德拉·佩纳、哈维、莫雷罗、罗杰·加西亚和贝拉乌桑几乎#idiom000000#的情况下，他们依然在坚持， 最终他们等到了哈维的成熟，等到了普约尔、梅西、伊涅斯塔的横空出世。',
        '可是，至少现在我们已经看到了一种清楚的方法来为那些糟糕的资产解扣，或者，更为重要的是，能够确定它们的价值。几乎所有投资者都会相信，只要我们向着这个方向努力，无论我们采取怎样的措施，其结果都会好过#idiom000001#，只是在那里坐等又一份宣布减记消息的银行报告。',
        '股指蹒跚，与经济基本不确定性相关联。目前中国经济的格局是无近忧，有远虑。一般而言，政府自己比谁都清楚问题在哪，如果出事最可能发生在哪？温总理也认为去年是最困难的，今年是最复杂的，明年是什么呢？难说。国资委上周五通知78家不以房地产为主业的国有企业在15个工作日以内制定出如何退出房地产业务的计划，这些企业必须加快重组过程。并且国资委主任李荣融在与国有企业举行的会议上强调，中央直属的房地产开发商，尤其是这些公司的高层必须进一步加强对政府房地产开发政策措施、关注民生政策和房地产市场趋势的理解。中央企业必须认识到当前经济形势的复杂性，把防范风险放在首要位置，加强成本控制。我们认为这是李荣融主任委婉提示房地业的风险，并#idiom000002#劝说央企地产公司“不要崽卖爷田，不心痛”，要认识到当期经济的复杂性，关注房地产市场趋势及其蕴含的风险，控制成本，不要认为房价只升不降，可以无限吸收“不断膨胀的地王”成本。政府要求非主营地产业务的央企退出地产业，控制“政府军”的地产敞口，以免到时出事，“一只老鼠坏了一锅汤”。神光认为，中国经济未来如出现大的地震，震中很可能来自地产业，在资产市场中，股市政府可以用扩 容与做空制度来操控泡沫，房地产因各方面利益纠缠不清，政府难以重拳刺泡沫，只能#idiom000003#，但是泡沫迟早会破的，政府已经#idiom000004#了，投资者是否应该跟随呢！',
        '王英英踏进狼窝再想拔足就难了，任百万#idiom000005#，结果这头胎就让任百万给种下了一个儿子。任大壮在部队听说英英给他生了个#idiom000006#的，高兴地从部队赶了回来。任洪义不知这盆水有多深，全当是英英争了气，多少妇女头胎全是女孩，他张罗给孩子办了一个很气派、很体面的满月。'],
        dtype=object), 'answers': {'text': array(['碌碌无为', '无所作为', '苦口婆心', '得过且过', '未雨绸缪', '软硬兼施', '传宗接代'],
        dtype=object), 'candidate_id': array([3, 5, 6, 1, 7, 4, 0], dtype=int32)}}
        ```
        要获取最终完整正确的句子

    - cluewsc2020：
        ```json
        {'idx': 0, 'text': '裂开的伤口涂满尘土，里面有碎石子和木头刺，我小心翼翼把它们剔除出去。', 'label': 1, 'target': {'span1_text': '伤口', 'span2_text': '它们', 'span1_index': 3, 'span2_index': 27}}
        ```
        提取text作为最终文本即可

    - cmnli：
        ```json
        {'sentence1': '从概念上讲，奶油略读有两个基本维度-产品和地理。', 'sentence2': '产品和地理位置是使奶油撇油起作用的原因。', 'label': 0, 'idx': 0}
        ```
        两个句子都要，作为两个条目

    - cmrc2018：
        ```json
        {'id': 'TRAIN_186_QUERY_0', 'context': '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世...', 'question': '范廷颂是什么时候被任为主教的？', 'answers': {'text': array(['1963年'], dtype=object), 'answer_start': array([30], dtype=int32)}}
        ```
        一个 context 会有多个question，只提取context即可，因此按照 id 中 "TRAIN_" 后面紧跟的序号作为去重依据
    - csl：
        ```json
        {'idx': 0, 'corpus_id': 1, 'abst': '目的探讨常见氧化铁纳米粒子几种神经干细胞标记技术的标记效率.材料与方法使用超顺磁性氧化铁纳米粒子(SPIO)...', 'label': 0, 'keyword': array(['粒子', '铁化合物', '联合', '0.05'], dtype=object)}
        ```
        只提取 abst，但多个条目之间的 abst 有可能重复

    - drcd：
        ```json
        {'id': '1001-10-1', 'context': '2010年引進的廣州快速公交運輸系統，屬世界第二大快速公交系統，...', 'question': '廣州的快速公交運輸系統每多久就會有一輛巴士？', 'answers': {'text': array(['10秒鐘'], dtype=object), 'answer_start': array([84], dtype=int32)}}
        ```
        只提取context，但依旧会出现多个条目之间context一样的情况，根据 id 中的 xxxx-yy-z 中的 xxxx-yy 做去重

    - iflytek：提取 `sentence` 键即可
    - ocnli： 忽略这个分类
    - tnews：提取 `sentence` 键即可


### Chinese Fineweb Edu Dataset V2.2

- 服务器路径：`/share/project/wuhaiming/data/dataset/fineweb-edu-chinese-v2.2/4_5/`

- 子文件路径（一共9767个文件）：
    ```bash
    000000.parquet
    ...
    009766.parquet
    ```

- 原始数据结构：
    ```python
    import pandas as pd

    data = pd.read_parquet('/share/project/wuhaiming/data/dataset/fineweb-edu-chinese-v2.2/4_5/000000.parquet')
    print(data.head())

    ## 列名：text   score   source                                                                            
    0   一、公司清算所有者权益怎么分配\n公司清偿所有债务后，有剩余财产的，所有权权益由股东按出资的...   0.801270    CCI3
    1   分散采购的优缺点（ 集中采购与分散采购该如何进行选择）\n你是否经常为采购流程的效率和效果而...    0.817383    CCI3
    ```

### FineWeb-Edu

- 服务器路径：`/share/project/wuhaiming/data/dataset/fineweb-edu/sample/10BT`

- 子文件路径（一共14个文件）：
    ```bash
    000_00000.parquet
    ...
    013_00000.parquet
    ```

- 原始数据结构：
    ```python
    import pandas as pd

    data = pd.read_parquet('/share/project/wuhaiming/data/dataset/fineweb-edu/sample/10BT/000_00000.parquet')
    print(data.head())

    ## 列名：text   id  dump    url file_path   language    language_score  token_count score   int_score
    0  The Independent Jane\nFor all the love, romanc...  <urn:uuid:0d8a309d-25c5-405d-a08a-c11239f0d717>  CC-MAIN-2013-20      http://austenauthors.net/the-independent-jane  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/se...       en        0.974320          845  2.750000          3
    ```

## Phase 1 - 语义空间对齐数据

### 整体数据配比

| 数据类别 | 比例 | 目标 token | 对齐目标 |
| --- | ---: | ---: | --- |
| 中文自然高质量文本 | 45% | 450M | 对齐常用单字序列和通用中文语义 |
| 被裁多字 token 桥接文本 | 15% | 150M | 让原多字 token 的语义由单字序列重新表达 |
| 新增汉字覆盖文本 | 5% | 50M | 为新增和低频汉字 embedding 提供真实上下文 |
| 英文及非中文保持数据 | 15% | 150M | 约束原有非汉字 embedding，减少能力遗忘 |
| 中英混排文本 | 10% | 100M | 对齐汉字与英文、数字、URL 和术语的边界 |
| 代码、数学和结构化文本 | 10% | 100M | 保持代码、公式、符号和结构化格式能力 |

数据总量为 1B token，全部在清洗和去重后使用新 tokenizer 计算。六类数据使用互不重复的文档，类别和总量误差均不得超过 ±1%。

Phase 1 的目标不是一般领域继续预训练，而是建立新词表与冻结 Backbone 之间的语义映射。数据构建必须同时覆盖新增汉字、被裁多字 token 的单字组合，以及原有非汉字 token；不能单纯提高中文网页语料占比。

### Phase 1 数据构建与统一约束

各类别配额均以清洗、去重后使用新 tokenizer 统计的 token 数为准。不得通过复制样本、重复单字、字表罗列或低质量合成文本补足配额；来源不足时必须补充同类别的合规来源，配额不足不得验收通过。

构建前先生成 `reports/source_inventory.json`，逐源记录文档数、原始字符数、新 tokenizer token 数、许可证分布、无效样本数和去重后预计可用量。任何单一中文来源不得超过中文部分的 40%，避免模型过度适配单一网页风格。

已有来源按以下规则处理：

- **中文自然高质量文本**：从 OpenCSG Fineweb-Edu-Chinese-V2.2、CCI3-HQ 和 WanJuan 中抽取。使用正文并保留原始来源字段，过滤乱码、模板页、低信息密度和 PII。
- **CLUE benchmark 训练集**：不再作为 Phase 1 中文主体。其标签移除后主要是任务片段，对词表语义对齐的价值有限。默认从 1B 训练数据排除；现有格式处理规则仅供独立诊断使用。
- **英文及非中文保持数据**：使用 FineWeb-Edu sample-10BT，仅保留英语正文，建议 `language_score >= 0.90`，完成全局去重后采样 0.15B token。

中文自然文本按自然段或句子边界切成 128 到 512 个新 token 的窗口，最长不超过 1024 token。不得在代码围栏、公式或表格内部截断，也不使用任意位置切分的 2048-token 长文档作为 Phase 1 主体。

### 被裁多字 token 桥接文本（0.15B token）

以词表构建阶段删除的多汉字 token 为检索词表，在中文候选语料中抽取包含这些词的真实句段，使冻结 Backbone 重新适应由多个单字表示同一词语的输入形式。

1. 使用词表构建阶段相同的解码和分类规则，从原 tokenizer 导出被裁多汉字 token，并新增中间产物 `removed_multi_hanzi_tokens.json`；不包含 mixed-hanzi、special token 和异常 token。`semantic_vocab_manifest.json` 继续记录数量及该清单的 SHA256。
2. 在去重后的中文候选语料中扫描命中位置，按句子或自然段提取 128 到 512 个新 token 的上下文。
3. 按候选频率分层抽样：高频词限制重复，低频词提高抽样概率；同一词优先选择不同文档和不同左右上下文。
4. 候选频率最高的 5,000 个被裁 token 原则上各保留至少 256 次上下文；可用上下文不足时保留全部并报告缺口，不复制样本。
5. 桥接文本与中文自然文本使用不同文档，避免同一内容重复计入两个配额。

### 新增汉字覆盖文本（0.05B token）

以 `new_hanzi_token_ids.json` 中的新增汉字 token 为目标，从中文候选语料中定向选择真实上下文。

1. 先统计中文自然文本和桥接文本中的新增汉字频次及不同文档数，再只针对覆盖缺口补充。
2. 候选语料中出现过的新增汉字至少保留一次；《通用规范汉字表》、常用繁体字表和 `rare_high_freq.txt` 中的新增字，至少 95% 达到 128 个不同文档上下文。
3. 同一汉字优先选择来源、主题及左右邻接词不同的样本，并限制相对自然分布的过采样倍数。
4. 没有自然上下文的极罕见字保留初始化 embedding，并在报告中列为未覆盖；不使用连续单字、字表释义模板或重复生成文本强行覆盖。
5. 覆盖报告分别统计新增字的输入出现次数、预测目标次数、不同文档数和来源数。

### 中英混排数据（0.1B token）

优先从已有 Fineweb-Edu-Chinese、CCI3-HQ 和 WanJuan 中筛选自然混排文本，再使用显式许可的 GitHub/Hugging Face 文档、Wikimedia、OpenAlex 和技术站点补充。

有效混排样本须满足：汉字占可见文字的 20% 到 80%，拉丁字母或英文词占 5% 到 50%，并至少存在两处有效中英文边界。URL、文件名、导航栏和版权信息造成的英文命中不计入；代码占比过高的样本归入代码类别。

采集外部文本时继续遵守以下约束：

1. GitHub/Hugging Face 只接收许可证明确且允许用于当前研究的数据，固定 commit 或 revision，并记录来源和许可证。
2. Wikimedia 使用官方 dump；OpenAlex 只使用其提供的开放元数据，不抓取链接论文全文。
3. HTML 去除导航、广告、评论和重复页脚；Markdown/RST 保留标题、列表、表格和行内术语。
4. 单仓库最多占混排类别的 0.5%，单站点最多占 5%。
5. 某一来源不足时只能由其他合规混排来源补足，不能使用代码或纯英文数据回填。

### 代码、数学和结构化文本（0.1B token）

| 子类 | 比例 | 目标 token | 主要来源 |
| --- | ---: | ---: | --- |
| 代码 | 40% | 40M | GitHub 显式许可仓库 |
| 数学和科学文本 | 40% | 40M | FineWeb-Edu、Wikimedia 数学条目、可用的 Stack Exchange dump |
| Markdown/结构化格式 | 20% | 20M | GitHub 文档与 JSON、YAML、TOML、XML 文件 |

代码排除二进制、vendor、生成文件、lock 文件、超大文件、重复模板和疑似密钥；近重复采用词法 token 签名。数学文本保留自然语言解释和 LaTeX/MathJax 上下文。JSON、YAML、TOML 和 XML 必须能够由对应解析器解析，且保持原始层级和缩进。

三个子类必须分别达到配额。数学或结构化数据不足时不得使用代码数据回填。

### 唯一分组与采样

1. 抽取正文并标准化为统一 JSONL，保留可追溯元数据。
2. 执行编码、正文质量、PII/密钥、语种和格式可解析性过滤。
3. 执行文档内、精确和 MinHash 近重复去重，并使用 Phase 0 及下游评测集做污染检查。
4. 使用新 tokenizer 统计每个文档的 token 数、新增汉字命中、被裁多字 token 命中、语言比例及代码/数学特征。
5. 按“新增汉字覆盖、被裁多字 token 桥接、中英混排/专项、普通语料”的优先级将文档唯一分组，防止同一文档跨类别重复计数。
6. 按类别、来源、领域和长度分层采样；使用固定随机种子打乱并输出 shard。
7. 写入 manifest、文件哈希、统计报告和失败样本，不因单条坏数据中断整批构建。

### Phase 1 验证集

验证数据在训练采样前按文档划分，不能从已经入选训练集的 shard 中抽取。

- `validation_natural.jsonl`：约 2.5M token，保持训练集的自然类别分布，用于 `eval_loss` 和 checkpoint 选择。
- `validation_alignment.jsonl`：约 1M token，平衡包含新增汉字、被裁多字 token、原有汉字和非汉字 token 的文本，用于诊断各 token 组的对齐情况，不作为唯一的最佳模型判据。

验证报告分别计算新增汉字、原有汉字、英文及非中文、中英混排、代码和数学文本的 loss，并记录 5-shot 中文评测 prompt 在新 tokenizer 下的长度及截断比例。

### Phase 1 验收条件

- 总量和六个一级类别的 token 误差不超过 ±1%。
- 代码、数学和结构化三个子类分别达到配额，不能相互回填。
- 未知或冲突许可证样本数为 0，全局精确重复数为 0，训练集与验证集文档重叠数为 0。
- 新增汉字和被裁多字 token 达到本节覆盖要求；未覆盖目标必须有真实候选不足的统计依据。
- 输出来源集中度、近重复率、序列长度分布、token 分组频次和评测污染报告。
- 任一类别未达配额或关键报告缺失时，整体构建报告必须为 `passed=false`。

### 参考依据

- [Bilingual Adaptation of Monolingual Foundation Models](https://arxiv.org/abs/2407.12869)：先训练新 embedding、再进行全参数继续预训练的两阶段适配方法及原语言数据回放。
- [An Empirical Comparison of Vocabulary Expansion and Initialization Approaches](https://aclanthology.org/2024.conll-1.8/)：新增词元初始化和目标语料继续预训练的实验依据。
- [FineWeb: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)：网页语料过滤、去重和质量控制流程。
- [DataTrove](https://github.com/huggingface/datatrove)：大规模精确、句子级及 MinHash 去重实现。
- [The Stack: 3 TB of permissively licensed source code](https://arxiv.org/abs/2211.15533)：代码许可筛选和近重复去重方法。
- [CLUECorpus2020](https://arxiv.org/abs/2003.01355) 与 [CLUE benchmark 数据集卡](https://huggingface.co/datasets/clue/clue/blob/main/README.md)：区分大规模中文预训练语料与现有 benchmark 子任务；Phase 1 默认不将 benchmark 训练集计入 1B 配额。
- [OpenCSG Fineweb-Edu-Chinese-V2.2](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.2) 与 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)：数据规模、字段和许可证。
- [GitHub repository licenses API](https://docs.github.com/en/rest/licenses/licenses)、[repository contents API](https://docs.github.com/en/rest/repos/contents) 和 [rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)：许可核验、版本化获取与抓取限速。
- [Wikimedia dump licensing](https://dumps.wikimedia.org/legal.html)、[OpenAlex About](https://help.openalex.org/hc/en-us/articles/24396686889751-About-us) 和 [Stack Overflow licensing](https://stackoverflow.com/help/licensing)：各来源的许可与归因要求。
- [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use)：Common Crawl 不替代原网页内容许可。
