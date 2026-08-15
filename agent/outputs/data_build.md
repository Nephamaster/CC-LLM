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

## Phase 1 - 语义空间对齐数据

### 整体数据配比

|数据类别	|比例	|说明|
|:---|:---|:---|
|中文通用短文本 |40% |CLUE benchmark 训练集；不足部分由 OpenCSG 中文语料补足|
|中文高质量知识文本	|20%	|OpenCSG Fineweb-edu-chinese|
|中英混排	|15%	|README、API 文档、中文技术博客、论文摘要|
|非中文保持	|15%	|FineWeb-Edu|
|代码/数学/格式/汉字覆盖补充	|10%	|代码、JSON、Markdown、罕见字、繁体、special token|

数据总量：2B tokens （新 tokenizer 计算）

### 已有数据说明

- **CLUE benchmark 训练集**

    服务器路径：`/share/project/wuhaiming/data/dataset/clue/`

    子文件路径：
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

    原始数据结构：
    ```python
    import pandas as pd

    data = pd.read_parquet('/share/project/wuhaiming/data/dataset/clue/afqmc/train-00000-of-00001.parquet')
    print(data.head())

    ## 列名：sentence1  sentence2   label   idx
    0   蚂蚁借呗等额还款可以换成先息后本吗  借呗有先息到期还本吗    0   0
    1   蚂蚁花呗说我违约一次    蚂蚁花呗违约行为是什么  0   1
    2   帮我看一下本月花呗账单有没有结清    下月花呗账单    0   2
    3   蚂蚁借呗多长时间综合评估一次    借呗得评估多久  0   3
    4   我的花呗账单是***，还款怎么是***    我的花呗，月结出来说让我还***元，我自己算了一下详细名单我应该还***元    1   4  
    ```

    - c3 的 head 为 id context question choice answer，需要拼接的是context、question和answer

    - chid 的数据结构如下例所示：
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
    两个句子都要，作为两个条目
    ```

    - cmrc2018：
    ```json
    {'id': 'TRAIN_186_QUERY_0', 'context': '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业。范廷 颂于1940年在河内大修道院完成神学学业。范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务。1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕。1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作。范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望 保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主 教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。', 'question': '范廷颂是什么时候被任为主教的？', 'answers': {'text': array(['1963年'], dtype=object), 'answer_start': array([30], dtype=int32)}}
    ```
    一个 context 会有多个question，只提取context即可，因此按照 id 中 "TRAIN_" 后面紧跟的序号作为去重依据

    - csl：
    ```json
    {'idx': 0, 'corpus_id': 1, 'abst': '目的探讨常见氧化铁纳米粒子几种神经干细胞标记技术的标记效率.材料与方法使用超顺磁性氧化铁纳米粒子(SPIO)和超微超顺磁性氧化铁纳米粒子(USPIO)以25μgFe/ml分别单独标记、与多聚赖氨酸(PLL)及脂质体联合标记神经干细胞,以未标记细胞做对照,采用普鲁士蓝染色评价细胞标记率,并采用4.7TMRIT2WI多回波序列测量T2弛豫率(R2)评价细胞内的铁摄取量,比较各组R2的差异.结果①普鲁士蓝染色结果:SPIO及USPIO单独标记组标记率为60％～70％,低于联合标记组的100％;②MRI结果:未标记细胞R2为(2.10±0.11)/s,SPIO、USPIO单独标记组细胞R2分别为(3.39±0.21)/s、(3.16±0.32)/s,SPIO-脂质体联合标记组及USPIO-脂质体联合标记组R2分别为(4.03±025)/s、(3.61±0.32)/s,SPIO-PLL联合标记组及USPIO-PLL联合标记组R2分别为(5.38±0.52)/s、(4.44±0.35)/s,SPIO、USPIO与PLL联合标记组R2大于SPIO、USPIO与脂质体联合标记组(P＜0.05);而与脂质体联合标记组R2大于单独标记组(P＜0.05);SPIO与USPIO单独标记细胞时R2差异无统计学意义(P＞0.05),SPIO与脂质体或PLL联合标记时R2高于USPIO(P＜0.05).结论SPIO、USPIO单独标记及与PLL、脂质体联合标记均可以成功标记神经干细胞,提高R2,其中SPIO与PLL联合标记效率最高.', 'label': 0, 'keyword': array(['粒子', '铁化合物', '联合', '0.05'], dtype=object)}
    ```
    只提取 abst，但多个条目之间的 abst 有可能重复

    - drcd：
    ```json
    {'id': '1001-10-1', 'context': '2010年引進的廣州快速公交運輸系統，屬世界第二大快速公交系統，日常載客量可達100萬人次，高峰時期每小時單向客流高達26900人次，僅次於波哥大的快速交通系統，平均每10秒鐘就有一輛巴士，每輛巴士單向行駛350小時。包括橋樑在內的站台是世界最長的州快速公交運輸系統站台，長達260米。目前廣州市區的計程車和公共汽車主要使用液化石油氣作燃料，部分公共汽車更使用油電、氣電混合動力技術。2012年底開始投放液化天然氣燃料的公共汽車，2014年6月開始投放液化天然氣插電式混合動力公共汽車，以取代液化石油氣公共汽車。2007年1月16日，廣州市政府全面禁止在市區內駕駛摩托車。違反禁令的機動車將會予以沒收。廣州市交通局聲稱禁令的施行，使得交通擁擠問題和車禍大幅減少。廣州白雲國際機場位於白雲區與花都區交界，2004年8月5日正式投入運營，屬中國交通情況第二繁忙的機場。該機場取代了原先位於市中心的無法滿足日益增長航空需求的舊機場。目前機場有三條飛機跑道，成為國內第三個擁有三跑道的民航機場。比鄰近的香港國際機場第三跑道預計的2023年落成早8年。', 'question': '廣州的快速公交運輸系統每多久就會有一輛巴士？', 'answers': {'text': array(['10秒鐘'], dtype=object), 'answer_start': array([84], dtype=int32)}}
    ```
    只提取context，但依旧会出现多个条目之间context一样的情况，根据 id 中的 xxxx-yy-z 中的 xxxx-yy 做去重

    - iflytek：提取 `sentence` 键即可
    - ocnli： 忽略这个分类
    - tnews：提取 `sentence` 键即可


- **Chinese Fineweb Edu Dataset V2.2**

    服务器路径：`/share/project/wuhaiming/data/dataset/fineweb-edu-chinese-v2.2/4_5/`

    子文件路径（一共9767个文件）：
    ```bash
    000000.parquet
    ...
    009766.parquet
    ```

    原始数据结构：
    ```python
    import pandas as pd

    data = pd.read_parquet('/share/project/wuhaiming/data/dataset/fineweb-edu-chinese-v2.2/4_5/000000.parquet')
    print(data.head())

    ## 列名：text   score   source                                                                            
    0   一、公司清算所有者权益怎么分配\n公司清偿所有债务后，有剩余财产的，所有权权益由股东按出资的...   0.801270    CCI3
    1   分散采购的优缺点（ 集中采购与分散采购该如何进行选择）\n你是否经常为采购流程的效率和效果而...    0.817383    CCI3
    2   绪论马克思主义是关于无产阶级与人类解放的科学\n【教学目的与要求】通过本章的学***要使学生...  0.850098    CCI3
    ```

- **FineWeb-Edu**

    服务器路径：`/share/project/wuhaiming/data/dataset/fineweb-edu/sample/10BT`

    子文件路径（一共14个文件）：
    ```bash
    000_00000.parquet
    ...
    013_00000.parquet
    ```

    原始数据结构：
    ```python
    import pandas as pd

    data = pd.read_parquet('/share/project/wuhaiming/data/dataset/fineweb-edu/sample/10BT/000_00000.parquet')
    print(data.head())

    ## 列名：text   id  dump    url file_path   language    language_score  token_count score   int_score
    0  The Independent Jane\nFor all the love, romanc...  <urn:uuid:0d8a309d-25c5-405d-a08a-c11239f0d717>  CC-MAIN-2013-20      http://austenauthors.net/the-independent-jane  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/se...       en        0.974320          845  2.750000          3
    1  Taking Play Seriously\nBy ROBIN MARANTZ HENIG\...  <urn:uuid:316c7af5-14e1-4d0b-9576-753e17ef2cc5>  CC-MAIN-2013-20  http://query.nytimes.com/gst/fullpage.html?res...  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/se...       en        0.961459         1055  2.562500          3
    ```

### Phase 1 数据核验与统一约束

各类别配额均以**清洗、去重后使用新 tokenizer 统计的 token 数**为准：中文通用短文本 0.8B、中文高质量文本 0.4B、中英混排 0.3B、非中文文本 0.3B、代码/数学/格式/汉字覆盖补充 0.2B。总量目标为 2B，类别和总量允许误差均为 ±1%。不得通过复制样本补足配额。

构建前先生成 `reports/source_inventory.json`，逐源记录文档数、原始字符数、新 tokenizer token 数、许可证分布、无效样本数和去重后预计可用量。若任一来源去重后不足，应从同类别的已批准来源补充，而不是降低质量或许可要求。

> **CLUE 数据说明**：Phase 1 暂时使用当前目录中的 CLUE benchmark 训练集，不使用 CLUECorpus2020。该 benchmark 规模明显不足以单独提供 0.8B token，因此清洗、去重后全部计入中文通用短文本，剩余配额由 OpenCSG 中文语料中的非重复样本补足；不得复制或重复采样 benchmark。真正的 CLUECorpus2020 留到 Phase 2 再重新规划。只抽取 train split 的文本字段，排除 dev/test、标签和索引，避免评测污染。

已有三个来源按以下规则处理：

- **CLUE benchmark 训练集**：启用当前 `afqmc`、`c3`、`chid`、`cluewsc2020`、`cmnli`、`cmrc2018`、`csl`、`drcd`、`iflytek`、`ocnli`、`tnews` 的 train split；按任务结构抽取完整文本，过滤标签、索引、空值和重复样本。全部保留数据按实际 token 数计入中文通用类别。
- **OpenCSG Fineweb-Edu-Chinese-V2.2**：使用 `text` 字段并保留原始 `source`；过滤乱码、模板页、低信息密度和 PII。先为中文高质量类别采样 0.4B token，再以互不重复的样本补足中文通用类别相对 0.8B token 的缺口。许可证按数据集卡记录为 Apache-2.0。
- **FineWeb-Edu sample-10BT**：仅保留英语文本和有效正文，建议 `language_score >= 0.90`；完成全局去重后采样 0.3B token。许可证按数据集卡记录为 ODC-By 1.0。

### 中英混排数据构建（0.3B token）

这部分以技术语境中的自然中英混排为目标，不收集仅含英文网址、版权页或菜单的伪混排文本。目标来源分布如下；某一来源不足时只能由其余合规来源补齐：

| 来源 | 目标占比 | 目标 token | 主要内容 |
| --- | ---: | ---: | --- |
| GitHub 显式许可仓库 | 45% | 135M | 中文 README、文档、教程、API 说明 |
| Hugging Face 显式许可仓库 | 15% | 45M | 模型卡、数据集卡、技术说明 |
| Wikimedia 与 OpenAlex | 20% | 60M | 技术百科、双语术语、论文元数据 |
| 显式许可技术站点白名单 | 20% | 60M | 官方文档、开放教程、技术博客 |

#### 采集流程

1. **建立许可白名单**：GitHub 仅接收仓库 License API 可识别且允许再分发的许可证，并记录仓库、文件路径、commit SHA 和许可证；Hugging Face 仅接收卡片元数据中声明了可用许可证的仓库；Wikimedia 使用官方 dump；OpenAlex 仅使用其 CC0 数据字段。许可证缺失、`NOASSERTION`、文件头与仓库许可证冲突的数据进入隔离区，不进入最终集。
2. **优先离线或版本化获取**：GitHub/Hugging Face 使用 API、仓库归档或固定 revision，Wikimedia 使用 current dump。允许从 Common Crawl 的 CDX 索引和 WARC range request 获取白名单站点的历史页面，但 Common Crawl 本身不授予页面内容许可；无法在站点级确认许可的页面直接丢弃。只有缺少离线渠道时才直接访问网页，并遵守 robots.txt、站点条款、速率限制和带联系方式的 User-Agent。
3. **提取正文**：优先读取 Markdown、RST、纯文本等源文件；HTML 使用正文抽取器去除导航、广告、评论和重复页脚，同时保留标题、列表、代码块、表格文字和行内 API 标识。按章节切分为 128-4096 个新 tokenizer token，不能从代码围栏或表格内部截断。
4. **判定真实混排**：样本至少包含 20 个汉字和 5 个拉丁词；在“汉字数 + 拉丁词数”中，汉字占比须位于 15%-85%。至少一个段落同时出现中英文，且英文内容应包含可解释的术语、标识符或完整短语。仅由 URL、文件名、导航栏造成的英文命中不计入。最后用中英语言识别器复核并过滤乱码。
5. **质量与安全过滤**：删除模板页、机器翻译痕迹严重文本、SEO 拼接、密钥/令牌、个人邮箱和电话等敏感信息；排除 vendor、生成目录、变更日志和压缩文件。单仓库最多占该类别 0.5%，单站点最多占 5%，避免来源垄断。
6. **去重与抽样**：执行本文统一的文档内、精确和 MinHash 近重复去重，再按来源、领域和长度分层采样到目标 token 数。输出每个来源的发现量、许可淘汰量、质量淘汰量、重复率和最终 token 数。

建议 GitHub 文件范围为 `README*`、`docs/**/*.{md,mdx,rst,txt}` 和明确的教程目录；Hugging Face 固定 revision 下载 `README.md`。OpenAlex 只使用其提供的标题、摘要倒排索引还原文本、概念等元数据，不抓取其链接论文的受版权保护全文。

### 代码/数学/格式/汉字覆盖补充（0.2B token）

该类别由许可明确的仓库、官方 dump 和确定性生成数据共同构建，不将普通网页抓取作为主来源。

| 子类 | 占比 | 目标 token | 主要来源 |
| --- | ---: | ---: | --- |
| 代码 | 52% | 104M | GitHub 显式许可仓库 |
| Markdown/结构化格式 | 25% | 50M | GitHub 文档与配置文件、Stack Exchange dump |
| 数学 | 18% | 36M | Mathematics/TeX/Cross Validated dump、Wikimedia 数学条目 |
| 汉字覆盖补充 | 4% | 8M | 既有汉字资源、中文 Wiktionary/Wikipedia dump |
| special token/chat 格式 | 1% | 2M | 按 Qwen chat template 确定性生成 |

#### 代码与格式数据

1. 通过 GitHub API 筛选许可证明确、非 fork 的仓库，固定 commit 获取文件。覆盖 Python、JavaScript/TypeScript、Java、C/C++、Go、Rust、Shell、SQL 等常见语言，并限制单仓库和单语言占比。
2. 排除二进制、压缩/压缩后代码、vendor、生成文件、lock 文件、超大文件、自动复制的模板及疑似密钥。存在文件级许可证且与仓库许可证冲突时，以更严格结果处理或隔离。
3. 对支持的语言使用 Tree-sitter 或对应解析器验证语法；其余语言至少执行扩展名、可打印字符率、最长行和重复行检查。代码近重复采用词法 token 5-gram、256 个 MinHash permutation 和 Jaccard 0.85 阈值，与 The Stack 的代码去重思路一致。
4. Markdown/RST 保留标题、列表、代码围栏、表格和链接文本；JSON、YAML、TOML、XML 必须能够由对应解析器解析。结构化文件保持原始层级和缩进，不展平成普通句子。

#### 数学数据

1. 使用 Stack Exchange 官方数据 dump 中 Mathematics、Cross Validated、TeX 等站点的问题标题、正文以及 accepted/high-score answer；HTML 清洗时保留 Markdown、`<pre><code>` 和 MathJax/LaTeX。
2. 使用 Wikimedia current dump 中数学、物理、计算机等相关条目补充自然语言解释和公式上下文，保留原公式标记。
3. Stack Exchange 内容的 CC BY-SA 版本随发布时间变化，必须按官方规则记录 2.5、3.0 或 4.0，并保留 post id、站点、作者标识、URL 和时间等归因字段。不得只在数据集级写一个笼统许可证。
4. 过滤无正文、仅图片、公式无法闭合、答案只含外链以及低质量重复问答；按站点、主题和公式密度分层抽样。

#### 汉字覆盖与 special token

1. 以现有 `resources/hanzi/`、Unihan 和已生成的 `hanzi_set`/`char_features` 为覆盖目标，先统计 Phase 1 其他语料中每个目标汉字的出现次数。
2. 对缺失或低频汉字，优先从中文 Wiktionary/Wikipedia dump 中抽取带释义或上下文的合规句子；仍未覆盖的字符才使用本地字形、拼音、部首等已验证字段生成简短模板。模板数据必须有实际语义，不生成连续重复单字，并对合成样本设置上限。
3. 验收时要求目标汉字集合 100% 至少出现一次，同时报告每字频次和合成占比；繁简体不做 NFKC 或自动互转。
4. special token 数据由 Qwen tokenizer 的 `chat_template` 渲染合法的 system/user/assistant 多轮对话，覆盖中文、英文、代码、Markdown 和 JSON 内容。不能随机把 special token 字符串插入正文，也不能构造不闭合的会话结构。

### Phase 1 端到端流程与验收

1. 生成来源清单和许可清单，冻结 revision、dump 日期及下载 URL。
2. 抽取正文并标准化为统一 JSONL；保留可追溯元数据。
3. 执行编码、正文质量、PII/密钥、语种、格式可解析性过滤。
4. 执行统一去重，并用 Phase 0/下游评测集做精确与近重复污染检查。
5. 使用新 tokenizer 统计 token，按类别、来源、领域和长度分层采样。
6. 使用固定随机种子打乱并输出 shard；写入 manifest、哈希、统计报告和失败样本。

最终验收条件：总量及各类别 token 误差不超过 ±5%；未知/冲突许可证样本数为 0；全局精确重复数为 0；近重复率、来源集中度和合成数据占比均有报告；目标汉字覆盖率为 100%；结构化文件可解析；chat template 样本可由 tokenizer 正常编码和还原。

### 参考依据

- [FineWeb: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)：网页语料过滤、去重和质量控制流程。
- [DataTrove](https://github.com/huggingface/datatrove)：大规模精确、句子级及 MinHash 去重实现。
- [The Stack: 3 TB of permissively licensed source code](https://arxiv.org/abs/2211.15533)：代码许可筛选和近重复去重方法。
- [CLUECorpus2020](https://arxiv.org/abs/2003.01355) 与 [CLUE benchmark 数据集卡](https://huggingface.co/datasets/clue/clue/blob/main/README.md)：区分大规模中文预训练语料与现有 benchmark 子任务；Phase 1 暂用后者的训练集。
- [OpenCSG Fineweb-Edu-Chinese-V2.2](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.2) 与 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)：数据规模、字段和许可证。
- [GitHub repository licenses API](https://docs.github.com/en/rest/licenses/licenses)、[repository contents API](https://docs.github.com/en/rest/repos/contents) 和 [rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)：许可核验、版本化获取与抓取限速。
- [Wikimedia dump licensing](https://dumps.wikimedia.org/legal.html)、[OpenAlex About](https://help.openalex.org/hc/en-us/articles/24396686889751-About-us) 和 [Stack Overflow licensing](https://stackoverflow.com/help/licensing)：各来源的许可与归因要求。
- [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use)：Common Crawl 不替代原网页内容许可。

## Phase 2 - 全参继续预训练数据

### 整体数据配比

|数据类别	|比例	|说明|
|:---|:---|:---|
|中文高质量通用文本 |30% |CCI3.0-HQ、WanJuan1.0|
|中文知识密集数据	|20%	|OpenCSG Fineweb-edu-chinese、书籍、百科、论文、古文古诗(Wikisource )|
|中英混排高质量语料	|15%	|README、API 文档、中文技术博客、论文摘要|
|代码 / 数学 / 科学文档	|10%	|The Stack v2、代码文档、算法教程|
|英文 / 多语言保持数据	|15%	|FineWeb-Edu、The Pile|
|长文本专项数据 |10%    | 长篇教材章节、政策法规全文、长篇中文论文/报告、古文长篇章节 (文本长度>4k,<=32k)|

数据总量：20B tokens （新 tokenizer 计算）