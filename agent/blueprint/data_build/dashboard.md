## Phase1：语义对齐

目标：适配单汉字 tokenizer 和新 embedding/lm_head，恢复原模型基础能力；数据应以高质量、短文本、分布稳定为主，不追求过多古文和长文本。

**Qwen3-1.7B: 1B tokens**

| 类别 | 比例 | 数据源 |
|---|---:|---|
| 中文通用 | 30% | CCI3.0-HQ 70% + WanJuan1.0 30% |
| 中文高质量知识 | 20% | FineWeb-Edu-Chinese V2.2 |
| 英文能力保持 | 15% | FineWeb-Edu-English (sample 10BT) |
| 中英混排/技术中文 | 10% | 从 CCI3.0-HQ 40% 、FineWeb-Edu-Chinese 40%、The Stack v3 20% 的 README/docs 中规则筛选 |
| 代码/数学/结构化 | 10% | The Stack v3 70% + OpenWebMath 30%；优先代码、Markdown、JSON/YAML/TOML、TeX |
| 新增汉字 Token/边界覆盖 | 15% | 从CCI3-HQ、FineWeb-Edu-Chinese、WanJuan中二次筛选新增单汉字 token 高覆盖文本；极低覆盖字符允许少量 FineWeb-zhtw/Wikisource 补充 |

Phase1 原则：短文本为主，主要控制在 512–2048 tokens；不需要专门引入大比例古文、繁体或长文档。

---

## Phase2：全参数继续预训练

目标：真正提升中文理解与生成，同时强化新增单汉字 token 和 PGCA 音形表示，并保持英文、数学、代码等通用能力。

**Qwen3-1.7B: 10B tokens**

| 类别 | 比例 | 数据源 |
|---|---:|---|
| 中文高质量通用 | 30% | CCI3.0-HQ 50% + FineWeb-Edu-Chinese V2.2 35% + WanJuan1.0 15% |
| 中文知识密集 | 25% | Chinese Cosmopedia 40% + 中文 Wikipedia 25% + FineWeb-zhtw 20% + Wikisource 10% + ect-krp ≤5% |
| 英文/多语/中英混排 | 20% | FineWeb-Edu-English 55% + FineWeb2 Multi 15% + 中英技术混排 30% |
| 数学/代码/科学 | 15% | The Stack v3 45% + OpenWebMath 30% + peS2o 25% |
| 新增汉字 Token 与音形覆盖增强 | 10% | 从全部中文候选池二次筛选，重点覆盖新增单汉字 Token、多音字、繁体、古文、异体字等 |

另外加两个横向约束，不单独占 bucket：

- 长文本：总 token 中至少 10%，主要来自 Cosmopedia、Wikipedia、Wikisource、科学论文。
- 古汉语/文言：建议总 token 中约 8%–12%，主要来自 Wikisource、ect-krp 和部分古籍类中文语料。

注意：古汉语不要额外再设一个和“中文知识密集”重叠的独立 15% bucket，否则类别重复。

---

## 下游任务微调阶段

这里不再强调统一 token 配比，而是每个任务使用对应监督数据集，所有模型使用相同训练集和微调配置。

| 下游能力 | 训练集 | 测试集 |
|---|---|---|
| 中文拼写纠错 CSC | twnlp_csc_data | LEMON、CSCD-NS |
| 中文语法纠错 CGEC | twnlp_cgc_data| FCGEC、NaSGEC |
| 古汉语理解 | Chinese Classical Corpus 为主，可加入 HistoryTrans、EvaHan | C³Bench + Fùxì |
| 中文基础表示 | OntoNotes/Weibo NER、MSRA/PKU CWS | 对应官方 test |
| 中文阅读/语义 | LCQMC、CMRC2018 等 | 对应官方 test |

其中古汉语训练数据：

- Chinese Classical Corpus：60%
- HistoryTrans：25%
- EvaHan 断句/标点：15%

## 附录

### 数据集网址

- CCI3-HQ:https://www.modelscope.cn/datasets/BAAI/CCI3-HQ
- Finweb-Edu-Chinese: https://www.modelscope.cn/datasets/opencsg/Fineweb-Edu-Chinese-V2.2
- Wanjuan: https://www.modelscope.cn/datasets/Shanghai_AI_Laboratory/WanJuan1_dot_0
- Finweb-Edu-English: https://www.modelscope.cn/datasets/HuggingFaceFW/fineweb-edu
- The-Stack-v3:https://www.modelscope.cn/datasets/HuggingFaceCode/stack-v3-train
- Chinese-Cosmopedia: https://www.modelscope.cn/datasets/opencsg/chinese-cosmopedia
- Wikipedia: https://www.modelscope.cn/datasets/wikimedia/wikipedia
- Finweb-zhtw: https://www.modelscope.cn/datasets/voidful/fineweb-zhtw
- Wikisource: https://www.modelscope.cn/datasets/wikimedia/wikisource
- ect-krp: https://github.com/direct-phonology/ect-krp (下载地址：https://github.com/direct-phonology/ect-krp/releases/download/v1.1.0/ect-krp-v1.1.0-txt.zip)
- Fineweb2_Multilingual: https://www.modelscope.cn/datasets/HuggingFaceFW/fineweb-2
- OpenWebMath: https://www.modelscope.cn/datasets/jordangong/open-web-math
- peS2o: https://www.modelscope.cn/datasets/allenai/peS2o (只保留source == "s2orc")