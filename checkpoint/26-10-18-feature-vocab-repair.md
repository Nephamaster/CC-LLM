# 音形词表语义修复：feature schema v2

依据 `26-10-18-feature-vocab-audit.md` 和对应 JSON 核对源码；原始审计文件保留不改写。

## 根因与修复

- 原 `pinyin_candidates` 捕获所有异常并退到 kMandarin；未安装 pypinyin、库故障、合法缺失无法区分。现在强制 `pypinyin==0.55.0`，失败显式报错。合并 pypinyin / Unihan.kHanyuPinyin / Unihan.kMandarin，逐读音记录来源及来源文件哈希。
- pypinyin 缺读音时不再回传原汉字。未知候选仍有 `<unk>` 占位，但声母、韵母也必须 unknown，原始 tone=null，pinyin_mask=false。JSONL label_masks/PT 属性 mask 区分各类监督有效性；`auxiliary_feature_targets` 生成候选对齐的 -100 忽略标签，不增加辅助头、不做静态读音选择。
- Unicode NFD 处理元音/辅音附加调号，`ň/ḿ/ǹg` 正规化为 n3/m2/ng4；严格 helper 恢复 j/q/x 的 v、零声母 y/w 和 iou/uei/uen 等。音节性 m/n/ng/hm/hng 显式约定为零声母＋完整音节韵母。
- 当前真实 28,134 字集合仅用固定版本 pypinyin 逐字检查：19,780 字一个候选、5,334 字两个、1,498 字三个，最大 11，未发生规范化/拆分异常。7 字超过原 8 槽（呵、敦、欸、湛、繆、誒、那）。新构建默认 16 槽，超限失败而非截断；索引和 PGCA 读取 Manifest 槽数。
- 上述检查有 1,002 字在 pypinyin 单一来源中未知；不是最终合并 Unihan 后的缺失率，也不能与审计的 993 直接当作同源比较。正式重建仍需原始 Unihan/结构资源。
- 验收从“每个汉字至少一个真 mask”改为原始标签→ID→JSONL→PT 的语义一致性、常见异读、合法未知和哈希绑定。旧 schema 必须重建，不按 ID 范围正常就放行。未知结构不构造有效监督，合法字形保持可用。
- embedding_config 新增 feature schema、vocab Manifest 和 PT 哈希；新 PGCA 配置记录同样标识，并绑定实际 persistent lookup tensors 的独立指纹，remote-code 加载时重新计算检查。PGCA 构建先核对所有输入，拒绝陈旧配置、旧 PGCA checkpoint 充当新特征基座及已有输出目录，避免同尺寸但不同 ID 语义的混用。
- Vocab/Embedding/PGCA 命令行验收失败返回非零状态，避免 report.passed=false 但 shell 继续后续构建。指定新 Char 路径时，vocab facade 的 features 默认路径和验收报告位置跟随该目录，不写回旧实验。
- 修正当前 `modeling_qwen3_pgca.py` 中 `pgca_enabled` 分支的 return 缩进错误。保留用户已有开关及 ungated PGCA 设计，不添加门控或改变训练阶段逻辑。

## 兼容性与验证范围

- 不改语义 tokenizer、语义词表 ID 或已有模型目录；原 `feature_index.pt` SHA256 仍为 `7bdceee07302c4a4d2895b8dab521e7fa694e8b4ebdf198f016e6217268eddc8`。
- 旧 PGCA checkpoint 保留原映射/行为；修正不会自动进入它们打包的 remote code。不能把新表直接替换到旧 PGCA 权重中。新实验需要从 Char 权重重建音形 embedding、persistent index 和 PGCA。
- 验证覆盖来源合并、版本/执行失败、未知与轻声区分、Unicode 特殊音节、禁止截断、文件篡改、JSON/JSONL/PT 一致、字形保留和辅助忽略标签；包含真实 Transformers tiny Char→v2 PGCA 构建、AutoClass remote-code 重载及前向/反向，也保留旧 ungated 回归。
- 初始本地无 `resources/`，修复过程中用户同步了汉字集合、结构表及 Unihan 原文。随后已使用真实资源重建完整 28,134 字音形词表及 154,019 行 JSONL/PT，输出到独立 `data/feature_vocab_repair_v2/`，不覆盖任何原模型。尚未构建/训练正式 1.7B PGCA。

## 全量产物结果

现存报告 `models/feature_vocab/features/feature_validation_report.json`：passed=true、154,019 行、28,134 汉字、原标签/ID/JSONL/PT 无不一致。`models/feature_vocab/vocab_validation_report.json` 的七项检查全部通过，包含真实 AutoTokenizer、全汉字单 token、映射元数据及 wrapper 对齐。

| 有效拼音候选数 | 原审计 | v2 全量重建 |
| --- | ---: | ---: |
| 0 | 993 | 992 |
| 1 | 27,043 | 19,786 |
| 2 | 98 | 5,335 |
| ≥3 | 0 | 2,021 |

多音字合计由 98 增至 7,356；最大 11。992 个未知拼音均正确屏蔽，不再作为零声母/轻声；结构未知仍为 27，未伪造缺失字形。音形词表大小为 pinyin=1517、shengmu=25、yunmu=45、tone=8、stroke_count=44、radical_stroke=3100、structure=16。

新 PT SHA256：`8ba2b71c55c1b6c67f30042fdd654b1f4688ede618698895dc6f6eed0885d5fb`。独立 bundle 同时保存语义 vocab 副本和匹配的 embedding_config（16 拼音槽＋1 字形槽）；它不是包含模型权重的可训练 checkpoint。本地 Char 目录没有 safetensors/pytorch_model 权重，正式 1.7B 重建需在服务器执行。

回归验证：71 项测试及 6 个子测试通过，包含古汉语 SFT/Benchmark reader 回归；有一项上游 PyTorch 弃用提示，不影响通过结果。

## 2026-10-19：构建包迁移

用户将原 `data/feature_vocab_repair_v2/` 移至 `models/feature_vocab/`。这是模型构建产物归类调整，不是数据版本或特征 ID 变更。

- 代码不存在旧目录硬编码。加载以传入根目录和文件相对路径为准；`EmbeddingFeatureConfig.from_artifacts("models/feature_vocab")` 可校验新位置的文件及绑定关系。
- `feature_index_manifest.json` 的 features_dir/output_jsonl/output_pt 和原 vocab_validation_report 的 features_dir 记录的是原构建/验收位置，保留原值。它们不是加载定位依据，不能将修改历史路径当作重新完成语义验收。
- 移动后重新调用实际配置加载代码，核验所有 Manifest 引用的文件 SHA256、语义 vocab 与特征索引绑定，并与现存 embedding_config 比较；目录移动不要求重建词表、索引或模型。原全量语义验收记录仍保留。
- 此包不含模型权重。构建 PGCA 时仍需将其 features 和匹配配置部署到新 Char 模型目录，并确认语义 vocab 相同；不能把 `models/feature_vocab` 当作 PGCA 的 `--model-path` 或训练的 `--dataset`。
- 包位于 Git 忽略的 models 下，应单独同步/归档；发布正式模型前保留，模型独立加载验收后可移到长期归档，但应保留可解释数字 ID 的词表与来源信息。

## 服务器重建步骤（保留旧实验）

确保服务器有原始 `resources/`，在项目根目录执行。目标 Char/PGCA 路径必须尚不存在；复制 Char 的目的是保留语义权重与 tokenizer，仅重建音形产物。

```bash
set -euo pipefail
pip install 'pypinyin==0.55.0'

FEATURE_CHAR=models/Qwen3-1.7B-Base-Char-FeatureV2
FEATURE_PGCA=models/Qwen3-1.7B-Base-Char-PGCA-FeatureV2
[[ ! -e "$FEATURE_CHAR" && ! -e "$FEATURE_PGCA" ]] || {
  echo '请使用尚不存在的新实验目录' >&2
  exit 1
}
cp -a models/Qwen3-1.7B-Base-Char "$FEATURE_CHAR"

python -B -m src.vocab.build_vocab feature-vocab \
  --char-model-path "$FEATURE_CHAR" --max-pinyin-per-char 16
python -B -m src.vocab.build_vocab feature-index \
  --char-model-path "$FEATURE_CHAR"
python -B -m src.vocab.build_vocab validate \
  --char-model-path "$FEATURE_CHAR"
python -B -m src.embedding.build_embedding config \
  --char-model-path "$FEATURE_CHAR"
python -B -m src.embedding.build_embedding validate \
  --char-model-path "$FEATURE_CHAR" --skip-model-weights
python -B -m src.pgca.build_pgca \
  --model-path "$FEATURE_CHAR" --output-path "$FEATURE_PGCA" \
  --pgca-layers 11,12,13,14,15,16
python -B -m src.pgca.validate_pgca --model-path "$FEATURE_PGCA"
```

不要为本次修复重跑 semantic-vocab 或 migrate；tokenizer 未变时，既有文本与 tokenization 缓存可复用。但新 PGCA 音形参数需要重新训练，不能直接续用旧 PGCA 优化器状态。若使用做过语义对齐的 Char 权重，需另行确认其语义 ID 和词表完全一致；此步骤不实现旧 PGCA 参数的按标签迁移。

官方语义依据：[pypinyin strict 约定](https://pypinyin.readthedocs.io/zh-cn/master/usage.html)、[Unihan 属性定义](https://www.unicode.org/reports/tr38/)。
