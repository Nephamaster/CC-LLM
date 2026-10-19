# CGEC SFT 数据整理

## 输入问题与已确认规则

`data/downstream/cgec/sft/raw/nacgec_all_type.json` 含全部 NaCGEC test 原句，不纳入训练。`train_FCGEC.txt` 也与 FCGEC valid/test、NaCGEC dev/test 重合；用户已确认按原句过滤这些评估数据，以及 MuCGEC 验证原句。

保留每个原句的第一个有效参考答案，不展开多参考增加样本权重。训练来源顺序 CGED → FCGEC；验证集仅来自 `valid_MuCGEC.txt`，不随机切分。仅清理字段两端空白、CGED 空 TAB 列及 ID 列，不做模糊匹配、标点/繁简归一化或额外采样。

## 产物

位置 `data/downstream/cgec/sft/final/`：

- `train.jsonl`：53,759 条（CGED 20,437；FCGEC 33,322）。
- `validation.jsonl`：1,137 条，由 MuCGEC 2,467 行按原句去重得到。
- `prepare_report.json`：处理数量与规则。

训练原始 57,803 行，去重 21 条、过滤重合原句 4,023 条。CGED 修复 402 个空 TAB 行、1 个夹带 ID 行。各 benchmark 重合计数可能互相重叠，不可直接求和。

格式与 CSC 相同：每行仅含 `messages`，一条 user、一条 assistant。初版 user 使用 CGEC 指令；按后续约定，现已改为 source-only，user 仅原句、assistant 仅纠正后文本。原条数、去重/防泄漏规则不变，保留原本无错的 source=target 样本。提示词迁移细节见 `26-10-19-correction-source-only.md`。

简单检查已通过：两份 JSONL 格式正确、各自原句唯一、训练与验证及四个评估 split 均无完全相同的原句。未做模糊近重复检查。原始文件未修改。

转换入口：`python -m scripts.data_factory.prepare_cgec`，已同步直接生成 source-only 数据。默认拒绝覆盖已有三个产物；`data/` 被 gitignore 忽略，实际训练文件需单独同步到服务器。
