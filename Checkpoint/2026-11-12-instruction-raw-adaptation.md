# 指令原始数据归位与读取适配

基线：`master` / `cf9a71f`，开始时工作区干净。仅整理本地数据位置、适配 instruction 准备器及对应文档，未执行训练或正式清洗/去污染。

## 数据与格式

原文件以移动方式归位，未重写对话内容：

| 数据 | 当前位置 | 全量解析行数 |
| --- | --- | ---: |
| COIG-CQIA | `data/instruction/raw/COIG-CQIA/coig_cqia.jsonl` | 44,693 |
| Tulu 3 | `data/instruction/raw/tulu-3-sft-mixture/tulu3.jsonl` | 931,481 |

两份文件每行均为 `sample_id/source/messages/dedup_text/first_user_prompt/metadata`。COIG 的原始 ID 为 `metadata.source_row_id`，任务类型、人工校验及版权也在 metadata 中；Tulu 的原始 ID 为 `metadata.original_id`，含 source_subset，实际覆盖 18 个来源。不能继续只读顶层字段而丢失这些信息。

## 改动

- `prepare.convert` 直接复用 messages；补读嵌套原始 ID、任务类型、版权、许可及人工校验信息，顶层同名字段优先。
- 保留 sample_id 与完整 source_metadata（JSON 字符串），让不同来源的元数据保持可追溯且避免 Arrow 嵌套类型冲突；不重新生成或校验 ID 哈希。
- 保持原生 COIG instruction/input/output 支持、清洗规则、去重、去污染和混合配置。冗余 dedup_text/first_user_prompt 不进入训练内容。
- 更新根 README、instruction README 与 TODO 的数据状态。

## 核验与边界

- 全量流式解析 976,174 行，检查字段与角色形态，解析完成。
- 每个来源取前 30 条转换，共 570 条；567 条通过，3 条 Tulu 空内容按既有规则过滤。核对实际消息、原始 ID、任务信息及元数据保留。
- 临时回归检查覆盖原生 COIG、顶层字段优先、零值 ID、sample_id/文件行号回退、标量元数据输出及非法输入；通过。
- Python AST、Ruff F/E9、格式检查和 diff 空白检查通过。临时扫描/检查脚本及结果在 `/tmp/cc-instruction-format-20261112/`，未写入代码目录。
- 未进行全量清洗、Benchmark 去污染、Tokenizer Token 统计或 GPU 训练；格式读取通过不代表数据质量及正式混合规模已验收。

保留 Tulu 上限 60% 时，COIG 原始数量对应总池理论上限约 111,732 条，已低于首轮 150,000 目标；清洗与其他约束还会进一步减少。当前未授权调整混合目标，故保留配比并由准备报告记录缺口。
