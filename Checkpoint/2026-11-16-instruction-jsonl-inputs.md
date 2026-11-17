# 指令数据准备入口直接读取两个 JSONL

基线：`master / 28f56e1`，开始时工作区干净。用户服务器的文件直接位于 raw 下，本轮仅调整 prepare 输入逻辑与对应文档，不移动或重写原始数据。

- 默认 COIG 输入改为 `data/instruction/raw/COIG-CQIA.jsonl`。
- 默认 Tulu 输入改为 `data/instruction/raw/tulu-3-sft-mixture.jsonl`。
- 删除递归目录扫描与 JSON/Parquet 格式分支，直接复用 iter_jsonl 逐行读取这两个文件。自定义 --coig/--tulu 仍接受 JSONL 文件路径。
- 缺失文件或传入目录时在基准输入收集、输出目录创建前报告具体文件路径。清洗、嵌套 metadata 处理、去重、混合和划分规则未改。
- 根 README、instruction README 的目录结构与输入说明同步更新。

验证：`/tmp/cc-instruction-jsonl-20261116/check.py` 检查默认参数、直接读取、BOM/空行，以及四条真实 JSONL 样例经过转换/SQLite 去重混合/Train-Dev 划分/报告和 latest 指针的链路，均通过；目录和缺失文件负例在创建输出前失败。基准索引用替身隔离，未验证真实去污染、原始全量数据或 GPU 训练。Ruff F/E9、格式和 diff 空白检查通过，临时脚本未放入仓库。

现在默认命令即可运行：`.venv/train/bin/python -B -m scripts.train.instruction.prepare`。服务器需已具备上述两个文件及原有基准数据前置条件。
