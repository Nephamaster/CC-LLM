# 指令成品目录与 main/master 职责拆分

基线：master `1fa5544`、main `ca0eed0`，开始时工作区干净。main 是 master 的祖先，落后 22 个提交。用户要求 master 使用成品数据跑正常实验；main 同步 master，同时保留数据处理、验证、消融和探索内容。不操作被 ignore 的本地数据、环境或产物，不推送。

## 已准备的改动

- 正常训练和集中冒烟默认读取 `data/instruction/train.jsonl`、`dev.jsonl`；保留 INSTRUCTION_DATA、TRAIN、DEV 的显式覆盖，不读取 prepared/latest 或 final 回退目录。
- master 删除原始数据清洗 prepare.py、decontamination.py 和其专用 pyahocorasick 训练依赖；保留模型相关 budget.py、finish.py、run.sh 及训练模板/EOS/生成验收。
- 活动 README/pipeline/smoke 文档改成成品数据包直接运行。
- main 预览包含迁移后的 `scripts/data_factory/instruction/{prepare.py,decontamination.py,README.md}`；子进程模块路径同步改为新位置。main 的准备产物仍可保留 prepared/vN 供研究选择，发布数据包时使用选中 Train/Dev 两文件放到 data/instruction 根目录。
- main 预览保留原 main 已跟踪、但被 master 精简掉的研究资料共 162 个文件，涉及 agent、历史 checkpoint、data_factory、diagnostics、download、validation、ancient diagnostics、tests、tools、原研究 requirements 及 src 备份；正常实验代码采用 master 新实现，不恢复被替换的旧训练入口或旧评估依赖锁。

## 已做验证

临时预览、迁移副本与一次性检查均在 `/tmp/cc-branch-split-20261117/`。

- 冒烟用成品根目录正常抽样，即使存在失效的 prepared/latest 指针也不读取它。
- 隔离执行真实 instruction/run.sh（训练/Python 执行使用替身），验证默认根目录、INSTRUCTION_DATA 覆盖、TRAIN/DEV 覆盖均取到预期文件。
- main 预览中的数据准备模块可正常导入和显示 CLI；四条 JSONL 小样例完成准备/报告/划分，并核对新的去污染子进程模块路径。基准索引和子进程执行用替身，不代表真实全量去污染验收。
- main 研究文件预览与原 main 的相应内容逐一相同；未运行原有全部历史测试，也未整体重构其旧目录引用。
- Shell、Ruff F/E9、diff 空白检查通过；未执行 GPU 训练或全量数据准备。

## Git 落地安排

按用户提供 AGENTS.md 第 1 节，git add/git commit 需要明确授权。用户已明确允许本地提交和合并；不推送、不包含根目录新增的数据压缩包。

master 提交成品化改动；main 合并 master，并保留上述原研究文件、加入迁移后的指令处理模块。main 分支中同名记录维护其合并与迁移结果。最终工作分支回到 master。

## main 的落地结果

master 成品化提交为 `518a578`。本合并将其完整纳入 main，并保留原 main 的 162 个研究文件；新增指令处理入口位于 scripts/data_factory/instruction/。正式训练、冒烟继续使用 master 同一套成品路径。

主流程文件除分支说明外与 master 一致。历史研究脚本/测试按原 main 内容保留，未作全量兼容重构。后续正常实验更新由 master 同步到 main；main 的研究专属代码不应整枝反向合并到 master。

执行了本地提交/合并，不推送，不操作数据压缩包或被 ignore 的本地文件。最终合并提交可通过 main 的 Git 日志识别。
