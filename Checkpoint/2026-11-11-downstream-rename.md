# Downstream 阶段改名

基线：master，HEAD `e3278c9`，开始时 Git 工作区干净。用户确认将 finetune 阶段统一命名为 downstream；本轮未提交或推送。

## 改动

- `scripts/train/finetune/` 整体移动到 `scripts/train/downstream/`，保留 9 个文件；同步总训练、General-SFT、评估、冒烟的路径、模块导入及命令。
- 本地 `data/finetune/` 整体改名为 `data/downstream/`，仅做文件系统目录移动，不改写数据内容。数据受 Git 忽略，服务器/旧数据包需要单独同步改名；根 README 已补迁移命令。
- 公共函数改为 `run_sft`，目录变量改为 `DOWNSTREAM_DIR`，总入口使用 `SFT_MODEL_NAME`，不改变训练参数或模型输出名称。
- 新运行统一写入 `sft_report.json`，同步元数据读取和冒烟验收。
- 已有 checkpoint 的已知旧插件路径映射到当前 downstream 插件；Dev 路径支持将旧 data/finetune 相对位置定位到新目录，兼容读取旧报告名。保留原权重、args、历史报告与历史 outputs 目录，不增加旧命令入口或复制包。
- 根 README、pipeline.md、训练与评估说明同步修改。Checkpoint 历史记录不重写。`finetuned` 等通用术语保持原意。

## 验证

临时脚本和日志位于 `/tmp/cc-downstream-rename/`，环境复用 `/tmp/cc-todo-env/`（Python 3.11.16、ms-swift 4.6.0、Transformers 5.19.0、Torch 2.15.0+cpu；正式环境仍为 Python 3.12）。

- 旧/新插件路径、历史 ms_swift 路径、未知自定义插件保留、旧文件仍存在时正确迁移，均通过。
- 新/旧报告名读取、三项真实 Dev 路径映射、显式自定义 Dev、旧 args 回退读取，均通过。
- 冒烟读取新的 downstream 数据布局及 sft_report，通过。
- 新 downstream/csc.sh 入口真实执行受控小模型两步 LoRA：两次保存均完成 Dev 检查，写入新报告并正常发布检查点，冒烟读取通过；见 training.log。该结果验证调用链，不代表正式模型效果。
- 历史合并模型 args 仍保存旧插件路径；通过新模块入口实际加载并完成完整生成/Cache 验收，见 historical.log、historical-check/v0/report.json。
- 相关 Python AST、shell bash -n、git diff --check 通过。未执行真实 GPU、多卡、全量训练或完整 benchmark；模型源码与超参未修改。
