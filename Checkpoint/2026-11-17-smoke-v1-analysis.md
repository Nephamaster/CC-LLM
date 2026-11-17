# Smoke v1 未通过：指令成品仍按旧 prepared 指针读取

分析基线：本地 main `4b5d05d`，输入为用户下载的 `outputs/smoke/v1`。仅阅读结果和源码、运行临时路径样例，未修改程序、数据或远端环境，未切换分支、提交或推送。

## 结论与证据

summary.json 共 78 项：26 passed、1 failed、51 blocked。唯一直接失败是 prepare/instruction_sft；日志末行打开服务器的 `data/instruction/prepared/v5/train.jsonl` 时 FileNotFoundError。General-SFT 输入未准备成功，Base/Char/PGCA 三条路线的指令训练、三项专项微调和后续全部通用/专项评估因此被阻断，并非各自独立报错。

失败日志：`outputs/smoke/v1/logs/prepare--instruction_sft.log:21`。三条路线每条 17 个后续步骤被阻断，合计 51 项。

本地当前 main 和 master 的 smoke/prepare.py 已直接读取 data/instruction/train.jsonl、dev.jsonl，不读 prepared/latest。此日志产生时执行的指令路径逻辑仍是旧逻辑；日志没有服务器 Git 提交信息，不能推定服务器当前提交。应同步最新 smoke/prepare.py 和 train/instruction/run.sh，并确认服务器两份成品文件在根目录。

## 已通过与尚未覆盖

- 环境初检、其他 13 项数据抽样、Char/PGCA 模型构建通过。
- Char Phase1/CPT、PGCA Phase1/Phase2-A/Phase2-B 共 5 次训练：trainer_state 均 global_step=2，所记录 loss/eval_loss/eval_clm_loss 均有限。验证 CLM loss 范围约 2.78–2.85，无本轮 NaN 证据。
- Base、初始 Char、Char-CPT、PGCA-ALIGN、PGCA-CPT 的五次 PPL 检查 complete=true，均覆盖六域。
- General-SFT、下游 LoRA、全部通用/专项推理评分均未执行，不能据此宣称通过。环境初检只检查 CUDA 和模块存在性，不导入 vLLM 扩展；本轮也不能证明 vLLM 实际推理已验收。
- 这些都是短序列/每阶段两步的冒烟，不证明模型效果、长期稳定性或全量数据内存表现。

## 当前源码检查与下一步

在 /tmp 临时构造成品 Train/Dev，并放置失效的 prepared/latest 指针，实际调用当前 smoke.prepare，正常抽样成功；验证现有修复路径有效，没有新增产品代码改动。

本次路径错误不要求重装环境或重做原始指令数据清洗。同步最新读取逻辑、确认成品文件后重新跑集中冒烟（会生成新 vN）。本轮通过的前序只能作为已运行部分的证据，不能把 51 项 blocked 当作通过。
