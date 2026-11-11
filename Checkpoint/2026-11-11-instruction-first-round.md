# 首轮 General-SFT 规模

基线：master、HEAD `e3278c9`，在上一轮 downstream 改名的未提交工作上增量修改，未提交或推送。

用户确定约 15 万条高质量样本、单 Epoch 约 0.12–0.23B Token、训练 1–2 Epoch，并明确 Token 指完整输入（Prompt + 回答），Assistant 有效监督量另计。

## 落实

- 数据准备默认 `--samples=150000`，保留该参数原有的“总样本池含 Dev”口径。默认 2% 验证，即约 14.7 万训练、3000 验证。保留现有清洗、去污染及混合约束，不重复补样；报告新增目标上限、实际 Train/Dev 和不足量。
- 默认 1 Epoch，可设置 `NUM_TRAIN_EPOCHS=2`。移除正数 SUPERVISED_TOKENS 会静默强制 Epoch=1 的行为；此可选参数按每 Epoch 的 Assistant 有效监督量选择完整对话，默认 0 不裁样本视图。
- 每个输入视图分别统计 input_tokens（含模板，不含 padding）及 supervised_tokens（shift 后有效 labels），均为单 Epoch 量。完整输入规划范围不写入监督预算，不作为硬门槛。
- 选模报告列出单 Epoch 输入、配置 Epoch、计划总输入，以及每 Epoch/总监督目标和实际训练、实际选中模型的监督计数；不把准备量冒充实际训练量。
- 指令训练说明、根 README、pipeline.md、TODO 同步该规划。2 Epoch 输入量预估 0.24–0.46B，均为计划值，真实值待数据就绪后统计。

## 验证

临时目录 `/tmp/cc-instruction-first-round/`；复用 `/tmp/cc-todo-env/`（Python 3.11.16、ms-swift 4.6.0、Transformers 5.19.0、CPU Torch 2.15.0）。

- 默认采样量 150000；小池不足时如实报告，40 条目标/30 条可用对应短缺 10、27 训练/3 验证。
- 实际 Swift 模板与小 Tokenizer 的五条多轮样例：输入 60 Token、有效监督 12 Token，口径分离通过。
- 受控选模报告的 2 Epoch 计划计数及总监督目标计算通过；属于报告逻辑测试，不是训练效果验收。
- 真实 shell 入口配合替身运行器验证：正数监督目标不会覆盖 NUM_TRAIN_EPOCHS=2。
- Python AST、Ruff F/E9、shell bash -n、git diff --check 通过。

未下载/构建正式 15 万样本，未实测 0.12–0.23B 输入规模，未启动正式训练或 GPU 两 Epoch 验收。清洗后可用量、长度过滤及现有来源混合限制仍可能使实际样本不足，按报告决定后续配置，不伪称目标已达成。
