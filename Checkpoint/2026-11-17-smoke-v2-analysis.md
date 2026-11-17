# Smoke v2：General-SFT 训练结束后候选验收未通过

分析输入：outputs/smoke/v2/summary.json 和 logs 中实际提供的 Base、PGCA 两份 instruction 日志；Char 的失败日志和候选验收 report.json 未同步。本轮只分析，不改程序或运行环境。

- 汇总为 27 passed、3 failed、48 blocked。v1 指令数据路径错误已不再是本轮阻塞点；三条路线失败阶段均为 train/instruction。
- Base、PGCA 已完成两步 LoRA 训练、保存 checkpoint-2，训练进程 exit_code=0，验证 loss 有限（保存时分别约 2.577、3.004）。不能把整个阶段失败描述成训练进程崩溃或 loss NaN。
- 两份日志随后均显示 `[dev-check] passed=False full=True`，finish.py 报没有通过 finite-loss + generation acceptance 的已保存候选。失败发生在候选重新加载验收、合并发布之前；未生成可供下游接续的完整 INS 模型。
- 当前源码的冒烟设置 DEV_CHECK_MODE=report；该模式允许质量门槛未达标，但 completed/integrity 检查失败仍退出 1。故不能仅凭 passed=False 归因于“两步训练太少、生成质量差”。需要检查候选报告的 mode、error、checks、integrity_passed、quality_passed。
- 日志没有打印具体检查项；尚不能确定是 EOS/模板、Adapter 生效、Cache 数值/生成一致性，还是服务器实际模式不同。需要同步各模型 `v0/candidate_checks/checkpoint-2/v0/report.json`，并补 Char 日志，才能确定底层原因。

Base 路径例：outputs/smoke/v2/training/Qwen3-1.7B-Base-INS/v0/candidate_checks/checkpoint-2/v0/report.json。PGCA 同级目录为 Qwen3-1.7B-Base-PGCA-CPT-INS。未据现有证据建议重训或放宽正式验收门槛。
