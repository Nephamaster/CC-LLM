# Hierarchical-12L 改为默认构建路径

基线：master `724d125`，开始时工作区干净。用户要求新方案成为正常执行路径，Shared 为显式对照。本轮无提交/合并/推送。

- 正常模型构建 shell 默认 CONTEXT_MODE=hierarchical，沿用 `<Base>-PGCA` 的模型路径、完成指针和主线名称。已有正常训练与评估命令不用改架构后缀。
- Python build_pgca_model 及 CLI 默认启用 context 并选择 hierarchical，省略模式参数同样构建三级编码器。CLI 仍接受显式 shared；高级静态 PGCA 可显式 shared + --no-context-encoder，不自动降级。
- 旧 Shared 对照的文档命令显式设置 CONTEXT_MODE=shared 和独立 PGCA_MODEL。对照训练仍需设置相应独立 MODEL_NAME/PHASE1_MODEL，避免混用完成指针。
- 不修改 Config 缺字段时的 shared 解码规则：它用于兼容旧 checkpoint，不是新构建入口默认值。正常载入旧权重不会自动增添 C2/C3；显式升级入口不变。
- 冒烟构建复用上述正常构建 shell，默认自动检查新 Hierarchical；显式 shared 仍可检查旧结构。
- 根 README、冒烟说明、架构文档与工程报告同步更新。辅助任务开关等其他默认行为未改。

验证证据在 `/tmp/cc-hierarchical-default-20261117/`：真实 CPU Char → 默认 Hierarchical / 显式 Shared 两种构建、AutoModel 加载、结构与 Cache 校验通过；CLI 默认和显式参数检查通过；原有 Shared/训练/辅助开关快速回归合计 67 passed、1 skipped、6 个 unittest 子测试通过。跳过的慢用例为前轮已执行的实际 Swift CPU 短跑，本轮没有重复运行。

隔离执行整个模型构建 shell（重计算 Python 使用替身），确认默认模式 hierarchical、默认输出 Qwen3-Base-PGCA，Shared 覆盖模式与指定路径、两个完成指针正常。Shell、Ruff F/E9、AST 和 diff 空白检查通过。未运行真实规模 GPU 模型构建或训练，未改环境、数据和已有模型目录。
