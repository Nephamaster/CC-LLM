# CPT 独立阶段脚本

- 用户明确要求一个阶段一个脚本、不要 common。当前入口为 `phase1.sh`、`phase2_a.sh`、`phase2_b.sh`，各自顶部包含完整参数，直接准备数据并调用 Swift；不要为了消除重复再引入隐藏的公共 shell/runner。
- 移除 `cpt/common.sh` 与合并的 `phase2.sh`；未提交的用户参数已迁入三个脚本：每卡训练 batch=1、梯度累积=32、预处理进程=4。
- A 显式 MODEL 优先，否则读取 ALIGN_OUT 指针；B 显式 MODEL 优先，否则读取 ADAPTATION_OUT 指针。统一使用 RESUME_FROM_CHECKPOINT，仅恢复本阶段，不自动运行下一阶段。
- CPT_ROOT 为 A/B 聚合根目录；OUT 现在明确表示当前阶段根目录。保留 adaptation/joint、原生版本目录、完成指针和 B 向 CPT_ROOT 发布最终结果的布局；既有 checkpoint 无需迁移。
- 预算、冻结规则、学习率、模型插件及数据准备逻辑不改。A/B 必须共享数据顺序、tokenizer、长度和预算；不同 batch/卡数需要匹配的数据视图，不能复用不匹配的 padding 几何。
- 更新冒烟入口和 README；22 项快速脚本检查通过，覆盖自动阶段衔接、显式模型覆盖、各阶段恢复、参数覆盖和缓存复用；shell 语法检查通过，未启动真实训练。
