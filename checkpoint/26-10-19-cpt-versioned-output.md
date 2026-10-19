# CPT 自动版本目录与阶段衔接

- 新训练恢复 ms-swift 原生 `add_version=true`；稳定根目录可以存在，日志由框架置于实际版本目录。Phase2 的 adaptation/joint 各自编号，不假设 A/B 的编号相同。
- 插件在 `on_train_end` 保留 run 内指针，并原子更新 `CC_LLM_OUTPUT_ROOT/next_stage_model.txt`。默认阶段衔接读取最近完成结果，不扫描最大版本号；显式 `MODEL` 优先。
- 断点恢复使用 checkpoint 的父目录并设置 `add_version=false`，不产生新版本；仍检查 checkpoint 属于指定阶段输出根目录。
- 不改变缓存、训练参数或模型结构。按用户要求，未运行测试或验证命令；仅同步了受影响的脚本测试断言。
