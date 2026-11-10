# Base / Char / PGCA 集中冒烟

基线：`master / 17b3355` 加本轮工作区改动；用户未跟踪的 `experiment.xlsx` 保持原样。用户确认使用完整 Base 输入，实际完成模型构建、训练与全部评估；服务器 GPU 0/1/2/3 各 40GB，一张卡同时只放一个模型。无裁判接口的 Fuxi 项单独标记。

## 实施

- 入口 `scripts/smoke/run.sh /path/to/Base`；`prepare.py` 生成少量真实输入，`run.py` 串联现有构建/训练/评估脚本。
- 构建在 CPU 完成。默认三条模型路线各独占一张卡并行，路线内部串行；卡不足时通过 GPU 队列排队，支持单卡串行。
- 覆盖实际 Char/PGCA 构建、Char Phase1+CPT、PGCA Phase1+A+B、三模型各三项 LoRA、五组 PPL、18 次通用 benchmark 运行及 18 次本地下游评估。MMLU/CMMLU/CLUE 覆盖全部叶子任务，各一条；C³Bench 五任务、Fuxi 21 任务均覆盖。
- 每阶段默认两步，CPT 长度 128，训练原文最多 512 字符；原始数据与压缩包保持原样。少量训练/评测数据、模型、缓存和报告集中放在新 `outputs/smoke/vN/`。
- 训练验证实际步数、有限 loss 与检查点发布；评估检查完整推理和评分/指标产物。必测失败和后继阻断分别记录，独立项目继续。
- 增加模型构建参数 `MODEL_ROOT` 隔离全局构建指针；CPT 增加 `MAX_STEPS` 与 `SAVE_ONLY_MODEL`，下游增加 `SAVE_ONLY_MODEL`。正常训练默认值保持不变。
- 所有必测通过而 Fuxi 裁判未配置时：退出 0，汇总为 `partial`，`complete=false`，三项裁判 `pending`。真正失败退出 1，中断退出 130。密钥不写入汇总；运行代码不新增 Git、哈希、依赖版本锁定行为。

## 本地验证

临时检查脚本与产物在 `/tmp/cc-smoke-check.d27DRt/`；未添加仓库内测试目录或模拟结果作为真实冒烟报告。

- Shell 语法、Python AST、CLI help 与 `git diff --check` 通过。
- 使用当前真实数据完成全部 12 类采样，含 tar.xz 前缀读取、古汉语三任务与 metadata 配对、CSC 八域、C³ 五任务及 Fuxi 21 任务。读取流式 tar 时发现并修复 TextIOWrapper 的 seekable 问题；按真实导出格式保留单 assistant 的 messages。
- 使用正式 ms-swift 模板编码真实采样数据：Phase1 train/val 为 3,517 / 100 个有效目标，Phase2 为 4,970 / 150；CSC 为 1,484 / 78，CGEC 为 1,436 / 22，古汉语为 3,828 / 228。所有编码序列均有有效监督，实际完成 A=512、B=剩余的原生缓存准备。
- 隔离调度检查覆盖两卡并行、单卡串行、GPU 不共享、独立输出、失败注入后的依赖阻断、Fuxi pending / 已配置裁判时 complete、真实非零子进程退出和产物检查。裁判分支用替身，无 API 请求，汇总未写密钥；调度检查不等同于模型训练/推理通过。
- 实际执行四份 CPT shell 的参数转发检查（替身 Python），验证 max_steps=2 与 save_only_model=true。实际执行构建总 shell 的隔离检查（替身子脚本），正式两个模型指针内容未变。
- 临时本地环境为 Python 3.11.15、ms-swift 4.6.0、Transformers 5.19.0；服务器应使用项目的 Python 3.12 两套环境。本地没有完整权重/GPU，没有执行真实模型构建、实卡训练或完整推理。

## 用户下一步

两套环境与根目录 data/ 就绪后，传入 Base 模型路径运行集中入口。提供本轮 `summary.json`、`summary.md` 和失败项目日志即可继续定位。少步结果不代表模型性能、EOS 学习、全量数据内存、四卡 DDP 或优化器恢复验收。
