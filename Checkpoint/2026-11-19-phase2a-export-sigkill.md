# Phase2-A 原始编码导出被 SIGKILL

基线：当前 master 工作区；此前 prepare_data.py 的 batch/卡数自动重建与两份 README 修改尚未提交，保持不动。本次仅分析用户服务器日志，未修改训练代码。

用户日志：原始读取完成 5,739,897 条 train / 5,746 条 val，进入 phase2a/export/encode，以 num_proc=32 编码。处理至 124,000 条时导出子进程被 SIGKILL，总耗时 3003.1 秒；容器 OOM 计数不可用。出现若干 131k–204k Token 原文警告。尚未进入 prepared 视图准备或 GPU 训练。

当前代码：phase2_a.sh 默认 MAX_LENGTH=4096、truncation_strategy=split、DATASET_NUM_PROC=32，当前 LOAD_FROM_CACHE_FILE 默认 false，可从环境覆盖。plugin.py 只在 num_proc=1 时将编码转为当前进程执行（None），32 保持并发。核对相邻 ms-swift 源码，其 split 路径先完整编码文档再切分，编码 map batch 为 1000；这份相邻源码不等于已确认服务器当前依赖版本。

判断：优先怀疑 CPU 编码内存压力，尤其并发与长文档扩展输出；SIGKILL 本身尚不能证实 OOM，也不能排除容器/平台外部终止。超长 Token 警告不等于训练将接收同长度序列，split 后应遵循 4096 限制。读取阶段单 shard 自动降进程提示不是本次强杀原因。

建议先用 DATASET_NUM_PROC=1、TOKENIZERS_PARALLELISM=false、LOAD_FROM_CACHE_FILE=true 重试既有阶段，保留原训练 batch 和输入 checkpoint。若导出目录已经不完整，保留原目录，显式 RAW_CACHE 指向新的 phase2_4096_retry.raw，并让 A/B 共用该值。此前已完成 Phase1 不必重跑，SIGKILL 的导出尚不能当作完成缓存使用。进一步仍失败时，需要失败时内存/平台终止原因，并针对编码 batch 增加窄配置；当前不擅自改采样数据或截断策略。

证据边界：仅源码核对、用户日志与已下载历史小样本编码报告；没有在用户机器复现，也未执行全量编码/GPU 训练。

## 编码缓慢补充

用户新截图实际仍为 num_proc=16，108,000 / 5,739,897 条耗时 35:29，整体均速约 51 条/秒，按已处理条数线性外推总计约 31 小时（非稳定完成时间承诺）。用户报告 Phase1 1B 编码约 10 分钟，但缺少对应日志以确认冷编码、缓存命中或其他阶段耗时。

历史数据看板（2026-11-10）列 Phase1 543,894 文档 / 980,118,370 原文 Token，Phase2 5,739,897 文档 / 10,035,777,803 原文 Token：文档约 10.55 倍、Token 约 10.24 倍；平均每文档约 1802 / 1748 Token。不能用“Phase2 平均文档更长”解释这次差距；长尾存在，但其性能贡献未知。

两阶段调用相同导出和模板路径，GPU 音形分支不是该 CPU 编码工作的新增计算。Phase2 的 10B 预算与 A 的 100M 划分在原始编码之后执行，A 首次导出会编码完整共享池供 B 使用；这里不据此假定实际输入超出已统计规模。当前 Phase1 默认 load_from_cache_file=true、Phase2-A=false，可能导致旧缓存复用与冷编码不可比，但用户实际环境覆盖及历史缓存命中尚未确认。

下一步应比较 Phase1 同阶段编码日志，并检查 CPU/内存/磁盘实际负载；需要优化时可仅开放编码 map batch，降低单批扩展内存后做同样本吞吐对照，不把增加进程、关闭警告、改变训练 batch 当作已验证的提速方案。尚未修改产品代码。

## 编码批次控制实施（接续上述分析）

用户补充：单进程预计约 20 小时才改为 16 进程，且进程再次退出。已异步请求最新退出前 10 行日志，尚未取得；不把本次退出直接定性为 OOM 或 SIGSEGV。

修改 CPT 插件既有预处理包装：使用 ENCODING_BATCH_SIZE（默认 32）覆盖编码 map 的单批文档数，启动时打印实际进程数与编码批次；保留单进程转 None 的修复以及既有模板、标签过滤、Token 切分逻辑。四个 CPT 脚本顶部直接暴露并导出该参数，DATASET_NUM_PROC 默认从 32 调为 4；Phase2-A 的 LOAD_FROM_CACHE_FILE 默认改为 true。同步根 README 与 CPT README。未修改此前 prepare_data.py 的视图重建实现，也未修改数据、模型或训练 batch/学习率。

临时验证均在 /tmp/cc-encoding-batch-20261119：check.py 的 4 项检查通过，覆盖包装参数透传、单进程不建池、批次覆盖/非法值、重复包装；四份 shell 的 bash -n 与 git diff --check 通过。compare.py 使用实际 Shared tokenizer、1024 条 Phase2 验证文本及一条重复扩展的长文本，在真实 datasets/Arrow 上比较分词与切分：1 进程/批1000 为 3.050 秒，4/1000 为 1.427 秒，4/32 为 1.675 秒，2/32 为 2.017 秒；四组输出 1131 行，Arrow 表内容完全一致。日志 compare.log，结果 comparison.json。

上述对照仅验证原生 tokenizer/Arrow 的小样本通路，不包含完整 Swift 模板/导出流程，不能推导服务器全量耗时或证明 4/32 是最佳配置。较小批次控制文档展开的中间结果规模，不保证纯吞吐提升。当前临时环境 Python 3.14.6，未安装 Torch/ms-swift，未做完整 Swift 或 GPU 训练验收。推荐服务器以 4 进程、批32 开始观察，并使用独立新 RAW_CACHE 保留已中断的目录；若再次退出，必须结合最新退出信号与内存/CPU/磁盘证据继续定位。
