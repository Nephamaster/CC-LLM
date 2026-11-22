# Phase2 GPU OOM 与旧 40GB 运行的比较

基线：master / 981b38f，保留工作区现有未提交的编码批次/缓存修复。本次仅分析，不修改训练配置或模型。

用户日志：训练进度 0/2054，CUDA OOM，片段包含“33.80 GiB memory in use”及“Including non-PyTorch memory, this process has 12.95 GiB memory in use”，本进程 PyTorch allocated 11.80 GiB，reserved but unallocated 742.53 MiB。随后 rank0 exit=1、torchrun 终止另一个 rank，NCCL 提示未 destroy。这次已进入 GPU 训练，与此前 CPU 编码 SIGKILL 不同。

近期为编码问题所作工作区 diff：DATASET_NUM_PROC 默认 32→4、ENCODING_BATCH_SIZE=32、Phase2-A 原生缓存默认复用，以及 prepared 视图自动重建。未改训练 batch、长度、BF16、梯度检查点、训练 loss 或前向图。

当前 Phase2-A 脚本 MAX_LENGTH=4096，BF16、梯度检查点开启。此前提供的完整命令显式每卡 batch=1；本次服务器 args 尚未提供，不能仅据命令建议确认实际生效参数。

本地历史真实配置：outputs/experiments/rnd2/training/Qwen3-1.7B-Base-PGCA-CPT/adaptation/v0-20261023-101846/args.json 为每卡 batch=1、max_length=4096、BF16、梯度检查点=true、8 卡、梯度累积8。旧 checkpoint-1027/config.json 的 PGCA 层为 11..16 共6层；当前 models/Qwen3-1.7B-Base-PGCA-Shared/config.json 为 9..17 共9层。两者上下文编码器均4层、hidden512、heads8、FFN1536、最大读音数16。新产物的 PGCA 层数确实增加，不能声称模型训练显存与旧版本完全一致；增加的显存比例也不能直接等同层数比例。

优先调查同卡显存占用：PyTorch 完整 OOM 文本会列出其他进程的占用，用户片段缺失了 33.80 GiB 前面的 PID/上下文，尚不能直接确认归属。若是其他进程，则与本进程12.95 GiB相加约46.75 GiB，足以解释48GB卡反而OOM；需完整OOM首行及 nvidia-smi GPU/compute-apps 输出核实。不得盲杀其他用户进程。

reserved未分配约0.73GiB并不大，不能把 expandable_segments 当作清除数十GiB占用的解决方法。NCCL/其他rank终止是错误后的清理现象，不是目前证据支持的首发原因。优先使用空闲GPU；若GPU独占仍OOM，再核对当前run args/model config和完整分配失败信息。本地没有本次GPU权重/运行环境，未复现或验收。
