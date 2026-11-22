# Phase2-B 准备数据时磁盘写满后的继续方式

证据：用户附件 /home/nepham/.codex/attachments/dc814601-ce33-4198-b7da-98dacb617f23/已粘贴的文本.txt。本地 master / 981b38f，保留既有未提交变更；本次未改训练代码。

Phase2-A 已完成 2054/2054，exit_code=0，发布 checkpoint：outputs/train/Qwen3-1.7B-Base-PGCA-Shared-CPT/adaptation/v1-20261121-203539/checkpoint-2054。随后 B 已确认 phase2_4096_batch32.raw 完整（train 6,608,103 行，val 6,592 行），在生成 prepared 的 train_b 到 4,188,999 行时因空间不足失败；尚未启动 B 训练，不存在本次 B 的训练恢复步数。

用户清理空间后只运行 STAGES=phase2_b，PHASE2_B_MODEL 显式指向上述 A checkpoint，RAW_CACHE 沿用 outputs/cache/Qwen3-1.7B-Base-PGCA-Shared/phase2_4096_batch32.raw。不使用 RESUME_FROM_CHECKPOINT，不重跑 A，不重新导出原始编码。B 的准备视图需要重新生成，不能把生成器进度当作可续写断点。

命令示例沿用此前两卡、每卡 batch=1、梯度累积16、max_length默认4096；若服务器实际实验参数不同，应沿用实际参数和原辅助任务开关。不要在清理空间时删除上述原始编码或 A checkpoint。确认是目标文件系统有可用空间；准备期间同时存在临时数据与最终数据，占用不能只按最终文件大小预留。

仅核对用户日志、当前启动器/阶段脚本；未在用户服务器重新执行准备或训练。
