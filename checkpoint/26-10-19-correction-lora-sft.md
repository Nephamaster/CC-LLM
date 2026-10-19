# CSC / CGEC 下游 LoRA 入口

## 范围与数据

- 入口：`scripts/train/ms_swift/finetune/{csc,cgec}.sh`，共享 `common.sh`，复用 ms_swift/env。
- CSC 使用 `data/downstream/csc/sft/{csc_mix_dedup,cscd_dev}.jsonl`；CGEC 使用 `data/downstream/cgec/sft/final/{train,validation}.jsonl`。
- 两者都是 source-only messages：不重新添加纠错指令，不重划数据，不使用测试文件训练。不另造 CPT 式缓存，使用原生 `load_from_cache_file`。
- 古汉语暂不接入，等待服务器 `data/downstream/ancient/train/final` 完成。

## PGCA 与 LoRA

- 用户确认主干、PGCA 均为 all-linear LoRA，关闭字音/字形辅助损失。默认沿用旧 CSC 的 rank=16、alpha=32、dropout=0.05、LR=1e-4、5 epochs、长度 768；参数均在顶部。
- `all-linear` 会匹配辅助头的部分线性层。插件在 Trainer `on_init_end` 冻结闲置辅助头（含其 LoRA），早于优化器/DDP 初始化；否则关闭辅助 loss 后仍可能留下无梯度的可训练参数。
- `cc_llm_correction` 继承原 qwen3_nothinking 编码，使用标准 assistant-only SFT loss，不导入 CPT 多任务 loss/optimizer。训练前向显式禁用 KV cache 和辅助 logits。
- ALIGN config 保存的 pgca_enabled=false 不能沿用到 SFT/adapter 评估：本模板在训练及生成前开启 PGCA，不修改基座磁盘文件。现有评估器从 adapter args.json 加载插件/模板，已覆盖原基座加 adapter 的路径。

## 产物与恢复

- 输出为 LoRA adapter，不能把 adapter 目录当完整模型。评估使用原基座 MODEL 加 adapter LORA。
- 默认输入为最近完成的 PGCA-CPT 指针；显式 MODEL 优先。输入名称尾部 -ALIGN 不传递到下游名称，按根 README 的 -CSC/-CGEC 命名。
- 原生版本目录及日志，成功训练发布 next_stage_model.txt。恢复时读取 adapter 原基座，不误用后来更新的 CPT 指针；继续原版本。

## 验证边界

- Transformers 5.19.0 / PyTorch 2.10.0 CPU / 普通安装的项目固定 ms-swift：14 项测试通过，包含脚本参数及恢复路径解析、辅助头冻结、PGCA/context LoRA 梯度、Base 与 PGCA 真实小模型 SFT、双进程 Gloo、模板编码对照以及现有评估入口加载 adapter 并生成。
- Transformers 5.12.1：12 项相关测试通过，包含 PGCA 单进程真实训练及 adapter 推理。原生 Dataset map 即使 num_proc=1 也可能使用 Manager IPC；受限沙箱禁止本地 socket 时可表现为 EOFError，应开放测试所需本地 IPC，而不是改模型代码。
- 未执行服务器 1.7B GPU 微调；默认 batch=8 仍需按实际显存调整。超长样本按旧 CSC 的 delete 策略删除，跨 tokenizer 对照应记录实际保留数量，不能假设原始文件相同就意味着有效样本相同。
