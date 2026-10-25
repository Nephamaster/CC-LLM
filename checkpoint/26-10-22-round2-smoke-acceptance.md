# 第二轮实卡冒烟验收：可以进入正式训练准备

## 依据与结论

检查 outputs/experiments/rnd2/smoke/pgca-smoke.5z6BB2 的 preflight/smoke report、三阶段 trainer_state/config/cc_llm_state/args、logging.jsonl 和 console log。没有重新运行模型、读取 optimizer pickle 或修改训练代码。

双 A100-SXM4-40GB、BF16、DDP、每卡 batch=1、累积=2 的三阶段冒烟通过。TODO 第 3 项可按“小规模实卡验收”完成；下一步是第 4 项的正式配置与训练，不是直接接着冒烟 checkpoint 训练。

| 阶段 | 优化步 / epoch | 本阶段有效目标 | 最后 eval_clm_loss | 日志最大 memory(GiB) |
| --- | --- | ---: | ---: | ---: |
| Phase1 | 5 / 1.0 | 32,768 | 0.85785980 | 21.68 |
| Phase2-A | 5 / 1.0 | 65,536 | 0.82358646 | 18.81 |
| Phase2-B | 5 / 1.0 | 65,536 | 0.75011382 | 31.17 |

memory 来自 logging.jsonl；summary 的 memory_gib=null 只是 trainer_state.log_history 没有保存这项字段，不是零显存或缺少训练。记录值不等于独立监控得到的所有卡最坏峰值。

## 交叉核验

- 三阶段实际输入路径依次为 models/rnd2 初始模型、本次 ALIGN checkpoint-5、本次 A checkpoint-5；没有串用第一轮或旧完成指针。三阶段 best/latest 均指向本次 checkpoint-5。
- config 保存字音/字形 0.03/0.03、辅助 warm-up 10M；Phase1 PGCA 旁路，A/B 开启。实际训练长度为 2048/4096/4096，WD=0.01，LR warm-up 为 3%/3%/1%。
- Phase1 所有分支梯度为零、主干有梯度；A 主干梯度为零，所有分支组出现非零梯度；B 主干与分支都有非零梯度。
- A 的 Q/K/context 在第 3 步开始有非零梯度，与零初始化 O 和首步学习率 warm-up 相符；不能把前两步内部零梯度误判为断路。
- 三阶段的 4 次验证（step 0/2/4/5）全部有限，均计数 1,047 个有效目标。eval_loss=eval_total_loss；Phase1 还等于 eval_clm_loss；A/B 分项加权比例及总损失关系重新计算一致，未见双卡倍数偏差。
- 训练日志 clm_tokens 分别累加为 32,768 / 65,536 / 65,536；持久化辅助累计计数为 0 / 65,536 / 131,072。独立 preparation_report 文件未同步，但 Phase1/A console 中有完整准备 JSON：A=[0,65536)、B=[65536,131072)，分母与区间可交叉核对。
- A 初始 CLM 验证值精确等于 Phase1 末值；B 初始值精确等于 A 末值，符合阶段加载衔接。
- 真实模型预检的汉字、英文、零目标和未知读音案例均通过。英文辅助损失为零；零目标总损失为零；未知读音案例 phon_count=0、phon_loss=0，仍保留有效字形监督；梯度有限。

## 非阻塞提示与边界

- 最终辅助 ramp=131072/10M=1.31072%，没有实卡覆盖完整 warm-up；本地已验证 0/5M/10M/20M 数学逻辑。辅助头只训练了少数步，字音准确率仍为 0 不据此否决链路，也不能据低 CLM loss 宣称效果改善。
- 有 NCCL barrier/device 和退出时未显式 destroy_process_group 的警告。本轮三个阶段均保存、衔接并结束，未发现这些警告导致结果失效；后续启动前确认旧进程已退出。没有在本次分析中改动框架退出逻辑。
- 恢复 best 时出现 lm_head.weight missing 提示。配置 tie_word_embeddings=true，checkpoint 配套模型明确声明 lm_head.weight 与 model.embed_tokens.weight 绑定，与共享权重保存的情形一致；不将此提示单独判为模型头丢失。本地未提供 safetensors 大权重，未额外装载核验实际参数别名。
- 这是合成重叠数据、两卡和少量步骤；不是模型效果评测，也不保证八卡长训/完整验证集的最坏显存或稳定性。

## 正式启动前仍需做

从 models/rnd2/Qwen3-1.7B-Base-PGCA 的未训练初始模型启动 Phase1，不使用 smoke 或第一轮 checkpoint。隔离第二轮训练根目录与 A/B 衔接指针。

正式配置为 8 卡、每卡 batch=1、梯度累积 32/8/32、WD 全部 0.01、LR warm-up 3%/3%/1%；学习率沿用方案。当前正式脚本仍有旧默认值（WD=0、A 累积=32、B warm-up=3%，Phase1 默认也不是 rnd2 模型路径），需要明确配置后才能启动；本次没有擅自修改这些脚本。

开始后重点看首轮完整验证和前几十步的 loss、有效 token 计数及显存，不将冒烟的合成数据结果当正式基线。
