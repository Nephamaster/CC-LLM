# CSC / CGEC 裸句输入协议

## 决策与边界

用户观测到纠错输出夹带提示词，希望去除每条输入里的固定任务说明。对分别训练的 CSC/CGEC 专用模型，采用 `user=带错原句、assistant=正确句` 是可行的输入协议；但不能推断“重复输入提示词必然导致模型生成提示词”。本地 ms-swift/Nepham/CCLLM/csc_char_pgca.sh 显式使用 loss_scale=default；框架该策略只监督 assistant。未取得具体异常输出对应 checkpoint 的实际标签/训练参数，因此未认定根因，也未声称去提示词已改善模型行为。

只移除 user 内容里的旧业务包装，不改变 messages、聊天模板、assistant 目标、样本顺序、数据划分或训练 loss 策略。CPT 的 loss_scale=all 是不同用途，本轮未改。若将两类任务合并为同一个模型，需要重新提供任务区分。

## 实施

`python -m scripts.data_factory.strip_correction_prompts --apply`：默认预览，--apply 时每个 JSONL 留 `.with-prompts.bak` 备份，再原子替换；不全局 replace，不裁剪原句内部文本。既有备份不覆盖。常规重复执行不会再次修改已转换数据。

实际已转换，行数与目标不变：

| 文件 | 条数 |
| --- | ---: |
| csc/sft/csc_mix_dedup.jsonl | 352019 |
| csc/sft/cscd_dev.jsonl | 5000 |
| cgec/sft/final/train.jsonl | 53759 |
| cgec/sft/final/validation.jsonl | 1137 |

CSC 当前没有 final 子目录，其 sft 根目录两份文件就是现用成品；没有另行移动或重命名。CGEC prepare_report 更新为 prompt="" / input_format=source_only_messages，prepare_cgec.py 今后直接生成裸句 user。

CSC/CGEC 评估 prepare 同步使用原句 user，协议标识升级到 source-only-v2，防止旧带提示词产物被复用；CGEC 默认 test 根目录同步为实际的 data/downstream/cgec/test。原始 benchmark 文件、评分算法和回答清理不变。新评估用新的 RUN_NAME/OUT。

## 使用注意

- 用新 JSONL 重新或继续 SFT，并重新编码；已有模型行为、旧 tokenized cache 不会自动变更。聊天模板与 assistant-only 监督仍应保留。
- 仓库当前仅有 CPT 训练入口，没有 CSC/CGEC SFT 启动脚本；没有把 CPT 脚本改成下游训练。所查看的外部 CSC 旧脚本仍指向 CLLM 数据目录，使用它训练时需明确指向这批新数据；本轮未修改其他仓库。
- data/ 被 Git 忽略，需要单独同步；备份不是训练集，不应使用不限定扩展名的目录扫描把它们加入训练。
