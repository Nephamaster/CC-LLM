# SFT 固定 Dev 生成验收

范围：在 `master`、HEAD `341860b` 的已有未提交工作上增量实现；保留此前参数、环境和 README 改动。未提交或推送。数据、模型源码和训练超参不变。

## 已落实的决策

- CSC、CGEC、ACT，以及 Base/Char/PGCA 路线共用检查。每次保存时每任务取 Dev 前 1 条，只记录结果并继续训练，保持原有 loss 选模。
- 正式下游 `all|infer` 前重新加载选中基座和保存的 LoRA，每任务取前 10 条，ACT 按三个 channel 共 30 条。每任务停止率与 im_end 结束率均至少 90%，截断、重复、异常率各至多 10%。默认未通过即停止该项正式推理。
- 完整验收检查实际训练模板的 EOS label 和推理前缀；每任务首条另做 LoRA 开关 logits/生成对照，以及完整前缀对逐 token Cache 的 logits、长度和生成健康状态对照。PGCA 检查主干及音形 Cache。
- Cache 生成文字允许因近似并列 logits 而不同；数值需在容差内，停止、截断、重复、异常标记需一致。失败需要调查，不自动归因为 Cache 实现错误。
- `DEV_CHECK_*` 配置集中在 `dev_metrics.py`。报告在 checkpoint 或评估目录的 `dev_check/vN/{report.json,samples.jsonl}`；不复用旧验收，不增加哈希或 Git 运行依赖。
- 冒烟显式每任务 1 条、使用自身生成上限，`report` 模式只豁免生成质量门槛，结构/LoRA/Cache 失败仍阻止步骤继续。

## 代码和文档

新模块 `scripts/train/finetune/dev_check.py` 执行验收，`dev_metrics.py` 定义门槛与输出统计。训练插件增加 on_save 回调及 Dev/task 记录；训练 common 传递任务和 Dev 路径。三个下游评估入口通过 evaluation/common 调用完整验收；推理 manifest 记录模式及报告位置。集中冒烟设置独立的小样本检查参数。

根 README、训练/评估/冒烟 README 和 `pipeline.md` 的 5.5 节已更新。未重写其他研究方案内容。

## 本轮验证

临时脚本、数据和小模型均位于 `/tmp/cc-sft-dev-check/`；环境位于 `/tmp/cc-sft-check-env/`，从本机已有依赖缓存恢复。Python 3.11.16、ms-swift 4.6.0、Transformers 5.19.0、Torch 2.15.0+cpu、PEFT 0.21.2。项目正式环境要求仍是 Python 3.12。

| 检查 | 本轮结果 | 证据 |
| --- | --- | --- |
| 普通 Qwen3 与上下文 PGCA 的真实小模型、真实非零 LoRA、序列化后 CLI 重载 | 通过；主干/音形 Cache 长度正确，完整/增量 logits 最大误差均小于 1e-6，LoRA 开关有数值变化；零 B adapter 被拒绝 | `models.log`、`native-cli/v1/`、`pgca-cli/v1/` |
| 两种小模型各执行真实 ms-swift 两步 LoRA 训练，每步保存 | 通过；两个 checkpoint 均完成轻量检查、发布选中 checkpoint；普通模型质量失败未中止第二步训练 | `fit.log`、`fit-pgca.log` 及对应 checkpoint 报告 |
| 对真实两步普通模型的选中 adapter 执行 required 完整验收 | 结构检查通过、质量失败，CLI 按预期返回 2 | `trained-required.log`、`trained-required/v0/report.json` |
| 本地真实 Char tokenizer 与真实三个 Dev 子集的模板编码 | CSC 10、CGEC 10、ACT 30 条全部 EOS label 受监督、推理前缀与训练前缀一致；该项不执行大模型前向 | `templates.log` |
| 错误 logits、错误音形 Cache 长度注入，异常时状态恢复，回调仅主进程执行 | 通过；注入异常被拒绝，训练/冻结/cache 设置恢复 | `negative.log` |
| 90%/10% 边界、分任务判定、罕见汉字提示、重复与异常检测、固定采样、报告保存、required/report 退出码 | 通过；策略与 CLI 控制测试，不是模型质量验收 | `check_policy.py` |
| 三个 shell 入口的失败拦截/成功放行；prepare/score/judge 不加载验收模型 | 通过；使用替身命令验证调用顺序和退出行为，未执行真实 benchmark | `check_entries.py` |
| Python AST、修改 shell 的 bash -n、git diff --check | 通过 | 本轮工具执行 |

## 验收边界

- 尚未执行真实 GPU 大模型和已训练业务 LoRA 的验收，也未验证多卡训练保存回调、BF16/FP16 GPU 容差或 vLLM/批量 padding 的数值路径。小模型 CPU 检查不能替代这些结果。
- 自动容差为 FP32 1e-4/1e-4、FP16 0.02/0.005、BF16 0.1/0.02（atol/rtol），实际值入报告并可覆盖；GPU 上出现失败时应据实调查。
- 自由生成不输入参考答案；参考长度只用于预算，真实参考 token 还用于 teacher-forced Cache 对照。默认上限 512，报告标明预算是否覆盖参考长度。
- 合法罕见汉字不会自动判坏；异常和重复检测属于启发式。此功能不证明答案正确、参考数据无错配，也不会修复既有权重中的 EOS 学习或重复问题。
- 后续直接使用现有训练和下游评估脚本。迁移数据后若原 Dev 路径不存在，设置 `DEV_CHECK_DEV`；需要提前检查时按训练 README 的独立验收命令执行。
