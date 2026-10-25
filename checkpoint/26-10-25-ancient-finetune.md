# 古汉语多任务 LoRA 入口

## 状态

已接入 `scripts/train/ms_swift/finetune/ancient.sh`，使用已验收的 `data/downstream/ancient/train/final`，不改写 final 数据。训练为 all-linear LoRA，PGCA 启用，辅助头冻结，字音/字形辅助损失关闭。

## 关键决定

- 根据 final metadata 生成仅含 `messages + channel` 的运行视图；每次启动核对 messages 一致性、任务类型、训练就绪标记和验收报告计数。
- 使用锁定 ms-swift 的原生 channel loss，记录 `eval_loss_c2m`、`eval_loss_m2c`、`eval_loss_punctuate`，以三者等权宏平均 `eval_macro_task_loss` 选优。
- 新增 `cc_llm_ancient` 模板，继承数据验收时的 Qwen3 mixed template，保留 non-thinking 前缀及 `disable_ignore_empty_think=false`。CSC/CGEC 仍使用 `cc_llm_correction`，编码行为不变。
- 命令行传 `truncation_strategy=delete`，锁定版 ms-swift 会将它映射为模板内部 `raise`；配合 `strict=true`，超长数据直接失败，不会静默删除。

## 验证证据

- 真实 final 数据视图：Train 53,000（c2m 38,000 / m2c 10,000 / punctuate 5,000）；Dev 1,800（三项各 600），与 `report.json` 一致。
- Shell/Python 语法、`git diff --check` 通过。
- `tests/test_ancient_finetune_views.py` 与 `tests/test_finetune.py`：13 passed；5 项需 ms-swift/torch 的集成测试在当前无该环境的本机跳过。
- 使用轻量框架桩验证了三项 loss 算术宏平均及 `trainer_state.log_history` 写入逻辑。

## 尚未验证

未在服务器执行 1.7B 四卡冒烟或正式训练。首次启动应确认 `args.json` 中的 template/channel/selection metric，并在首次验证后确认四个有限指标和 best checkpoint 选择正常。
