# 评估推理 KV cache 修复

## 问题

第二轮 CPT 和 LoRA 训练产物的 `config.json` / `args.json` 保存了 `use_cache=false`。评估入口加载模型后未覆盖该值，HF 自回归生成因而不使用 KV cache。C³Bench 实测在 batch 16 时约为 9.54 秒/样本，50,000 条请求预估超过 132 小时。

## 修复

- `scripts/evaluation/inference.py` 在完整模型加载后设置 `model.config.use_cache=true` 及 `model.generation_config.use_cache=true`。
- LoRA 加载后再次校正 engine 内的模型和 generation config。
- 预测身份和 predictor 运行信息记录 `use_cache=true`，不复用无缓存的旧输出。

PGCA 模型已有 `PGCACompositeCache`，同步维护主干和因果特征编码器缓存；本修复是启用已有生成能力，不改模型权重。

## 验证边界

本地单元测试验证 model/generation config 同时开启且 Manifest 记录新身份。尚未在服务器 A100 上复测实际吞吐、显存和 batch 上限。
