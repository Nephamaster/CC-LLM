# PPL 评估部署与口径

## 实施边界

入口 `scripts/evaluation/ppl/run.sh` / `python -m scripts.evaluation.ppl.evaluate`，独立环境 `scripts/evaluation/ppl/env/`，默认只读 `data/ppl/`，输出 `outputs/evaluation/ppl/<MODEL_NAME>/`。不依赖 ms-swift、CLLM 外部源码或 SFT args.json；不修改数据、模型、训练入口或 Phase0 验证。

支持本地完整 HF 模型及自包含 PGCA AutoClass；tokenizer-only、LoRA adapter、缺失权重、模型加载缺键会拒绝。每个模型用自己 checkpoint 内的 tokenizer，不用 Char tokenizer 给原始 Base 评分。

## CLLM 迁移时确认的问题

来源 HEAD `c7cf30a5ff92e13e68c1d2a07c4cccc50f10f02c`，`src/cllm/ppl.py` SHA256 `9b9179727d09d7ef89482f754fb7c55958277b1685520425cadf50b268cc2adb`。

- token offset 区间可能重复覆盖一个 Unicode 字符；长度求和不是可靠的原始字符数。改用 `len(text)`，另报 UTF-8 字节数。
- stride=window 时边界第一个标签会被 causal shift 丢弃，原计数却可能包含它。窗口被静默缩短后 stride 还可能大于窗口。现在要求 `0 < stride < window`，超出模型容量直接报错；长文逐窗全量计分，目标 1..L−1 恰好一次。
- 不读 `outputs.loss`，只从 logits 和实际 shift 后目标计算 FP32 CE、FP64 总和，避免评估口径与辅助损失或框架聚合绑定。PGCA 分支照常执行，不请求辅助头 logits。
- 分领域和 overall 均累计 NLL/分母后计算 PPL，不沿用旧代码的领域 PPL 算术平均。编码密度与有效目标密度分别报告。

默认仍为无 BOS/EOS、逐文档 window=2048/stride=1024。字符/字节分母是全文，但每篇首 token 概率不计入 NLL；指标属于条件 NLL 归一化，并非消除 tokenizer 差异后的完整文本概率。不同 tokenizer 的 token PPL 不可直接横比；相同 token 窗口也不保证相同原文上下文范围。

## 真实数据快照

2026-10-19 全量结构校验：六个文件共 1,522 文档，2,470,538 个 Unicode 码点、6,882,903 个 UTF-8 字节，无空文本或结构错误。未补齐/平衡各领域样本数。

用本地 `models/Qwen3-1.7B-Base-Char` tokenizer（Transformers 5.19.0）逐条编码：2,377,148 encoded tokens，预计无 BOS 协议计 2,375,626 个目标；无越界 ID、无少于两 token 的文档。最长文档 26,193 tokens，说明不能直接截断到 2048 后声称全量 PPL。该步骤只验证 tokenizer/输入，没有运行真实模型。

## 防误用与验证

保存逐文档 NLL/计数/文本哈希和 manifest，配置、tokenizer、自定义代码、输入有 SHA256；大权重仅记录路径/大小/mtime。manifest 不复制可能含凭据的 args.json。中断不发布成功 metrics，保留诊断；不自动续算或覆盖，重跑换 RUN_NAME/OUT。

新增测试覆盖目标计数、边界、Unicode/byte fallback、纯 LM loss、加权汇总、异常/溢出、失败产物保护，以及真实 CPU Qwen3 和自包含 PGCA checkpoint 的保存/加载/计分。

- Transformers 5.19.0：PPL、目录约定与 CSC 迁移测试合计 51 项通过；实际 shell 入口从 `/tmp` 跑通 CPU 小模型并生成报告。
- 固定 Transformers 5.12.1、PyTorch 2.10.0+cpu、Python 3.13.15：上述测试加已有下游评估回归，71 passed、3 skipped；跳过的是已有可选 THULAC/ms-swift 集成测试，不是 PPL 测试。PPL 自身 33 项全部通过。
- 初次扩大回归时旧临时环境缺少 SacreBLEU/Levenshtein，未改评分代码；新建 `/tmp/cc-llm-ppl-tests.2nhChi/env/` 普通安装依赖后复验通过。该环境依赖检查、shell 语法检查和 `git diff --check` 均通过；项目无 `__pycache__`。

实际服务器 1.7B/PGCA 权重、CUDA 显存/速度、全量 PPL 数值仍待服务器执行；CPU 小模型通过不等于已取得真实模型分数。
