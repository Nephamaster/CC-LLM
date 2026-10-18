# PPL 评估目录约定

本目录保留给 Base、Char 和 Phase1 语义对齐模型的 PPL 评估；本轮不新增尚缺的评估实现。独立环境使用 `scripts/evaluation/ppl/env/`。

结果应保存到 `outputs/evaluation/ppl/<MODEL_NAME>/`，其中语义对齐模型使用 `M-ALIGN`。原始模型用原名，Char 用 `<Base原名>-Char`；PGCA 与其他后缀规则见根目录 README。

实施时需固定同一份原始评估文本、文档边界、上下文长度、stride、BOS/EOS 和有效标签规则。按全部有效目标累计 NLL 与 token 数，再计算 exp(NLL/N)，不能先对文档 PPL 求平均，也不能使用含音形辅助损失的总 loss。

Base 与 Char 的 tokenizer 不同，原始 per-token PPL 不能直接横比；应同时记录 token 数、tokenizer 版本，并增加按原始文本字节/字符归一化的 NLL（明确计数口径）。

现有 `scripts/validation/validate_forward.py` 是迁移权重与有限 forward 诊断，依赖新增 token 映射、会限制输入长度；不是 Base/Char/ALIGN 统一 PPL 评估入口，不应改名后直接冒充该评估。
