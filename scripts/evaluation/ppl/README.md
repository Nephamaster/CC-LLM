# 原始文本 PPL 评估

从本地 CLLM 的 `src/cllm/ppl.py` 迁移并校正计分，支持完整 HF Base、Char、ALIGN、CPT checkpoint，包括 checkpoint 自带 AutoClass 的 PGCA 模型。直接使用各模型自己的 tokenizer 和语言模型 logits，不使用 SFT/chat template，也不要求训练 `args.json`，不依赖 ms-swift 或外部 CLLM 源码。

## 环境

运行前自行激活合适的环境即可，例如已有的 ms-swift 训练环境。`run.sh` 直接调用当前 `PATH` 中的 `python`，不加载 `scripts/evaluation/common.sh`，不绑定 `ppl/env/`，也不读取 `PYTHON` 环境变量。

当前环境已满足依赖时不需要重复安装；需要补齐依赖时，在该环境中执行：

```bash
python -m pip install -r scripts/evaluation/ppl/requirements.txt
python -m pip check
```

requirements 中的 Transformers 固定为与本仓库训练入口一致的 `5.12.1`，实际运行版本写入 manifest；安装前注意是否会改变共享训练环境的依赖版本。脚本不会自行安装或切换环境。默认 SDPA，不依赖 flash-attn。

## 数据与启动

默认只读 `data/ppl/*.jsonl`，每行一个文档，必须有字符串 `text`；可带 `id`、`domain`。领域取文件名去掉 `zh_`，如果行内有 domain，必须与文件名一致。原样保留空格、换行和标点，不清洗、不重采样、不拼接不同文档、不截断长文。当前六个文件共 1,522 条：finance 500、game 158、general 240、government 192、movie 346、tech 86。

先校验全部输入，不加载模型、不写数据（只用 Python 标准库即可）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m scripts.evaluation.ppl.evaluate --inspect
```

完整评估：

```bash
# MODEL 必须指向可加载的完整权重，而不是 tokenizer 目录或输出根目录。
CUDA_VISIBLE_DEVICES=0 \
MODEL=/path/to/Qwen3-1.7B-Base-Char-ALIGN/checkpoint-1000 \
MODEL_NAME=Qwen3-1.7B-Base-Char-ALIGN \
bash scripts/evaluation/ppl/run.sh

# 原始 Base；Char / PGCA / CPT 同样分别替换 MODEL 和 MODEL_NAME。
CUDA_VISIBLE_DEVICES=0 \
MODEL=/path/to/Qwen3-1.7B-Base MODEL_NAME=Qwen3-1.7B-Base \
bash scripts/evaluation/ppl/run.sh
```

脚本可从任意工作目录通过绝对路径启动。默认结果位于 `outputs/evaluation/ppl/<MODEL_NAME>/v0/`，可用 `RUN_NAME=checkpoint-2000` 在版本目录之前增加实验子目录，或用 `OUT` 指定父目录。同一命令再次运行自动创建 `v1/`、`v2/`，已有报告与中断产物保留，不必手动改目录。Python 的 `--output parent/metrics.json` 同样实际写入 `parent/vN/metrics.json`，运行时打印路径并在结果 `output` 字段记录。当前不实现断点续算，中断时保留逐文档诊断，manifest 的 complete 保持 false。

可调环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `DATA_DIR` | 仓库 `data/ppl` | 扫描目录内全部 JSONL |
| `WINDOW` / `STRIDE` | `2048` / `1024` | token 滑窗长度和步长 |
| `DEVICE` | `auto` | HF 自动分配；也可 `cpu`、`cuda:0` |
| `DTYPE` | `auto` | 权重 dtype；也可 `float32`、`bfloat16`、`float16` |
| `ATTENTION` | `sdpa` | 或 `eager` |
| `LOSS_CHUNK_SIZE` | `128` | 分块 FP32 交叉熵，减少临时显存，不改变窗口 |

每次前向只有一个文档窗口；`auto` 可分配多个可见 GPU，但不是多进程数据并行。混合 CPU/GPU 或 disk offload 会报错，不隐式退化为极慢的卸载评估。纯 CPU 可运行小模型验证。不支持直接加载 LoRA adapter，需先准备正确合并的完整权重及 tokenizer。

## 固定计分协议

- `add_special_tokens=False`，不加 BOS/EOS，首 token 只作上下文；文档有 L 个 token 时恰好计 L−1 个目标。窗口间重叠的已评分 token 只作上下文，每个目标只计一次。
- 要求 `0 < STRIDE < WINDOW`，至少保留一个上下文 token；WINDOW 超过模型/tokenizer 的已知容量直接报错，不偷偷缩短。比较不同模型时手动设置相同 WINDOW、STRIDE、dtype 和 attention。
- 从 `logits[t-1]` 与 `input_ids[t]` 计算交叉熵，不传 labels、不读取 `outputs.loss`，不申请音形辅助 logits；PGCA 分支仍按 checkpoint 配置执行，`PGCA_ENABLED` 环境覆盖值也记录到 manifest。
- 交叉熵使用 FP32，NLL 用 FP64 累加；token PPL 为 `exp(ΣNLL / Σ有效目标token数)`。分领域及全语料 `overall` 都从总和计算，不对文档 PPL 或领域 PPL 求算术平均。
- `char_count=len(原始text)`：Unicode 码点，包含空白和标点；不是汉字数、字形簇数，也不是 token offsets 长度之和。`byte_count` 是原始文本 UTF-8 字节数，不含 JSON 转义或字段名。
- 同时给出 CharNLL、CharPPL、BPC、ByteNLL、BytePPL、BPB。`tokens_per_char` 使用完整编码 token 数；`scored_tokens_per_char` 使用实际参与 NLL 的目标数。
- 空文本不前向；非空但只有一个 token 的文档同样不前向，NLL=0，保留原始字符/字节计数并单独报告 no_target_documents。非空文本被 tokenizer 编成零 token、数据结构错误、非有限 NLL 或某领域完全无目标时会报错，不静默跳过。
- 指数溢出时相应 PPL 为 null，并列入 `ppl_overflow`；NLL 仍可查看，不输出非标准 JSON 的 Infinity/NaN。

Base 与 Char 的 token PPL **不能直接横比**；应使用同一原始数据及协议，同时报告字符/字节归一化指标。这里的归一化 NLL **排除了每篇首 token 的概率，但分母是全文**，是条件 NLL 的归一化，不是完整文本概率；不同 tokenizer 的首 token 范围和相同 token 窗口覆盖的文本长度仍可能不同。特别要检查 no_target_documents，不能把归一化指标宣称为完全消除了 tokenizer/上下文差异。

滑窗方法参考 [Transformers PPL 文档](https://huggingface.co/docs/transformers/perplexity)，有效目标数以本实现实际 shift 后的目标为准，不直接套用示例中统一减 batch_size 的计数方式。

## 产物与迁移差异

- `metrics.json`：分领域和 overall 结果、协议、警告、耗时。
- `metrics.documents.jsonl`：按原始顺序的逐文档 NLL、计数、ID、文本 SHA256；不复制正文。
- `metrics.manifest.json`：源文件 SHA256、checkpoint 配置/tokenizer/自定义 Python 代码哈希、加载后的 tokenizer backend 哈希、依赖版本、实际模型开关、结果哈希。大权重仅记录路径/大小/mtime，不声称完整内容哈希。

本地 checkpoint 的自定义代码通过 `trust_remote_code=True` 加载，只应评估可信模型；`local_files_only=True` 不自动下载缺失模型。整个评估不写模型或原始数据。manifest 的 complete=true 表示评估执行完毕，不代表模型质量达标。

相对 CLLM，修正了重复 offset 字符计数、无重叠滑窗计数风险及窗口自动缩短后 stride 可能越界的问题，新增全语料加权汇总和字节归一化。`tokens_per_char` 现在是真实编码密度，不再将有效目标数冒充编码 token 数。旧报告不保证数值逐位相等；做对比应使用此入口重跑所有模型。

原 `scripts/validation/validate_forward.py` 仍是迁移权重与有限 forward 诊断，不能替代该入口。

## 测试

在已安装上述依赖及 pytest 的环境执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_evaluation_ppl.py
```

覆盖短文/长文/尾窗、逐 token 计数、Unicode 字符/字节口径、忽略模型 loss、语料加权汇总、异常输入、CPU Qwen3 保存/加载到最终报告，以及 PGCA 上下文分支的 logits 计分。实际服务器权重、GPU 性能和全量 PPL 需在服务器运行后确认。
