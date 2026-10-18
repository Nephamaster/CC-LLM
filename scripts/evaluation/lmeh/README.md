# 通用 benchmark：lm-evaluation-harness

本目录管理 MMLU、ARC（easy/challenge）、SQuAD v2、GSM8K、CMMLU、CLUE 的评估入口和自定义任务。它们不放在外部 harness 仓库，不使用 editable 安装，也不通过 PYTHONPATH 指向外部源码。

本轮只确定布局，不补齐尚未迁入的任务 YAML、utils 或启动脚本。后续任务放在 `tasks/`，配套的 `include` 模板和 `!function` Python 文件必须一起保留；固定数据集 revision、few-shot、prompt 和评分参数，不能因迁移改变口径。

环境固定为 `scripts/evaluation/lmeh/env/`。在仓库根目录创建环境并正常安装已选定版本，例如：

```bash
python3.11 -m venv scripts/evaluation/lmeh/env
scripts/evaluation/lmeh/env/bin/python -m pip install 'lm_eval[hf]'
```

正式实验需锁定并记录验证过的 lm_eval、Torch、Transformers 版本；上面的未锁版本安装不是已验证的版本锁。安装匹配服务器 GPU/CUDA 的 PyTorch，不照抄 CPU 测试环境。任务迁入后通过该环境的 `python -m lm_eval` 和绝对 `--include_path` 加载，不要求在 harness 仓库内启动。

产物统一为 `outputs/evaluation/<benchmark>/<MODEL_NAME>/`；benchmark 目录为 `mmlu`、`arc`、`squadv2`、`gsm8k`、`cmmlu`、`clue`。ARC 子集、CLUE 子任务和重复实验放在模型目录下面。目录名不等于 harness 注册的 task 名，实际任务名以任务 YAML 为准。

新 PGCA 上下文模型先验证 HF 后端，尚不能直接沿用旧 `--model vllm` 脚本。Base/Char/PGCA 的统一命名和全部输出规则见根目录 README。
