# CC-LLM

## 当前目录与环境约定

训练、评估的项目脚本都在本仓库管理；ms-swift 和 lm-evaluation-harness 作为普通安装的 Python 依赖，不使用 editable 安装，也不通过 PYTHONPATH 引用它们的源码仓库。

模型、词表实现仍属于 `src/`，数据工厂仍属于 `scripts/data_factory/`；它们不是需要迁入训练目录的运行入口。

```text
scripts/
  train/
    ms_swift/
      env/                       # 该训练后端的环境，不提交
      requirements.txt
      cpt/                       # Phase1 / Phase2、插件、数据准备
  evaluation/
    lmeh/                        # MMLU / ARC / SQuAD v2 / GSM8K / CMMLU / CLUE
      env/
    ppl/                         # Base / Char / 语义对齐模型的 PPL
    csc/                         # CSCD-NS / LEMON
      env/
    cgec/                        # FCGEC / NaCGEC
      env/
    ancient/                     # C³Bench / Fùxì
      env/
models/                          # 未训练的 Base、Char、PGCA 输入/构建产物
data/                            # 原始数据、数据工厂与预处理缓存
outputs/
  train/<模型名>/                # 训练 checkpoint、日志、阶段指针
  evaluation/<评估项>/<模型名>/   # 预测、分数、评估日志和必要中间文件
```

不同 checkpoint、seed、split 或 few-shot 的结果可进一步放在模型目录的子目录中，不改变模型名。构建过程的词表/索引校验报告可随构建模型保存在 `models/<模型名>/reports/`，不与 benchmark/PPL 结果混用。

`outputs/`、任意层级的 `env/`、`.venv/` 和 Python 缓存已忽略；脚本、任务 YAML、模板和辅助 Python 文件应正常纳入版本管理。旧目录或已有模型/语料不会自动改名、删除或搬迁。

## 模型命名

设原始 Base 模型名为 `B`，例如 `Qwen3-1.7B-Base`。三种基础模型分别为：

| 模型 | 名称 |
| --- | --- |
| 原始模型 | `B` |
| 单汉字模型 | `B-Char` |
| 单汉字＋完整 PGCA 分支模型 | `B-PGCA` |

PGCA 已包含字符级 tokenizer 与完整音形分支，不再使用 `B-Char-PGCA`、`PGCA-CONTEXT` 或把特征版本写进模型名称。特征版本、结构开关和哈希记录在 config/manifest。

令 `M` 为上述任一种模型名：

| 处理阶段 | 模型名称 |
| --- | --- |
| Phase1 语义对齐 | `M-ALIGN` |
| Phase2 继续预训练 | `M-CPT`，不叠加 ALIGN |
| CSC，含/不含 CPT | `M-CPT-CSC` / `M-CSC` |
| CGEC，含/不含 CPT | `M-CPT-CGEC` / `M-CGEC` |
| 古汉语训练，含/不含 CPT | `M-CPT-ACT` / `M-ACT` |

例：`outputs/train/Qwen3-1.7B-Base-PGCA-CPT/` 下保留 Phase2-A 的 `adaptation/`、Phase2-B 的 `joint/`，二者的 checkpoint/logs 都属于该模型目录。`next_stage_model.txt` 指向选出的真实 checkpoint；模型输出根目录不等于可直接加载的权重目录。

## 入口

- 模型构建：`scripts/build_vocab.sh`、`scripts/build_embedding.sh`、`scripts/build_pgca.sh`。完整 PGCA 构建入口启用上下文编码器、辅助头和零输出初始化；手工调用低层构建 CLI 时仍需显式指定相应开关。
- 训练：[scripts/train/ms_swift/cpt/README.md](scripts/train/ms_swift/cpt/README.md)。当前入口实现完整 PGCA 方案的 Phase1/Phase2；命名规范并不代表 Base/Char 对照组和全部下游训练脚本已经齐备。
- 下游评估：[scripts/evaluation/README.md](scripts/evaluation/README.md)。`MODEL` 是真实 checkpoint 路径，`MODEL_NAME` 是用于结果目录的完整模型名；LoRA 评估填 adapter 对应的模型名，而非基座名。
- 通用 benchmark：[scripts/evaluation/lmeh/README.md](scripts/evaluation/lmeh/README.md)。六类任务归入 lmeh；本轮不补齐任务 YAML/脚本。
- PPL：[scripts/evaluation/ppl/README.md](scripts/evaluation/ppl/README.md)。归入评估，不把 Phase0 forward 诊断当作完整 PPL 实现。

训练和下游评估脚本自行定位仓库根目录，默认调用所属子目录的 `env/bin/python`。训练用同一个 Python 执行 `swift.cli.main` 和数据准备，保留框架的 torchrun 分发，不再依赖 PATH 中另一个环境的 `swift`。可显式使用 `PYTHON_CMD`（训练）或 `PYTHON`（CSC/CGEC/古汉语评估）指定另一个已配置环境。PPL 入口独立于评估 common.sh，直接使用运行前激活环境的 `python`，不要求单独建立 env，也不读取 `PYTHON` 变量。

## 本轮审查后仍需注意

- 普通安装不等于版本随意替换。现有训练插件针对已验证的 ms-swift 4.7 开发快照，requirements 固定普通安装来源；lmeh 的正式版本、任务 revision、few-shot 和评分配置还需在任务迁入时锁定。
- 当前 CSC/CGEC/古汉语推理仍使用 ms-swift 的训练模板接口，因此各自评估环境也需要安装 ms-swift；不能只安装评分依赖。通用 benchmark/PPL 不走这个要求 SFT args.json 的入口。
- 旧 checkpoint 的 `args.json` 可能保存旧插件绝对路径。代码移动后，先核对依赖，再使用正确 checkpoint；本轮不修改已有 checkpoint 元数据来伪装兼容。
- 当前推理缓存没有把所有框架版本纳入复用身份；切换环境/升级依赖时用新的 `RUN_NAME/OUT`，不能把旧预测当作新环境重跑结果。
- Data Factory 的路径参与配置/运行身份。Phase2 tokenizer 默认改为共享 Char tokenizer，但旧 run 和 manifest 保留原始身份；不要仅为目录改名就重跑整条数据链路或覆盖旧产物。
- `scripts/build_embedding.sh` 当前使用 `all`，默认会执行语义权重迁移；若词表阶段已经迁移，应按 embedding 文档显式使用 `--skip-migrate`。不要对训练后的权重误跑初始化流程；本轮未改这项构建行为。
- 数据资源及 Base 权重仍有原服务器路径作为默认输入，跨机器需显式传入正确路径。CPT 默认要求足额有效监督 tokens；准备中断或预算不足时可能保留 raw 缓存，改变预算后仍需显式处理缓存，自动恢复逻辑本轮不扩展。
- Base 与 Char 的 tokenizer 不同，per-token PPL 不宜直接横比；需同一原始文本协议，并报告字节/字符归一化 NLL。旧 Phase0 诊断有截断和迁移检查，不能替代此评估。
- 新上下文 PGCA 尚未实现 vLLM 对应缓存/后端；先使用并验证 HF 路径。FCGEC test 无公开 gold、Fùxì 模型裁判未配置时只能保留 pending，目录调整不会使这些结果自动完整。
