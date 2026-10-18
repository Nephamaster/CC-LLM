# CSC / CGEC / 古汉语评估部署

## 用户确认的边界

- 代码与启动脚本在 `scripts/evaluation/{csc,cgec,ancient}/`。不移动或修改用户正在开发的 `scripts/train/` 与上下文音形/辅助训练代码。
- 推理以 SFT checkpoint 的训练模板为准，不提供显式模板覆盖或旧裸文本模式。
- CSC 默认只评估 CSCD-NS 和 LEMON；其他本地测试数据不删除、不重分布。迁移 CLLM 原评分逻辑，准备独立的评估输入即可。
- CGEC 是本地 **NaCGEC**，不是 NaSGEC；Dashboard 已纠正。FCGEC valid 本地评分，公开 test 仅导出；NaCGEC 使用官方两类 F0.5 及均值。
- Fùxì 模型裁判可选，不配置时对应任务保留 pending/null，不能用 Exact Match 替代。

## 重要实现结论

1. 本地 CLLM 的 CSC 推理是裸 prompt，但已提供的 CSC SFT 脚本训练使用 qwen3_nothinking。不能原样复制其推理包装。现在从模型/LoRA args.json 读取实际设置，使用 ms-swift TransformersEngine 和 get_template；指标和原预测清理逻辑迁移保留。
2. CLLM 本地 ChERRANT 缺少 data/ 字典。已从 REVISION 指定的官方 `39221934ee4ccf07801879136a75dd7e11319977` 补齐，迁移到 vendor/cherrant/assets，避免本仓库 data/、resources/ 忽略规则漏传。保留 char 模式延迟导入 LTP 的原兼容修复。
3. ChERRANT 上游 `.read().strip()` 会吞掉文件最后一条空预测的 TAB，随后丢失整个样本。改为保留字段的 splitlines，并登记空字符串→空 token 列表；空答案作为删除编辑参与评分，不删除样本。
4. FCGEC 真实 valid 有 Delete 与 Modify 重叠、Modify 内 INS tag、`INS_职业` tag、Switch 不包含尾标点等记录。按官方渲染顺序处理，不能用过强的“必须完整置换/所有操作互斥”校验拒绝它们。每个参考从原句独立展开，保留标签选项组合，不使用 eval。
5. MaxMatch 官方仍是 Python 2；仅迁移核心及 util，机械转 Python 3 并改相对 import。NaCGEC 只有平行参考，没有发布在本地的固定 word-gold M2，按官方 edit_creator 算法＋THULAC 重建，保存中间结果与参数，不能宣称未验证的排行榜逐位复现。
6. C³Bench 仓库只有数据和简短 README，完整 scorer 未提供。按论文中文指令和指标类别实现，明确记录 NER micro set-F1、typed-position punctuation F1、hierarchical-prefix retrieval、corpus SacreBLEU zh，`official_code_equivalence=false`。
7. Fùxì 不是全部用裁判：3 accuracy＋6 sentence BLEU＋2 格式规则＋10 judge。现有 5-shot 文件与 119 条 test 输入重合；保留全部样本并标记，不擅自改变测试分布，默认 zero-shot。
8. FCGEC test 导出 UUID→预测 JSON 和标准三列 CSV，不上传、不编造 gold/错误类型；当前 CodaBench 最终上传容器格式未独立核实，应提交前确认。

## 验证与产物

- 完整输入准备通过 shell 入口执行，临时输出在 `/tmp/ccllm-evaluation-smoke.ArB92C/`：CSC 26,731；FCGEC valid/test 2,000/3,000；NaCGEC test 5,869；C³Bench 50,000（10,000×5）；Fùxì 11,490。
- Fùxì 词牌规则对照本地官方实现：300 条 gold、带标题版、错误格式，共 900 次，零差异。
- 评分/断点恢复/模板/多参考等定向测试覆盖真实 ChERRANT 空输出、真实 THULAC/MaxMatch、官方 M2 发布样例 P/R/F0.5=0.8，以及实际 ms-swift CPU 小模型和 LoRA 推理。不是 mock-only 验证。
- 最终执行评估测试、迁移的 CSC 回归及既有 benchmark reader 回归，共 **37 passed**；另有一项上游 PyTorch 弃用提示。三个启动脚本 `bash -n` 和 `git diff --check` 通过，项目内 Python 缓存已清理。
- 没有运行正式模型的全量评估，没有调用任何裁判 API，没有读取 HF_TOKEN，也没有修改原始数据集。数据适配成功不等于待测模型的效果或 GPU 兼容性已通过。
- 第三方版本、许可证、来源文件 SHA256 与具体口径见 `scripts/evaluation/THIRD_PARTY.md` 和 README。先安装 evaluation/requirements.txt 到既有训练环境，不强制替换其 Torch/Transformers。

原始输出、清理输出、重试 journal、input/model/template identity 及评分中间 M2 均保留；数据、模型或推理设置改变时使用新的 OUT。评分失败/推理失败不伪装成成功或正常错误答案。
