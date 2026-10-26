# ACT 三项诊断工具

基线：CC-LLM main 工作区；用户报告古汉语评估 16 条均触及 2048 token 上限并循环生成。
这个现象不能单凭解码文本证明未输出 EOS；Manifest 中 use_cache=true 也不能证明实际逐 token cache 生效。

## 交付与检查范围

- `scripts/evaluation/diagnose.sh`：将原来的 Dev 生成脚本扩充为三项诊断入口，默认绑定第二轮
  CPT checkpoint-12500 与 ACT checkpoint-1000，使用服务器训练环境；每次新建时间戳输出目录。
- `diagnose_act.py`：全 Dev 结束符 labels、LoRA 开关 logits/实际生成、真实加载模型的全前缀与增量
  logits 对比。生成阶段挂载观测 hook，保留原始特殊 token 和 backbone/feature cache 长度。
- 读取 checkpoint 实际运行代码，报告其路径／依赖版本／dtype，不对源码或权重做哈希。
  不改训练参数、正式评估协议、权重、原数据；`temp.sh` 的既有用户修改保留。

## 已验证与未验证

- 隔离 `/tmp/cc-llm-act-check.V3Y1vX/env`：Python 3.13、torch 2.10.0+cpu、Transformers 5.12.1、
  PEFT 0.20.0、普通安装 ms-swift ce28e0288657d6472a439ea219f1bcc77310e764。
- `tests/test_act_diagnosis.py`：6 passed，覆盖 EOS 屏蔽负例、采样、非零 LoRA 小 PGCA 数值一致性、
  故意注入错误 feature cache 长度、实际 ms-swift adapter 开关生成及完整报告流程。
  完整流程测试仅替代 GPU 可用性预检查，计算实际使用 CPU 小模型。
- 未执行服务器 1.7B GPU 诊断。不得将上述测试当作用户 checkpoint 的验收。

## 下一步

服务器运行 `CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/diagnose.sh`，同步
`outputs/diagnostics/act_checks_<时间戳>/` 下 report、eos_labels、generations、selected_dev。
缓存默认比较三个有 padding 的短输入对，atol=0.05/rtol=0.01，数值超限需先核查 dtype 与误差分布。
标签失败检查编码／监督；adapter 无效检查加载；缓存不一致检查 cache 实现。
这些检查均通过后，再依据训练内任务的实际生成表现决定是否重训 SFT。
