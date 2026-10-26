# ACT P0 补充与 P1 来源抽查

基线：main c0a2203 工作区；用户要求复用六条 Dev 做结束位置概率审计，三条做 Cache On/Off，
三任务各抽 100 条 Train/Dev。古汉语专用脚本统一到 `scripts/evaluation/ancient/diagnostics/`。

## 已确认

- 服务器 baseline：EOS labels 1800/1800 通过；adapter 六条有明显 logits 差异且恢复一致。
  Cache BF16 带 padding 数值检查三组均超限（最大 1.5、1.5、2.125）；ACT 六条均 hit limit。
  这证明需要继续调查，尚未区分缓存数值、推理或训练根因。
- 当前 CPT tokenizer 与本地 Char tokenizer 中 `𫟦=107046`、`<|im_end|>=153995`，两者不是同一 ID；
  `<|endoftext|>=153993`。tokenizer 映射本身不证明 checkpoint 权重行与训练期间使用的词表一致。
- Dev 第 26 行（index 25）错配“谓犹应称兼慕，至卒哭。”→“太子萧统接到皇帝的圣旨后才吃了几合米做的稀饭。”
  可追溯到 Gujilab raw/translate.jsonl:1314933，反向记录 :1314934 同样错误；
  邻居 :1314931 和 :1314935 也出现祭丧议论与饮食叙事不对应。
  HistoryTrans-train.jsonl:583630 同样错误；:486442 的长段落包含相关祭丧议论及对应译文。
  因此该坏例已存在于上游，不能据此认定 final 拼接过程错位，也不能据此估计全库错误率。
- P1 600 条固定随机抽样已在本地执行，路径 `outputs/diagnostics/act_data_audit_20261026/`。
  六组各100条，600条 raw→final 配对核验一致，无抽样标点正文改写；这只证明来源一致，不证明语义正确。
  标点前置标点待审：Train 74/100、Dev 70/100；属于片段边界线索，不自动当作错标率。
- 进一步阅读样本中 Train/Dev 翻译各方向前5条，共20条；确认 Train m2c 第38311行也错配：
  输入“至元十四年世祖召见，命授后卫经历，佩戴金符。”，目标“二十六年，参议枢密院事。”
  id=a6b8da969161246319b8d32d47b0853d8c6718ade49dd9518b1eec3c4a4edf04，raw translate.jsonl:77582。
  其余样本未完成逐条语义判定；review.tsv 保持 pending，不把自动核验视作600条语义审核通过。

## 新工具与待完成

- `MODE=followup .../run.sh`：六条参考末尾（EOS 前位置）的 CPT/ACT 概率、logit、排名和 top10；
  每类第一条 ACT LoRA 开启下 Cache On/Off 各最多96 tokens，最终 generate kwargs 强制开关并观察实际 forward。
- `audit_data.py`：CPU 导出600条审查表及原始来源证据；保持 final/raw 文件不变。
- 9项本地测试通过：torch2.10 CPU、Transformers5.12.1、PEFT0.20、普通安装 ms-swift ce28e028。
  包含真实小 PGCA＋LoRA engine Cache Off/On、teacher-forcing 下标负例、完整 baseline/followup 工作流。
  服务器已有报告版本为 ms-swift4.6.0/Transformers5.19.0/torch2.10+cu126；本地测试不代替该服务器验收。
- P0真实1.7B补充报告待服务器执行；600条语义审核待继续，暂不计算总体语义错误率、不启动重训。
