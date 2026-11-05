# CPT 单进程编码修复

工作区：`/home/nepham/Lab/CC-LLM`，分支 `master`，修改前基线 `51c9341`。该基线工作区干净，未切换分支或提交更改。

## 依据与决定

服务器对照记录位于 `outputs/encoding-check-z2fh1d06/` 与 `outputs/encoding-check-jsuvowkz/`。前者在全量输入加载后的首批编码中以 SIGSEGV 退出，后者以主进程完成同等规模首批编码和导出；两者 OOM kill 增量均为 0。具体 native 缺陷未定位。

仅在 `scripts/train/cpt/plugin.py` 的 `EncodePreprocessor.__call__` 入口将 `num_proc=1` 转为 `None`。数据读取/字段转换的进程设置、batch、模板、标签和 A/B 切分逻辑保留原值；显式大于 1 的编码并发保持原值。包装函数通过闭包保存被包装调用，重复导入不会形成全局引用递归。对应说明更新在 CPT README。

## 验证与剩余工作

- 本地 Python 3.14.6 通过隔离模块导入了修改后的完整插件，验证三个 CPT 阶段、默认单进程、显式 None/0/4、其他参数与返回值传递、原始读取器不受影响、异常透传和重复注册。
- Python 语法及 `git diff --check` 通过。执行使用 `-B` 和 `PYTHONDONTWRITEBYTECODE=1`，未增加测试目录或项目字节码缓存。
- 本地没有 `.venv/train`，未执行真实 ms-swift、Arrow 或 GPU 实验。先前真实服务器首批对照已通过，不等于修改后正式入口的完整语料验收。
- 下一步：同步插件，保持 Phase2 `DATASET_NUM_PROC=1`，完成完整 Phase2 缓存导出后再衔接 A/B 训练。无需重训 Phase1。
