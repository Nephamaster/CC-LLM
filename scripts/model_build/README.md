# 模型构建

完整入口：

```bash
bash scripts/model_build/run.sh
```

默认读取 `models/base`、`data/model_build/hanzi` 与 `data/model_build/unihan`，依次调用 `build_vocab.sh`、`build_embedding.sh`、`build_pgca.sh`。隐藏维度从 Base 的 `config.json` 读取，PGCA 默认选择模型中间三分之一层，不绑定参数规模。

允许使用已有的 `CHAR_MODEL` 和 `PGCA_MODEL` 目录；重新构建会覆盖同名构建产物。

`MODEL_ROOT` 默认是 `models/`，同时决定 Char/PGCA 默认输出、构建资源和完成指针的位置。集中冒烟将其设为本轮输出目录中的 `models/`。

默认构建先完成 Char 改造，再加入 PGCA，因此产物名分别为 `<B>-Char`、`<B>-Char-PGCA`。`<B>-PGCA` 保留给未做 Char 改造时添加 PGCA 的命名，本次未新增该构建路线。完成指针中的模型名称取实际 PGCA 输出目录名。

训练阶段按顺序追加 `-Align`、`-CPT`、`-INS`，例如默认主线为 `<B>-Char-PGCA-Align-CPT-INS`。跳过阶段时不添加对应后缀。
