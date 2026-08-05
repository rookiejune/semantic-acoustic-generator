# 009 LongCat FM Numerical-Stability Continuation

## 目标

从已验证 finite 的 65k LongCat-FM checkpoint 以较低学习率续跑到 100k，确认长跑不会再次出现
non-finite loss，并用同一 held-out cross-text fixed-eval 集判断是否值得继续到 200k。

## 固定配置

- `backend=longcat`、`model/route=fm`；
- 从 65k checkpoint resume；
- `pl_module.learning_rate=1e-4`；
- `datamodule.batch_size=48`；
- `datamodule.batching.max_batch_seconds=576.0`；
- 单卡 RTX 4090、bf16 mixed；
- checkpoint 和 sample 每 10k steps；
- `trainer.max_steps=100000`。

## Fixed eval

对 65k 与 100k artifact 使用 006 已物化的同一组 16 条 held-out target/reference pairs：

1. 同一 target、seed 和 sampling 配置分别运行 with-reference / without-reference；
2. 导出两路 generation 以及 target/reference full-unit reconstruction；
3. 记录 feature MSE、reference gain、waveform finite 和推理耗时；
4. 使用本地缓存的英文 CTC ASR 比较生成转写与 target/reference 文本的字符距离；
5. 用 target/reference reconstruction 校准 ASR，避免把 ASR 自身失败误判成 generator 内容错误。

held-out codec store 如果缺少 `TextMeta.LANG`，只在私有 debug/staging 复制中运行 workspace 的
manifest-only migration；不修改正式 store，不重编码 payload，并保留原 manifest 备份。

## 决策门禁

只有同时满足以下条件才继续到 200k：

- 65k→100k 全程 finite，并正常导出 100k checkpoint/artifact；
- held-out target reconstruction error 和内容保持有一致改善；
- reference condition 带来稳定收益，且没有向 reference 文本偏移的证据；
- 改善幅度足以支持继续投入同一训练配置，而不是先修改 objective 或 conditioning。
