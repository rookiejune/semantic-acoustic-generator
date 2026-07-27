# TODO

本文只记录未完成事项。完成项应移动到 `docs/experiments/results/` 或由 git 历史保留。

## Cross-Text 训练验证

- 用新的 `qwen_cross_text` pair contract 重跑 LongCat / BiCodec × FM / RVQ single-pair overfit，分别记录
  with-reference 与 without-reference 的 loss、feature MSE、音频和 `reference_gain`。
- 对 BiCodec RVQ 的真实 backend 运行 32-slot generation，确认 temporal AR 的逐 slot 输出、finite decode
  和推理耗时；本地 contract test 不能替代真实 artifact 验证。

## Screening 与 Fixed Eval

- 完成约 1000 样本的四路线 screening，记录吞吐、显存、MFU、checkpoint、双路 reference 指标和失败样本。
- 在 held-out cross-text target/reference pairs 上完成至少 16 条 fixed eval，同时导出 with-reference、
  without-reference、full-unit reconstruction；BiCodec 额外导出 reference-token passthrough。
- 检查 reference 是否泄漏文本内容；验证完成前不把 reference gain 或音色保持写入 conclusion。
