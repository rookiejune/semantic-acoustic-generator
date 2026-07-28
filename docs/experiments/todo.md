# TODO

本文只记录未完成事项。完成项应移动到 `docs/experiments/results/` 或由 git 历史保留。

## Screening 与 Fixed Eval

- 完成约 1000 样本的四路线 screening，记录吞吐、显存、MFU、checkpoint、双路 reference 指标和失败样本。
- 在 held-out cross-text target/reference pairs 上完成至少 16 条 fixed eval，同时导出 with-reference、
  without-reference、full-unit reconstruction；BiCodec 额外导出 reference-token passthrough。
- 检查 reference 是否泄漏文本内容；验证完成前不把 reference gain 或音色保持写入 conclusion。
