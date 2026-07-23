# TODO

本文只记录未完成事项。完成项应移动到 `docs/experiments/results/` 或由 git 历史保留。

## P0 最小闭环

- 修复或绕过 `third_party/anydataset` 的 `TextMeta.SPEAKER_ID` 重复定义问题，再运行
  `wmt19_tts_codec(codec="longcat")` 单样本 data smoke。
- 实现每条路线的 single-step training smoke：forward/backward/optimizer 和 finite loss。
- 实现 `scripts/smoke.py`：覆盖数据读取、teacher feature、两条 route 的一次前后向和 runtime decode。
- 实现正式训练入口、Hydra configs 和 jobs wrapper。

## P1 质量前验证

- 单样本 overfit：两条路线分别记录 loss、waveform 长度、finite audio。
- 32-sample smoke：确认 dataloader、checkpoint、resume、metrics 和 audio logging。
- 与 full LongCat teacher reconstruction 做固定样本 A/B：不声明质量，只记录 baseline。

## speech-to-speech 接入前

- 固定 artifact schema version。
- 在本仓库提供 runtime loader，不依赖 `workspace` 或 `speech-to-speech`。
- 在 `speech-to-speech` 增加 runtime preset 前，确认依赖方向只有 `speech-to-speech -> semantic-acoustic-codec`。
- 用 `model/acoustic=none` 跑一条 TTS generation decode smoke。
