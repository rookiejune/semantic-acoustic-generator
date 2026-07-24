# 001 Runtime Scaffold

本次先落地 semantic-only codec 的公开边界，目标是让后续 `speech-to-speech` 依赖本仓库，而不是继续持有
codec oracle 细节。

已完成：

- 建立 `semantic_acoustic_codec` package、README 和设计文档。
- 迁移/改写 `speech-to-speech` codec oracle 中的 shared semantic conditioner、DiT/FM backbone、RVQ decoder 和三类 masked loss。
- 添加 LongCat prepared code parser：按 `[frame, codebook]` 拆分 `codes[..., :1]` semantic 和
  `codes[..., 1:]` acoustic RVQ codebooks。
- 添加 `LongCatBackend` 能力适配：sample/frame rate、semantic codebook、acoustic codebook sizes、
  acoustic codes 到 features、features 到 waveform。
- 添加 `SemanticCodecSupport` runtime：`encode(audio)` 只导出 semantic codes，
  `decode(semantic_codes)` 在本仓库内预测 acoustic side channel 后重建 waveform。
- 添加最小 artifact contract：`codec.json` + `model.ckpt`，并验证同 backend 下保存/加载后输出一致。

已知阻断：

- `third_party/anydataset/src/anydataset/types/item.py` 中 `TextMeta.SPEAKER_ID` 重复定义，Python 3.12 下会使
  `import anydataset` 失败；真实 `wmt19_tts_codec(longcat)` data smoke 需要先确认是否直接修复
  `third_party`。
