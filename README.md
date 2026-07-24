# semantic-acoustic-codec

`semantic-acoustic-codec` 提供从语义码重建波形的 codec 组件。第一阶段只使用现有
`wmt19_tts_codec(longcat)` prepared data，不新增 TTS 合成流程；目标是在 LongCat 的语义
codebook 上训练一个可替换的 acoustic decoder，使调用方只需要持有 semantic codes 也能生成波形。

仓库边界：

- 暴露 codec backend、semantic-only support wrapper 和 acoustic decoder 训练组件。
- 支持两条 acoustic decoder 路线：RVQ、FM。
- 复用 anydataset/anytrain 的 LongCat prepared view 与 codec backend adapter。
- 不依赖 `speech-to-speech`；后续由 `speech-to-speech` 反向依赖本仓库。

设计方案见 [docs/design.md](docs/design.md)。
