# semantic-acoustic-codec

`semantic-acoustic-codec` 提供从语义码重建波形的 codec 组件。当前训练主线使用 Qwen speaker
grid 离线生成的 codec units；`qwen_cross_text` 为每个 target 选择同 speaker、不同 grid text row、
不同样本且不同文本的 reference。推理可以提供 reference acoustic features，也可以只使用 semantic
codes 和 learned null condition。

仓库边界：

- 消费 anytrain 的 LongCat / BiCodec backend，暴露 codec-free support、runtime 和训练组件。
- 支持 RVQ 与 FM 两条 generator 路线；BiCodec RVQ 通过 temporal MTP 沿 32-slot 轴自回归生成。
- 固定 pair 日志以相同 seed 的独立 RNG 比较有 reference 与无 reference 两条路径。
- 不依赖 `speech-to-speech`；后续由 `speech-to-speech` 反向依赖本仓库。

设计方案见 [docs/design.md](docs/design.md)。
