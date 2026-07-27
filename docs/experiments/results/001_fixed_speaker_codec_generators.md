# 001 Fixed Speaker Codec Generator Overfit

## 范围

- 日期：2026-07-27。
- 机器：复旦 145，NVIDIA RTX 4090D；LongCat decode 在同机完成。
- 数据：Qwen fixed-speaker `vivian` 的 prepared target sample 0。
- 训练：当前 `semantic-acoustic-codec` package，8-layer decoder，seed 0，batch size 1，
  `lr=1e-3`，weight decay 0，20 optimizer steps，bf16 mixed precision。
- 入口：`scripts/train.py experiment=overfit`；overfit preset 关闭周期 checkpoint，训练结束只导出
  semantic codec artifact。

## 训练结果

| Codec | Route | Valid semantic/acoustic units | Loss first -> last | Minimum |
| --- | --- | ---: | ---: | ---: |
| LongCat | FM | 31 / 31 | flow `2.334587 -> 1.354992` | `1.265946` |
| LongCat | RVQ | 31 / 31 | total CE `9.208279 -> 0.303055` | `0.303055` |
| BiCodec | FM | 91 / 32 | flow `2.350678 -> 1.365754` | `1.200299` |
| BiCodec | RVQ | 91 / 32 | total CE `8.579102 -> 3.431641` | `3.420898` |

LongCat RVQ 的逐 codebook CE 为：

- codebook 0: `9.061774 -> 0.903858`；
- codebook 1: `9.288567 -> 0.002818`；
- codebook 2: `9.274496 -> 0.002490`。

BiCodec RVQ 暴露一个 acoustic codebook，20 steps 后 CE 约为 `3.43`，有明确优化信号但尚未完全
记忆该固定样本。FM 两条路线的 loss 均明显下降；这只是训练闭环验收，不是音质或泛化结论。

## Semantic-only Decode

四个 artifact 均通过 `scripts/eval_artifact.py` 的真实 `SemanticCodecRuntime.decode()`：

| Codec | Route | Semantic shape | Acoustic shape | Waveform | Finite | RMS |
| --- | --- | --- | --- | --- | --- | ---: |
| LongCat | FM | `[1,31,1]` | `[1,31,3]` | `[1,1,29760]`, 1.86s | yes | 0.1371 |
| LongCat | RVQ | `[1,31,1]` | `[1,31,3]` | `[1,1,29760]`, 1.86s | yes | 0.0927 |
| BiCodec | FM | `[1,91,1]` | `[1,32,1]` | `[1,1,29120]`, 1.82s | yes | 0.0731 |
| BiCodec | RVQ | `[1,91,1]` | `[1,32,1]` | `[1,1,29120]`, 1.82s | yes | 0.0805 |

All outputs use sample rate 16 kHz and had finite min/max values. The generated waveforms are only
fixed-sample contract checks; no listening-quality claim is made.

## Artifacts

Persistent remote output directories:

- `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/overfit-longcat-fm-8l-sample0`
- `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/overfit-longcat-rvq-8l-sample0`
- `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/overfit-bicodec-fm-8l-sample0`
- `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/overfit-bicodec-rvq-8l-sample0`

Each directory contains `artifact/codec.json`, `artifact/model.ckpt` and TensorBoard scalar events.
