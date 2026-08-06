# 016 LongCat Qwen-FiLM Temporal Anchor

## 结论

在保持 semantic/acoustic 逐帧严格对齐、码本轴继续使用 recurrent stage-AR 的前提下，
时间轴从 4 层 local depthwise conv 改为 1 层双向 full-context Qwen，并在 attention/FFN 前加入
逐帧 semantic FiLM，没有改善最终语音质量。

固定 5k endpoint 上，Qwen-FiLM full-N2 为 WER `0.007916`、UTMOS `3.00069`；015 local-4
baseline 为 WER `0.005277`、UTMOS `3.03872`。Qwen-FiLM 的 WER 多 `0.002639`，UTMOS 低
`0.03803`，未通过“WER 不恶化且 UTMOS 提升”的预注册 gate。因此默认结构保持
local-4 temporal anchor + recurrent depth-AR，不继续扩大时间轴 attention。

Qwen-FiLM 的四个 heldout factor accuracy 均高于 local-4，但 generated/stage0 projected MSE
同时更差，最终 full speech 也更差。这说明当前 factor top-1 不能代表 LongCat decoder 的感知几何；
本实验也没有证据支持“local conv 感受野过小是当前主要瓶颈”。后续若继续优化时间轴，应优先比较
保留局部归纳偏置的 dilation 或 gated temporal mixer，而不是增加 full-context Qwen 深度。

## 实验口径

- generator revision: `046180c`；launch record: `64579ac`
- host/device: 125, `CUDA_VISIBLE_DEVICES=0`, RTX 3090 24GB
- data: LongCat target-role `train_0_10000`
- train/heldout: sample `0--9983` / `9984--9999`
- batch: dynamic 96 samples / 1152 seconds
- optimization: 与 015 local baseline 相同的 seed、数据顺序、AdamW、学习率、bf16 和 clip
- budget: 5k optimizer steps；保存 1k/2k/3k/4k/5k checkpoint
- evaluator: Anytrain revision `c432d4d`，Whisper large-v3，temperature `0`，UTMOS
- primary gate: fixed heldout 16 的 full-N2 WER/UTMOS；stage0-only 和 latent 指标仅作诊断

## 结构

| component | Qwen-FiLM treatment | local baseline |
| --- | --- | --- |
| temporal anchor | 1 个双向 Qwen3 block，hidden 512，8 heads，FFN ratio 2 | 4 个 kernel-3 depthwise gated blocks |
| semantic injection | 每层 attention/FFN 前 low-rank FiLM，rank 64 | semantic projection 作为 anchor 输入 |
| depth predictor | 2 个共享 recurrent SwiGLU blocks | 相同 |
| depth causality | stage 间 AR；同 stage A/B 独立并行 head | 相同 |
| frame axis | 不改变 frame 数、顺序或 index | 相同 |
| total params | 17.8M | 18.2M |

## 效率

125/RTX 3090、相同 batch96 的 130-step probe：

| anchor | step time | valid frames/s | peak allocated / reserved |
| --- | ---: | ---: | ---: |
| Qwen-FiLM-1 | `54.8 ms` | `136.47k` | `5.15 / 5.95 GiB` |
| local-4 | `49.8 ms` | `150.18k` | `5.15 / 5.92 GiB` |

Qwen-FiLM 保留 local-4 `90.9%` 的吞吐，显存基本相同，因此效率 gate 通过。正式 5k
TensorBoard 的末 20 个窗口平均为 `163.81k valid frames/s`、padding `17.26%`；首末 train
scalar 的墙钟跨度为 `359.1s`。

## Latent evaluation

| metric | Qwen-FiLM 5k | local-4 5k | delta |
| --- | ---: | ---: | ---: |
| generated projected MSE | `1.49308` | `1.45318` | `+0.03990` |
| stage0 projected MSE | `1.31498` | `1.30018` | `+0.01480` |
| stage0 factor A accuracy | `0.16308` | `0.16041` | `+0.00267` |
| stage0 factor B accuracy | `0.14600` | `0.13516` | `+0.01084` |
| stage1 factor A accuracy | `0.07839` | `0.06937` | `+0.00902` |
| stage1 factor B accuracy | `0.06458` | `0.05610` | `+0.00848` |

分类 accuracy 的一致提升没有转化成更低的 decoder-space MSE，说明 exact factor index 命中率
没有编码不同错误 index 在 decoder 中的距离和感知代价。

## Speech evaluation

格式为 `WER / UTMOS`：

| output | Qwen-FiLM 5k | local-4 5k | result |
| --- | ---: | ---: | --- |
| full-N2 | `0.007916 / 3.00069` | `0.005277 / 3.03872` | fail |
| stage0-only | `0.002639 / 2.96899` | `0.005277 / 2.97610` | WER 较好，UTMOS 略低 |
| target reconstruction | `0 / 4.33852` | `0 / 4.33810` | evaluator sanity match |
| selected-codebook reconstruction | `0.002639 / 4.32298` | `0.002639 / 4.32335` | evaluator sanity match |

两组 reconstruction UTMOS 只差约 `0.0004`，说明 treatment/baseline 的 evaluator 与输入口径
一致。Qwen-FiLM stage0 的 WER 改善没有延续到 full-N2；加入自由生成的第二 stage 后，WER 反而
回到 `0.007916`。因此不能用 stage0 或 factor accuracy 的局部改善覆盖 full-N2 主 gate。

## 制品

- treatment: `125:/home/zhuyin/train/semantic-acoustic-generator/016/formal-qwen-film-train9984-5000-046180c-local`
- fixed speech eval: treatment 目录下 `heldout-target16-eval/anytrain-eval/summary.json`
- baseline speech eval: `121:/home/zhuyin/train/semantic-acoustic-generator/015/speech-formal-recurrent-step-00005000-heldout-target16-73f3a96/summary.json`
