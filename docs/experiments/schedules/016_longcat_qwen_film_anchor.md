# 016 LongCat Qwen-FiLM Temporal Anchor

## 问题

015 证明码本轴上长度仅为 2 的 Qwen causal Transformer 没有质量收益，stage-recurrent SwiGLU
更快、更省显存；但两组模型共用的时间轴 anchor 仍是 4 层 kernel-3 depthwise local conv，
有效 receptive field 只有约 9 帧。016 将结构职责拆开：

- 时间轴：full-context Qwen temporal block，一次并行输出全部 frame，不做时间自回归；
- 条件：每层在 attention norm 和 FFN norm 后加入 frame-wise FiLM，逐帧注入对齐 semantic condition；
- 码本轴：保持 015 已验证的 recurrent stage-AR；同 stage A/B 仍使用独立并行 heads；
- 时间长度：输入输出 frame 数和 frame index 完全不变，不做下采样、上采样或 duration expansion。

## Treatment

- temporal anchor: hidden 512，1 个 Qwen3 block，8 heads，FFN ratio 2，双向 full-context attention；
- FiLM: 每层独立的 low-rank MLP，rank=`max(hidden/8, 8)`，输出 attention/FFN 的 scale 和 shift；
- depth predictor: 512 hidden，2 个共享 recurrent SwiGLU blocks；
- target: 原始 LongCat N2 factor labels，`residual_retarget=false`；
- reference: `reference_dropout=1.0`，先只验证 semantic-to-acoustic 主路径。

1 层 Qwen 是参数匹配选择：它已经能提供全局时间 receptive field，同时避免把 temporal anchor
扩大成远高于 local-4 baseline 的模型。本地按正式维度统计，Qwen-FiLM anchor 为 `3.315M`
parameters，local-4 anchor 为 `3.690M`，treatment 反而少约 `10%`；远端 probe 再记录整模参数量、
显存和吞吐。

## 严格比较

- baseline: 015 recurrent-depth + local-4 anchor 的既有 5k checkpoint；
- treatment: Qwen-FiLM temporal anchor + 相同 recurrent-depth；
- data/split: train target-role `0--9983`，heldout target-role `9984--9999`；
- batch plan: 先 probe batch96 / 1152 seconds；若 OOM，按 64/768、32/384 回退并记录；
- optimization: seed、数据顺序、learning rate、bf16、gradient clip 与 015 相同；
- budget: 5k optimizer steps，保存 1k/2k/3k/4k/5k；不再重复已证明过训练的 20k；
- quality gate: Anytrain Whisper large-v3 WER + UTMOS，full-N2 同时比较 stage0-only；
- diagnostic: heldout factor accuracy、generated/stage0 projected MSE；MSE 不作为主 gate。

## Gate

1. 本地 shape/padding、full-context、backward、artifact roundtrip 和 Hydra composition 通过。
2. 145/GPU0 跑 16-sample 2-step smoke，完成 artifact export/reload 和 fixed-set decode。
3. 同卡跑 130-step batch probe，记录 valid frames/s、step time 和 peak VRAM。
4. treatment 吞吐不低于 local baseline 的 50%、显存不超过 24GB，且 2-step 无 NaN/OOM，进入 5k。
5. 5k full-N2 WER 不恶化、UTMOS 高于 015 local-5k，才保留 Qwen-FiLM；否则回到 local anchor，
   后续只做更轻量的 dilation/gated temporal mixer。

## Probe 结果

2026-08-06 在 125/RTX 3090 GPU0 做了同卡、同 batch96 / 1152 seconds、同 130-step callback
口径的严格校准：

| anchor | params | step time | valid frames/s | peak allocated / reserved |
| --- | ---: | ---: | ---: | ---: |
| Qwen-FiLM-1 | 17.8M | `54.8 ms` | `136.47k` | `5.15 / 5.95 GiB` |
| local-4 | 18.2M | `49.8 ms` | `150.18k` | `5.15 / 5.92 GiB` |

Qwen-FiLM 保留 local baseline `90.9%` 的吞吐，step time 高 `10.1%`，显存基本相同；2-step
LongCat artifact export/reload 和 heldout decode 同时通过，因此进入 5k gate。

## 正式结果

5k endpoint 上，Qwen-FiLM full-N2 为 WER `0.007916`、UTMOS `3.00069`，低于 local-4
baseline 的 `0.005277 / 3.03872`，因此语音 gate 失败。Qwen-FiLM 虽提高四个 factor
accuracy，但 generated/stage0 projected MSE 均更差；当前没有证据把主要瓶颈归因于 local
anchor 的有限 receptive field。详细结果见
[`016_longcat_qwen_film_anchor.md`](../results/016_longcat_qwen_film_anchor.md)。

## 入口

- treatment: `jobs/016/01_qwen_film_recurrent.sh`
- baseline: `jobs/015/01_factor_recurrent.sh`
