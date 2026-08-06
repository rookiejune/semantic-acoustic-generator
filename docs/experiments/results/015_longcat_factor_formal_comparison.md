# 015 LongCat Factor Depth Formal Comparison

## 结论

本实验在相同 target-role split、相同动态 batch、相同优化器和相同 20k step 预算下，比较了
stage-autoregressive factor predictor 的两种实现：共享 block 的 recurrent SwiGLU 与 Qwen3
causal Transformer。两者都保持 semantic/acoustic 的逐帧严格对齐，并在一个 stage 内用同一
hidden state 独立预测 A/B 两个 factor。

固定 20k 终点的主结果为：recurrent full-N2 WER `0.005277`、UTMOS `3.03837`；Qwen
full-N2 WER `0.010554`、UTMOS `2.98752`。recurrent 的 WER 约为 Qwen 的一半，UTMOS 高
`0.05085`。同一训练过程中 recurrent 参数量少约 `10%`，端到端训练墙钟短约 `19%`，
event-based valid-frame 吞吐高约 `87%`。

因此 015 的工程结论是：factorized direct classification 可以 scale 到双码本，当前应采用
recurrent depth-AR 作为默认结构；没有证据继续为这个只有 2 个 depth token 的问题保留 Qwen
attention。这个实验不证明 FM flow 本身有效，因为正式 treatment 使用的是 `fm_mode=anchor`
的 factor classification。

## 实验口径

- revision: `73f3a96`（正式训练代码；文档收口 commit 为后续 revision）
- host/device: 145, `CUDA_VISIBLE_DEVICES=0`, RTX 4090 D 24GB
- data: LongCat target-role `train_0_10000`
- train: sample `0--9983`, 9984 条
- heldout: sample `9984--9999`, 16 条；两模型完全复用同一 manifest
- batch: dynamic 96 samples / 1152 seconds，padding ratio 相同
- precision: bf16 mixed；optimizer、learning rate、seed、数据顺序和 callback 口径相同
- checkpoint: 5k/10k/15k/20k；另有 batch96 的 1k/2k early-stop run
- speech evaluator: Anytrain Whisper `large-v3` WER + UTMOS，temperature `0`

## 结构与效率

| predictor | structure | params |
| --- | --- | ---: |
| recurrent | 2 shared pre-norm SwiGLU blocks，stage 间串行复用 hidden，A/B 并行 head | 18.2M |
| Qwen | 2-layer Qwen3 causal Transformer，8 heads，depth sequence length 2，A/B 并行 head | 20.3M |

严格同口径的 130-step batch96 probe：

| metric | recurrent | Qwen | recurrent relative |
| --- | ---: | ---: | ---: |
| step time window | 38.5 ms | 65.3 ms | `-41.0%` |
| valid frames/s | 193.97k | 114.67k | `+69.2%` |
| peak allocated | 5.15 GiB | 8.34 GiB | `-38.2%` |
| peak reserved | 5.86 GiB | 14.00 GiB | `-58.1%` |
| padding ratio | 17.67% | 17.67% | same |

正式 20k run 的 event-based末 20 窗口 valid-frame 吞吐为 recurrent `246.4k/s`、Qwen
`132.0k/s`；端到端训练墙钟分别为 `1261s` 和 `1560s`。墙钟差距小于 kernel probe
差距，说明数据加载/NAS 和 Lightning 开销仍占正式长跑的一部分。

## Latent evaluation

`generated_projected_mse` 只作为诊断，不作为语音质量 gate。数值来自相同 heldout 16：

| checkpoint | recurrent generated / stage0 MSE | Qwen generated / stage0 MSE |
| ---: | ---: | ---: |
| 1k | 1.5076 / 1.3337 | 1.5342 / 1.3495 |
| 2k | 1.4701 / 1.3203 | 1.5121 / 1.3425 |
| 5k | 1.4532 / 1.3002 | 1.5026 / 1.3353 |
| 10k | 1.4422 / 1.3010 | 1.4574 / 1.2967 |
| 15k | 1.4680 / 1.3227 | 1.4724 / 1.3128 |
| 20k | 1.4639 / 1.3168 | 1.4567 / 1.3026 |

两者的 latent 曲线接近，且偶有反转；这再次说明 projected MSE 不能替代语音评估。

## Speech evaluation

下表的 full 是自由生成的 full-N2，stage0 是只使用第一个 codebook 的生成。格式为
`WER / UTMOS`。

| checkpoint | recurrent full | Qwen full | recurrent stage0 | Qwen stage0 |
| ---: | ---: | ---: | ---: | ---: |
| 1k | 0.007916 / 2.8877 | 0.018470 / 2.7757 | 0.005277 / 3.0249 | 0.010554 / 2.9499 |
| 2k | 0.007916 / 2.8698 | 0.007916 / 2.7043 | 0.005277 / 2.8820 | 0.005277 / 2.8068 |
| 5k | 0.005277 / 3.0387 | 0.005277 / 2.9751 | 0.005277 / 2.9761 | 0.002639 / 2.9271 |
| 10k | 0.005277 / 3.0078 | 0 / 2.9340 | 0 / 2.9885 | 0 / 3.0103 |
| 15k | 0.002639 / 3.0008 | 0.013193 / 3.0434 | 0.005277 / 2.9533 | 0.007916 / 3.0262 |
| 20k | 0.005277 / 3.0384 | 0.010554 / 2.9875 | 0.005277 / 2.9989 | 0.005277 / 2.9119 |

20k 是预先注册的 formal endpoint，不按 heldout 结果 cherry-pick。checkpoint 曲线显示：

- recurrent 在 5k/10k/15k/20k 的 full UTMOS 都高于自身 stage0-only；
- Qwen 在 10k 出现 full UTMOS 低于 stage0-only，其他 formal checkpoints 才略高；
- 16 条 heldout 太小，最佳 checkpoint 的 WER/UTMOS 会有噪声，不能据此宣称 recurrent 在
  每个 checkpoint 都全面胜出；但固定 endpoint 的 recurrent 优势和吞吐优势是一致的。

batch96 的 1k/2k early-stop 结果低于 5k 以后，不应与此前 batch32/2k screening 按
optimizer step 直接比较。正式 batch96 的 20k 约等于 83 个数据 epoch，固定 20k 已明显超出
合理训练预算；后续正式训练应使用独立 dev split 的 early stopping，而不是盲目延长到 20k。

## 已否决项与后续结构变量

- `residual_retarget=false` 保持默认。retarget 虽改善 latent MSE，但 full speech UTMOS 从
  stage0-only 的 `3.2435` 降到 `1.5306`，连 residual oracle 也只有 `1.8613`。
- 不继续扩大 depth Transformer。对长度为 2 的 depth 序列，attention 和 KV cache 没有带来
  可观测质量收益，却显著增加参数、显存和 kernel 时间。
- 下一轮真正值得验证的是严格对齐的 `AlignedAnchor`：当前 local depthwise kernel=3、4 层，
  有效局部 receptive field 约 9 帧。可在不改变 frame index、不做 time-axis down/up-sampling 的
  前提下比较 kernel=5/7、轻量 dilation 或 gated temporal mixer；factor head 本身暂不再加
  跨时间 attention。

## 制品

- recurrent formal: `145:/home/zhuyin/train/semantic-acoustic-generator/015/formal-recurrent-original-train9984-20000-73f3a96-local`
- Qwen formal: `145:/home/zhuyin/train/semantic-acoustic-generator/015/formal-qwen-original-train9984-20000-73f3a96-local`
- speech summaries: `121:/home/zhuyin/train/semantic-acoustic-generator/015/speech-formal-*-heldout-target16-73f3a96`
