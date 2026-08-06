# 015 LongCat Factor Depth Formal Comparison

## 问题

014 证明 stage 1 在正确 stage-0 prefix 下可达到 stage-0 量级的 factor accuracy，但自由生成时的
错误 prefix 会把 stage 1 拉回 parallel-N2 水平。015 将结构和目标分开验证：

- 结构：用共享参数的 stage-recurrent SwiGLU 替代长度仅为 2 的 Qwen causal Transformer；时间轴始终逐帧并行。
- 因果：stage 之间自回归，stage 内 A/B 共用 hidden、使用独立并行 heads，不在 A/B 之间引入伪自回归。
- 目标：原始 stage 标签和 generated-prefix residual retarget 分别 screening，不把 latent MSE 当成语音质量代理。

## 结构与效率

- anchor：local 512 hidden、4 layers、kernel 3，不改变 semantic/acoustic 的严格逐帧对齐。
- depth：512 hidden、2 个共享 recurrent SwiGLU blocks、FFN ratio 2、共享 final norm。
- 整模参数量：Qwen depth-N2 约 `20.3M` trainable；recurrent depth-N2 `18.20M`，减少约 `10%`。
- loss 直接消费 packed logits；factor top-1/detail 只在 trainer 日志步构造。
- 可信训练内环拆分 LongCat composite factor 时跳过重复的 CUDA value-range 同步；公共入口仍保留严格校验。
- factor 目标分支不再逐 step 遍历 decoder 获取 loss 不使用的 codebook。
- 正式配置关闭同步型 FLOPs timer，finite-loss check 每 100 steps，启用 frame throughput/padding 日志。

145/RTX 4090 D 的原始标签动态 batch probe：

| 模型 / batch plan | valid frames/s | padding | peak allocated / reserved |
| --- | ---: | ---: | ---: |
| recurrent, 32 samples / 384 s | `136.1k`（末窗口） | `10.1%` | 约 `3.81 / 4.23 GiB`（早期 probe） |
| recurrent, 64 / 768 s | `180.6k` | `15.0%` | 未记录 |
| recurrent, 96 / 1152 s | `192.5k` | `18.2%` | 未记录 |
| recurrent, 128 / 1536 s | `197.9k` | `18.7%` | `5.85 / 7.26 GiB` |
| Qwen, 32 / 384 s | `91.5k`（末窗口） | `10.1%` | 未记录 |
| Qwen, 96 / 1152 s | `123.3k` | `18.2%` | `8.34 / 13.99 GiB` |

正式配置选择两者都安全的 96 samples / 1152 s；相比 batch32，recurrent/Qwen 的吞吐分别提升约
`41%/35%`，Qwen reserved 仍保留超过 40% 的 24GB 显存余量。128 档对 recurrent 仅再提升约 3%，
不值得增加 Qwen OOM 风险和 padding。

## 严格数据口径

- 唯一 store 是 target-role `train_0_10000`。
- train：target-role sample `0--9983`，共 9984 条。
- heldout：target-role sample `9984--9999`，共 16 条；所有模型复用相同 manifest。
- 旧 010--014 的 target-role sample `0--15` 属于训练子集，只保留诊断意义。
- `role=source` 的 target reconstruction WER 为 `10.53`，不是同分布英文评估集，不作为泛化集。

## Screening 结论

### Residual retarget

128-sample/2k overfit 的四个动态标签 top-1 达到约
`99.23%/99.24%/99.70%/99.79%`，说明 recurrent 与在线 retarget 可优化；但严格 9984/16、2k
screening 虽把 stage0/full projected MSE 从 `1.2894` 降至 `1.1748`，语音质量却失败：

| 输出 | WER | UTMOS |
| --- | ---: | ---: |
| target reconstruction | `0` | `4.3381` |
| selected-codebook reconstruction | `0.00264` | `4.3234` |
| stage0-only | `0.00264` | `3.2435` |
| recurrent residual-retarget N2 | `0.01055` | `1.5306` |
| residual oracle | `0` | `1.8613` |

即使 residual oracle 也明显劣于 stage0-only，因此 residual retarget 目标被否决；latent MSE 不能作为本任务的主 gate。

### 原始 stage 标签

相同 9984/16、2k screening：

| 模型 | stage0/full projected MSE | stage0/full UTMOS | full WER |
| --- | ---: | ---: | ---: |
| recurrent-original | `1.2798 / 1.4347` | `3.1104 / 3.1700` | `0.00264` |
| Qwen-original | `1.2867 / 1.4437` | `3.1781 / 3.2316` | `0.00264` |

两者 full-N2 的 latent MSE 都比 stage0-only 更差，但 UTMOS 均小幅上升且 WER 不变，语音 gate 通过。
这进一步确认第二码本并非天然有害，失败来自 residual-retarget 目标。

## 正式比较

- treatment：recurrent depth-AR N2，原始 stage 标签。
- baseline：Qwen depth-AR N2，原始 stage 标签。
- 两者使用相同 seed、数据顺序、96 samples / 1152 s 动态 batch、优化器、学习率和 20k optimizer steps。
- checkpoint：5k/10k/15k/20k，先写 145 本机盘，避免 pami202 `folio_wait` 污染训练墙钟。
- 主判据：同一 Anytrain Whisper large-v3 的 WER 与 UTMOS。
- 辅判据：valid frames/s、padding、peak VRAM、参数量、各 factor accuracy、projected MSE。
- 训练串行使用同一 GPU，避免并发 NAS/I/O 影响吞吐对比；固定 heldout manifest 在训练完成后统一评估。

## 入口

- recurrent：`jobs/015/01_factor_recurrent.sh`
- Qwen baseline：`jobs/015/02_factor_qwen_baseline.sh`
- 默认输出：`015/depth-recurrent-original-2cb-9984-20000` 与
  `015/depth-qwen-original-2cb-9984-20000`
