# 007 LongCat-FM REPA / EMA Ablation

对应计划：[`../schedules/007_fm_repa_ema_ablation.md`](../schedules/007_fm_repa_ema_ablation.md)

质量侧 screening / fixed eval 仍按 todo 推进；本节先记录填满 GPU 的短跑效率探针。

## 短跑效率探针（2026-08-01）

### 设置

- 机器：`144`，单卡 RTX 4090 24GB；A0/A1/A2 并行于 GPU 0/1/2。
- 数据：LongCat `train_0_1000` staging store（补 `lang=zh`）。
- cost batching：`datamodule.batching.enabled=true`，`max_batch_seconds=360`，`batch_size=64`。
- `callback.performance.profile_flops=true`；sample / checkpoint 关闭；`trainer.max_steps=40`。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

packing 以 A1（REPA）显存为准：`420` 在 FLOPs profiling 峰值会 OOM；`360` 三组均跑通。同一 seed / 同一 packing 下三组的 batch 组成一致。

### 结果（逐条 decode，`ec1b21a`）

输出：`$DYNAMIC_HOME/train/semantic-acoustic-codec/ablation/longcat-fm/perf-short/{baseline,repa,ema}`

| Cell | MFU avg | step avg | FLOPs/step (last) | alloc / reserved peak | batch max/avg | frames max/avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 baseline | 0.265 | 0.19s | 6.76e12 | 13.2 / 13.4 GB | 54 / 24.9 | 5334 / 4565 |
| A1 REPA | 0.087 | 1.72s | 2.04e13 | 18.2 / 22.3 GB | 54 / 24.9 | 5334 / 4565 |
| A2 EMA | 0.258 | 0.19s | 6.76e12 | 13.9 / 14.1 GB | 54 / 24.9 | 5334 / 4565 |

### 结果（等长分组 decode，`332ffc7`）

同 packing / steps；输出：`.../perf-short-batchdecode/{baseline,repa,ema}`

| Cell | MFU avg | step avg | FLOPs/step (last) | alloc / reserved peak |
| --- | ---: | ---: | ---: | ---: |
| A0 baseline | 0.272 | 0.19s | 6.76e12 | 13.2 / 13.4 GB |
| A1 REPA | 0.087 | 1.69s | 2.04e13 | 18.2 / 22.0 GB |
| A2 EMA | 0.268 | 0.19s | 6.76e12 | 13.9 / 14.1 GB |

A1 decode 分组日志（`train/repa/*`）：

| metric | avg | max | 含义 |
| --- | ---: | ---: | --- |
| `decode_groups` | 20.6 | 38 | 每 step decode 调用次数 |
| `decode_group_mean` | 1.19 | 1.71 | 平均每组样本数 |
| `decode_group_max` | 2.78 | 6 | 最大同长组 |
| `decode_singleton_fraction` | 0.87 | 1.0 | 单条组占比 |

结论：当前 additive `max_batch_seconds` packing **几乎不共享 frame length**，分组 decode 相对逐条几乎无加速，MFU 不变。要让分组生效，需要 cost/sampler 显式鼓励同长（或分桶），而不是只卡总秒数。

## Pad ratio 探针（`8151851`）

接上 `DataThroughputCallback` 后，在 `144` 用 A0、40 step 对比 cost packing 与固定 batch（`profile_flops=false`，`log_every_n_steps=1`）。

输出：`$DYNAMIC_HOME/train/semantic-acoustic-codec/ablation/longcat-fm/pad-ratio-probe/{cost360,fixed8,fixed32}`

| 设置 | pad ratio avg (min–max) | batch avg | valid frames avg | step avg |
| --- | ---: | ---: | ---: | ---: |
| cost on，`max_batch_seconds=360`，`batch_size=64` | 0.437 (0.14–0.64) | 24.9 | 4565 | 0.12s |
| cost off，`batch_size=8` | 0.423 (0.26–0.60) | 8.0 | 1500 | 0.07s |
| cost off，`batch_size=32` | 0.545 (0.46–0.66) | 31.7 | 5764 | 0.17s |

相对固定 `batch_size=32`，cost360 的 pad ratio 更好；相对小固定 batch（8）几乎打平。加性帧预算能控制总有效帧和避免超大 pad batch，但不会把 pad ratio 压到接近 0——同 batch 仍按 max length pad。

补充校准（同机、更短 probe）：

- A0 单独 `max_batch_seconds=640`：alloc/reserved ≈ 17.2 / 21.3 GB，MFU ≈ 0.41。
- A1 `max_batch_seconds=480`：alloc/reserved ≈ 18.0 / 23.5 GB；再抬到 `420` 且开 `profile_flops` 长一点会 OOM。

### 为什么 A1（REPA）墙钟慢这么多？MFU 掉很多合理吗？

墙钟变慢**合理**；把 MFU 大跌归因于“加了 WavLM”**不合理**。

口径：`MFU = model_flops / step_time / hardware_peak`。WavLM 若被完整计入分子，且自身算子效率与 DiT 相近，应同时抬高 FLOPs 与 step_time，**MFU 不应只因多了 teacher 就掉一大截**。

实测倍率（同 packing、同 batch）：

| 量 | A0 → A1 |
| --- | ---: |
| counted FLOPs/step | ~3.0×（6.8e12 → 2.0e13） |
| step_time | ~8.9×（0.19s → 1.72s） |
| MFU | ~0.33×（0.265 → 0.087） |

含义：墙钟多出来的时间里，大约只有 ~1/3 能被多出来的 counted FLOPs 解释；其余是**分子没跟上分母**（未计入/低效开销），不是“比值定义上 WavLM 必然压低 MFU”。

更可能的原因：

1. **Teacher 管线本身很重（解释墙钟，不自动解释 MFU）**  
   `WavLMTeacher` 每 step：`codec.decode` → WavLM-base → 插值对齐。可训练参数几乎不变，多出来的 counted FLOPs 主要来自这条路径。  
   decode 已改为按相同 frame length 分组 batch（pad id 超出 codebook，不能整矩形直接 decode）；长度全不同时仍退化为多次调用。

2. **MFU 下跌指向计数与效率缺口**  
   - 变长分组 / Python 侧整理仍可能带来 launch、同步、宿主侧开销：加时间，加很少或不加 FLOPs。  
   - `profile_flops=true` 时 `FlopCounterMode` 的 dispatch 开销计入 `step_time`；A1 op 更多，profiler 对分母的膨胀可能大于对分子的贡献。  
   - Torch flop counter 对部分 codec/transformers 路径可能低估真实数学量。

3. **A2（EMA）与 A0 同级**  
   EMA 不改前向图；吞吐/MFU 与 baseline 一致，慢点不在 EMA 轴。

4. **显存**  
   同 packing 下 A1 reserved ≈ 22 GB；A0/A2 ≈ 13–14 GB 是为对齐 A1 cost budget。单独抬 A0 packing 可到 ~21 GB reserved。

结论：A1 墙钟慢一个数量级，主要是在线 decode + WavLM 的实现代价。MFU 从 0.26 掉到 0.09 **不能**写成“加 WavLM 的正常比值效应”；它说明额外时间里有大量未按 FLOPs 计价的开销（或 profile 口径偏差）。要区分“真低利用率”和“profile 膨胀”，需要关 `profile_flops`、用固定/校准 `model_flops_per_batch` 再比一版 MFU。