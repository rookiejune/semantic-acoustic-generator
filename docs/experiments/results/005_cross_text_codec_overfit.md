# 005 Cross-Text Codec Overfit

## 范围

- 日期：2026-07-28。
- 机器：复旦 145，NVIDIA RTX 4090 D；GPU6 / GPU7。
- Python：`/home/zhuyin/anaconda3/envs/py312/bin/python`。
- 隔离代码快照：`/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/cross-text-20260728/code`。
- 输出根：`/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/cross-text-20260728`。

共享 Git 工作树包含既有未提交改动，本实验通过 rsync 快照运行，没有修改共享工作树。

## Codec View

waveform grid 包含 1000 个 source、source/target 两种 role 和 vivian/ryan 两个 speaker，共 4000
个 flat samples。完整物化结果如下：

| Codec | Samples | 耗时 | Target column | Cross-text pairs |
| --- | ---: | ---: | ---: | ---: |
| LongCat | 4000 | 239.23s | 1000 | 1000 |
| BiCodec | 4000 | 238.86s | 1000 | 1000 |

两个 store 都有 `.ready`、`dataset.json`、`samples.parquet` 和 4000 行
`role_speaker_manifest.jsonl`。当前 `QwenCodecPairDataset` 加载首尾 pair 成功：

- LongCat target acoustic shape 分别为 `[31, 3]`、`[256, 3]`，与 semantic frame 对齐。
- BiCodec target acoustic shape 均为 `[32, 1]`，semantic shape 分别为 `[91, 1]`、`[767, 1]`。
- 首条 target/reference text index 为 1/3，末条为 1999/1；两端都满足同 speaker、不同 text
  index、utterance id 和文本。

## 训练

四条路线均使用 8-layer decoder、condition dim 1024、batch size 1、bf16、20 optimizer steps 和
`reference_dropout=0.5`。TensorBoard 中每条路线都有 20 个 finite loss event：

| Codec / route | Step 0 loss | Step 19 loss | Min loss | 优化信号 |
| --- | ---: | ---: | ---: | --- |
| LongCat FM | 2.332620 | 2.059648 | 2.017854 | yes |
| LongCat RVQ | 9.176748 | 2.521821 | 2.151000 | yes |
| BiCodec FM | 2.430224 | 1.119744 | 1.119744 | yes |
| BiCodec RVQ | 8.399414 | 0.000027 | 0.000027 | yes |

四个训练进程都因达到 `max_steps=20` 正常退出，并分别导出 TensorBoard event、
`sample_metrics.json` 和 schema 7 artifact。FM artifact 包含数据集级 feature mean/std；RVQ artifact
不携带无用的 feature stats。

## Paired Decode

第 20 step 使用同一 seed 分别生成 with-reference 和 without-reference。固定 target 是
`train-00000000-target-00000001-vivian`，reference 是
`train-00000001-target-00000003-vivian`；二者 speaker 相同、文本不同。

| Codec / route | MSE without ref | MSE with ref | Reference gain | Generated RMS without / with |
| --- | ---: | ---: | ---: | ---: |
| LongCat FM | 2.203818 | 2.203818 | 0.000000 | 0.126727 / 0.126729 |
| LongCat RVQ | 2.047119 | 2.148841 | -0.101722 | 0.103317 / 0.108857 |
| BiCodec FM | 0.00145077 | 0.00145066 | 0.000000114 | 0.041553 / 0.041547 |
| BiCodec RVQ | 0.000000 | 0.000000 | 0.000000 | 0.074080 / 0.074081 |

所有 generated、target full-unit reconstruction 和 reference full-unit reconstruction waveform
均为 finite。LongCat target 输出为 `[1, 1, 29760]`（1.86s），BiCodec 为
`[1, 1, 29120]`（1.82s）；BiCodec reference-token passthrough 也为 finite。落盘 artifact 重新加载后
再次得到相同 shape、MSE 和 RMS，并在 `eval/<codec>-<route>/` 导出 JSON 与 WAV。

## BiCodec RVQ 32-Slot

对落盘 BiCodec RVQ artifact 单独注册只计数的 temporal forward hook，真实生成结果为：

| 字段 | 结果 |
| --- | --- |
| Temporal calls | 32 |
| Acoustic codes | `[1, 32, 1]` |
| Acoustic features | `[1, 32, 128]` |
| Generated code range | 243..4070（codebook size 4096） |
| Generation | 0.428s |
| Decode | 0.549s |
| Waveform | `[1, 1, 29120]`, finite, RMS 0.0740804 |

这验证了单 codebook BiCodec RVQ 沿固定 acoustic slot 轴执行 32 次 temporal AR，而不是构造未使用的
intra-frame MTP 路径。

## Artifact 契约

四个 artifact 都是 schema 7，并记录了严格 backend metadata：

- LongCat：`frame_aligned`、16 kHz、16.6667 frame/s、3 x 8100 acoustic codebooks。
- BiCodec：`fixed_length`、16 kHz、50 semantic frame/s、32 acoustic slots、1 x 4096 codebook。

## 结论边界

当前 `qwen_cross_text` 数据契约已经完成真实 LongCat / BiCodec 物化、四路线 single-pair overfit、
artifact reload/decode 和 BiCodec 32-slot generation 闭环。四条路线都有 finite loss 和优化信号。

本实验不支持 reference 有效、音色保持、可懂度、音质或泛化结论。单样本 reference gain 为零、近零或
负值；BiCodec RVQ 的零 MSE 只说明它记住了固定 target。相关判断仍需 1000-sample screening、独立
held-out fixed eval 和 reference 文本泄漏检查。
