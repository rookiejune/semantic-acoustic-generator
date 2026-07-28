# 006 Cross-Text Screening And Fixed Eval

## 目标

在 005 已验证的真实 codec view 和 `qwen_cross_text` 契约上运行 LongCat / BiCodec x FM / RVQ
四路线 screening，并用训练未见的独立文本完成至少 16 条 fixed eval。实验记录训练稳定性、吞吐、
显存、MFU、checkpoint、双路 reference 指标和失败样本，但不把 screening 当作音质或泛化结论。

## 前置条件

1. 训练数据使用已验证的两个 `train_0_1000` codec store，但只读取 source index 0..983 的 984 个
   target pairs；远端 WMT19 源 store 共 1000 条，没有可用的 offset 1000 数据。
2. 从 WMT19 source index 984..999 另行准备 16 条 target/vivian waveform 及 LongCat/BiCodec view；
   只保留 fixed eval 需要的 role/speaker，held-out 数据使用独立目录，不能被训练 dataloader 读取。
3. codec materializer 的 resume check 必须识别当前 anydataset store 的 `.ready` 和 `dataset.json`。
4. 当前 `PerformanceCallback` 在没有 `model_flops_per_step` 或 `model_flops_per_batch` 时只记录 step
   time，不会产生 MFU。screening 启动前必须提供可信 FLOPs 输入；变长 anydataset cost batch 优先使用按 batch
   计算的 provider，不用单个静态数冒充所有 batch。

## 训练矩阵

沿用四个 001 experiment composition，只增加 screening 专属 override：

- `datamodule.sample_limit=984`；
- screening 单独设置 `datamodule.batching.max_batch_seconds=32`：LongCat/BiCodec 的 pair semantic
  duration 中位数分别为 13.32/13.26 秒，p95 为 23.42/23.38 秒，最大值为 33.78/33.74 秒；
  32 秒保留 980/984（99.6%），只过滤四个包含两个最长 utterance 的 pair；
- `trainer.max_steps=-1`、`trainer.max_epochs=1`，完整遍历一次 screening 数据；
- 保留默认 batch size 上限、anydataset cost batching、bf16 和 `reference_dropout=0.5`；
- sample/checkpoint 间隔缩短到本次 step budget 内，并保留 `last` 与所有周期 checkpoint；
- performance warmup/window 缩短到本次 budget 内，但不修改生产默认 preset。

四条路线使用独立输出目录。可以并行占用四张空闲 GPU；每条路线先单卡运行，避免把 DDP 差异混入
codec/route 对比。

32 秒同时转换为 anydataset 的 additive semantic-frame batch budget，`batch_size=8` 只作为样本数上限。
四路线先用该预算跑 20-step FLOPs calibration 并记录
峰值显存；如果任一路线发生 OOM，则全矩阵统一回退到 24 秒（保留 940/984，95.5%），不按路线使用
不同数据子集。生产 datamodule 的 8 秒默认值不随 screening 修改。

## 训练验收

每条路线记录：

- finite step 数、首末/min loss、实际 optimizer steps 和消费的 target/acoustic units；
- step time window、samples/s、semantic frames/s、acoustic units/s；
- GPU peak allocated/reserved memory、硬件峰值来源、model FLOPs/step 和 MFU；
- checkpoint 路径、大小、artifact schema/backend metadata 和 resume smoke；
- 数据过滤数、decode failure、非 finite metric 及其 sample metadata；
- 固定 pair 的 with/without-reference feature error、reference gain 和所有导出音频。

## Held-Out Fixed Eval

对每个最终 artifact 使用相同的 16 条 held-out target/reference pairs 和固定 seed：

- 导出 generated with-reference、generated without-reference、target full-unit reconstruction、
  reference full-unit reconstruction；
- BiCodec 额外导出 reference-token passthrough；
- 记录 pair metadata、feature MSE、reference gain、waveform shape/duration/RMS/finite 和推理耗时；
- 汇总均值、中位数、分位数和逐样本明细，不只保留平均值；
- 用 ASR 或等价文本检查比较生成结果对 target/reference 文本的偏向，单独列出疑似 reference 文本泄漏
  的样本供人工复核。

## 完成标准

1. 四条 screening 都从空输出目录正常结束并能从 checkpoint resume。
2. 四条路线都有可解释的吞吐、显存和 MFU 记录；缺少 FLOPs 时不得把该路线标成完成。
3. 每个 artifact 完成同一组至少 16 条 held-out paired eval，所有预期输出均落盘。
4. 失败样本和 reference leakage 检查有逐条证据。
5. 结果写入 `docs/experiments/results/006_cross_text_screening_fixed_eval.md`；只有经过 fixed eval 支撑的
   结论才可进入 `docs/experiments/conclusion.md`。
