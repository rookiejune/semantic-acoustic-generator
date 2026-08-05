# 006 Cross-text screening 与 fixed eval

对应计划：[`../schedules/006_cross_text_screening_fixed_eval.md`](../schedules/006_cross_text_screening_fixed_eval.md)

## Anydataset batching 迁移验证

- anydataset commit `813b9f7` 通过标准 `batch_sampler.sampler.set_epoch()` 契约暴露
  dataset-owned shuffle epoch；semantic-acoustic-generator commit `0e1ddd9` 改用
  `MapStyleABC.dataloader(...)`，并从依赖、配置、job 和实验 wrapper 移除
  length-based batching adapter。
- 本地 semantic-acoustic-generator 验证为 `70 passed`，ruff 与 basedpyright 均通过；真实
  Lightning 集成测试完成两个无长度 dynamic-loader epoch，规划 epoch 为 `[0, 1]`。
- `145` 上使用实际 codec store 的目标测试为 `2 passed`。32 秒 hard limit 对 LongCat 和
  BiCodec 都从前 984 pairs 过滤 4 条，训练集合为 980 pairs。
- 旧 LBA timing 在 step 38--39 失败，证据保存在
  `screening/timing-fd1024-failed/`；旧 32 秒校准保存在
  `screening/calibration-lba32/`。anydataset timing 四路线均写出至少 step 50 checkpoint，
  BiCodec FM 已写出 step 300 checkpoint，未复现 `received 0 items of ancdata`。

## 20-step FLOPs calibration

运行环境为 `145` 的单卡 NVIDIA GeForce RTX 4090 D，precision 为 bf16 mixed。每条路线使用
32 秒 additive semantic-frame budget、`batch_size=8` 上限和 profiler calibration 20 steps；
实际前 20 steps 的 batch size 均为 2。

| Codec | Route | Mean FLOPs/step | Peak allocated | Mean valid frames |
| --- | --- | ---: | ---: | ---: |
| LongCat | FM | 640.003B | 6.615 GB | 467.6 |
| LongCat | RVQ | 830.267B | 7.876 GB | 467.6 |
| BiCodec | FM | 1.783B | 2.144 GB | 1386.2 |
| BiCodec | RVQ | 54.170B | 4.938 GB | 1386.2 |

证据位于：

```text
/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/cross-text-20260728/
  screening/calibration/{longcat,bicodec}-{fm,rvq}/
```

这些数值反映 anydataset additive-cost batch composition，不复用旧 LBA calibration 的静态
FLOPs 均值。完整 profiler-free timing、held-out 16-pair fixed eval 和 reference leakage 检查仍按
实验 todo 继续；完成前不据此声称 reference gain 或音色保持。
