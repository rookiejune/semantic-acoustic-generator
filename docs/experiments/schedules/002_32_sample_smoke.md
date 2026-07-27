# 002 32-Sample Smoke

## 目标

在 single-sample overfit 完成后，用 Qwen fixed-speaker vivian 的 32 条 LongCat prepared samples 验证小批量真实训练闭环：

- DataModule 在 lba.enabled=false 时仍读取 32 条样本，而不是退化成重复单样本。
- checkpoint 能在 step 2 产生，并能从 last.ckpt resume 到 step 4。
- TensorBoard audio logging、JSON sample metrics 和 support artifact 都能产生。
- Lightning checkpoint 不保存 frozen backend 权重，只保存 semantic-acoustic support state 和 optimizer state。

## 数据

- 原始数据：/mnt/pami202/zhuyin/dynamic/datasets/wmt19_qwen_tts_role_speaker/train_0_1000。
- 条件：role=target、speaker_id=vivian、前 32 条。
- LongCat prepared store：/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/qwen-fixed-speaker-codec-32/longcat/train。
- 32 条样本中多数超过默认 8 秒 LBA 上限；本 smoke 使用 data.max_seconds=8.0 和 data.overlong=truncate 保留 32 条样本。

## 配置

新增 configs/experiment/smoke32.yaml：

- codec=longcat
- route=fm
- data.source=qwen_fixed_speaker
- data.sample_limit=32
- data.batch_size=8
- data.lba.enabled=false
- train.fixed_batch=false
- train.max_steps=4
- checkpoint.every_n_train_steps=2
- sample.every_n_train_steps=2

## 执行计划

1. 本地补齐训练入口：train.fixed_batch、checkpoint.resume_from、sample callback。
2. 本地补齐 checkpoint hooks：保存时剥离 backend.*，加载时校验 support state。
3. 本地跑 ruff 和 pytest。
4. 同步到远端共享代码目录。
5. 远端先跑 train.max_steps=2，检查 checkpoint、TensorBoard event 和 sample_metrics.json。
6. 从 checkpoints/last.ckpt resume 到 train.max_steps=4，检查 global_step=4 和 step 4 sample metrics。
7. 将结果写入 docs/experiments/results/002_32_sample_smoke.md，并从 TODO 删除本项。
