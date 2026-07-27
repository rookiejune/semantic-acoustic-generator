# 002 32-Sample Smoke

## 范围

- 日期：2026-07-27。
- 机器：复旦 144，NVIDIA RTX 4090，CUDA_VISIBLE_DEVICES=5。复旦 145 在执行时 SSH 超时，因此切到 144。
- 代码：/mnt/pami202/zhuyin/repos/semantic-acoustic-codec，本地与远端共享目录已同步关键文件。
- 数据：Qwen fixed-speaker vivian target 前 32 条 LongCat prepared samples。
- Prepared store：/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/qwen-fixed-speaker-codec-32/longcat/train。
- 输出：/tmp/semantic-acoustic-codec-smoke32-144-1785146567721/smoke32/longcat/fm。

## 代码改动

- 新增 train.fixed_batch：overfit 显式设为 true；32-sample smoke 使用 false，即使 data.lba.enabled=false 也走真实 DataModule。
- 新增 checkpoint.resume_from 并传给 trainer.fit 的 ckpt_path。
- 新增 SampleLogger：每 N step 写 TensorBoard audio，并追加 sample_metrics.json，记录 semantic-only generated audio、full-unit reconstruction、feature MSE、shape、finite 和 RMS。
- SemanticCodecModule checkpoint hook 现在保存 schema marker，删除 backend.* state，并在加载时强校验 support.* state。
- 新增 configs/experiment/smoke32.yaml，固定 32 samples / batch_size=8 / max_steps=4 / checkpoint+sample every 2 steps。

## 本地与远端校验

- 本地 /Users/zhuyin/miniconda3/envs/py312/bin/python：ruff 通过，pytest 为 29 passed。
- 远端 145 /home/zhuyin/anaconda3/envs/py312/bin/python：ruff 通过，pytest 为 29 passed。
- 144 的 py312 没有安装 ruff，因此只用于训练 smoke。

## 远端运行

第一段从空目录训练到 step 2：

- scripts/train.py experiment=smoke32 train.max_steps=2

第二段从 step 2 checkpoint resume 到 step 4：

- scripts/train.py experiment=smoke32 train.max_steps=4 checkpoint.resume_from=/tmp/semantic-acoustic-codec-smoke32-144-1785146567721/smoke32/longcat/fm/checkpoints/last.ckpt

两段运行都显式设置 LOCATION=fudan、WORKSPACE_SKIP_CONDA_ACTIVATE=1、WORKSPACE_PYTHON=/home/zhuyin/anaconda3/envs/py312/bin/python、HF_HOME=/mnt/pami202/zhuyin/huggingface、ANYTRAIN_HOME=/mnt/pami202/zhuyin/.anytrain 和 SEMANTIC_ACOUSTIC_CODEC_TRAIN_ROOT=/tmp/semantic-acoustic-codec-smoke32-144-1785146567721。

## 结果

| Check | Result |
| --- | --- |
| First run | max_steps=2 reached |
| Resume run | restored from last.ckpt, then max_steps=4 reached |
| Final global_step | 4 |
| JSON sample steps | [2, 4] |
| TensorBoard event files | 2 event files, one for each run |
| Checkpoint schema | schema_version=1, backend_state=external |
| Checkpoint state | backend_keys=0, support_keys=136 |
| Step 2 sample finite | generated yes, reconstruction yes |
| Step 4 sample finite | generated yes, reconstruction yes |

sample_metrics.json:

| Step | Batch idx | Feature MSE | Generated RMS | Reconstruction RMS | Shape |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 1 | 2.4925866 | 0.0267379 | 0.0738413 | [1, 1, 127680] |
| 4 | 3 | 2.4523644 | 0.0878881 | 0.0738413 | [1, 1, 127680] |

Artifacts and logs:

- checkpoints/step-00000002.ckpt
- checkpoints/step-00000004.ckpt
- checkpoints/last.ckpt
- sample_metrics.json
- lightning_logs/version_0/events.out.tfevents.*
- lightning_logs/version_1/events.out.tfevents.*
- artifact/codec.json
- artifact/model.ckpt

## 观察

- LongCat backend 加载很慢：每段都主要耗在 encoder/decoder 权重加载；实际 2 step 训练只需要数秒。
- Lightning 在 mid-epoch resume 时提示标准 DataLoader 不可恢复；这次 smoke 只验证 resume 机制和 artifact，不声明样本顺序严格可复现。正式长训应优先用 epoch checkpoint、LBA/resumable sampler 或记录 sampler state。
- last.ckpt 约 2.0 GiB，虽然 backend.* 已剥离，但 checkpoint 仍包含 175M support 参数和 optimizer state；导出的 runtime artifact artifact/model.ckpt 约 702 MiB。

## 结论

32-sample LongCat FM smoke 已通过。训练入口现在能区分 overfit fixed batch 和真实 DataModule；checkpoint/resume、sample metrics、TensorBoard audio event、support artifact 和 backend-free checkpoint state 均完成最小闭环。
