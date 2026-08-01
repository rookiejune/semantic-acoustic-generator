# semantic-acoustic-codec

`semantic-acoustic-codec` 提供从 semantic codes 重建波形的 reference-optional codec 组件。训练主线消费
workspace 离线生成的 Qwen speaker grid codec units；`qwen_cross_text` 为每个 target 选择同 speaker、不同
utterance 和不同文本的 reference。推理可以只使用 semantic codes 和 learned null condition，也可以额外
提供 reference acoustic features。

仓库边界：

- 消费 anytrain 的 LongCat / BiCodec backend，暴露 codec-free support、runtime 和训练组件。
- 支持 FM 与 RVQ 两条 generator 路线；BiCodec RVQ 通过 temporal MTP 沿 32-slot 轴生成。
- 提供固定 pair 的 with-reference / without-reference 日志和 artifact 导出。
- 不依赖 `speech-to-speech`；下游由 `speech-to-speech` 依赖本仓库。

设计与边界见 [docs/design.md](docs/design.md) 和
[docs/speech-to-speech-integration.md](docs/speech-to-speech-integration.md)。实验状态见
[docs/experiments/todo.md](docs/experiments/todo.md) 与
[docs/experiments/results/](docs/experiments/results/)。

## 安装与环境

在本仓库目录安装基础包；训练、RVQ 和测试分别通过 optional dependency 启用：

```bash
python -m pip install -e ".[train,rvq,test]"
```

真实训练还需要 workspace 已准备好的 codec 数据，以及 `anytrain`、`anydataset` 和 workspace 的运行环境。
job wrapper 会通过 [jobs/env.sh](jobs/env.sh) 设置 `PYTHONPATH` 和训练输出根目录；使用 wrapper 前请设置
`DYNAMIC_HOME`，或直接设置 `SEMANTIC_ACOUSTIC_CODEC_TRAIN_ROOT`。

## 快速验证

不依赖 prepared data 的本地 contract smoke：

```bash
python scripts/smoke.py
```

这个命令默认使用内置 fake backend，并跳过数据 smoke。验证真实 prepared data 时显式传入数据根目录：

```bash
python scripts/smoke.py \
  --codec longcat \
  --data-source qwen_cross_text \
  --data-root /path/to/prepared/codec-grid \
  --require-data
```

测试和 lint 以 `pyproject.toml` 为准：

```bash
pytest
ruff check .
basedpyright
```

## 训练

`scripts/train.py` 是 Hydra 训练入口。`configs/backend`、`datamodule`、`model`、`loss`、
`pl_module`、`runtime` 和 `callback` 对应 `src/semantic_acoustic_codec` 的模块边界；
`configs/experiment/` 显式组合这些 preset 并持有数据范围与训练预算：

```bash
# 最小真实 DataModule smoke
python scripts/train.py experiment=smoke

# 固定 pair overfit
python scripts/train.py experiment=overfit backend=longcat model/route=fm

# 984-pair screening；FLOPs calibration 时显式追加 callback.performance.profile_flops=true
python scripts/train.py experiment=screening backend=longcat model/route=fm

# LongCat-FM REPA / EMA ablation（screening 预算）
python scripts/train.py experiment=ablation_fm_baseline
python scripts/train.py experiment=ablation_fm_repa
python scripts/train.py experiment=ablation_fm_ema
```

`pl_module.ema_decay` 非空时启用 `anytrain.lightning.EMACallback`；sample / artifact 导出使用 EMA
权重。`loss.repa_loss_weight>0` 时 train 入口构造 `WavLMTeacher`（当前仅 LongCat frame-aligned FM）。

四条正式路线分别由 `experiment=001_longcat_fm|001_longcat_rvq|001_bicodec_fm|001_bicodec_rvq`
完整组合，也有对应的 job wrapper：

```text
jobs/001/01_longcat_dit.sh
jobs/001/02_longcat_rvq.sh
jobs/001/03_bicodec_dit.sh
jobs/001/04_bicodec_rvq.sh
```

训练输出目录默认由 `SEMANTIC_ACOUSTIC_CODEC_TRAIN_ROOT` 和 `output_subdir` 决定，也可以用
`output_dir=/path/to/output` 覆盖。启用 checkpoint 时，周期 checkpoint 位于
`<output>/checkpoints/`，训练结束导出的 runtime artifact 位于 `<output>/artifact/`。

FM 正式训练的 feature normalization 会遍历有效训练 subset 计算；不会只使用固定示例或首个 batch。
需要 held-out 验证时设置 `datamodule.validation_split=<split>`，并可用
`datamodule.validation_sample_limit=<count>` 限制固定评估集。验证以固定顺序分别记录有/无 reference 的
FM feature MSE 或 RVQ code error。

变长训练通过 anydataset `MapStyleABC.dataloader(...)` 按预计算 semantic frame cost 规划 batch；
`datamodule.batching.max_batch_seconds` 转换为 additive frame budget，`batch_size` 只限制单批最大样本数。
planner 在 worker 物化 codec Tensor 前完成，不依赖已归档的 length-based batching adapter。

## Artifact 评估

使用 `scripts/eval_artifact.py` 在一个 target/reference pair 上同时评估有 reference 和无 reference 路径：

```bash
python scripts/eval_artifact.py \
  --artifact /path/to/output/artifact \
  --codec longcat \
  --data-root /path/to/prepared/codec-grid \
  --output-json /tmp/semantic-codec-eval.json \
  --without-reference-wav /tmp/without-reference.wav \
  --with-reference-wav /tmp/with-reference.wav
```

artifact 目录只包含 `codec.json` 和 `model.ckpt`；当前 schema 为 `7`，`codec.json` 内保存 generator 配置、backend
兼容性 metadata、sampling 和 feature normalization。`SemanticCodecRuntime` 负责加载 artifact、生成缺失
acoustic units 并调用 backend 解码 waveform。

`callback.performance.profile_flops=true` 会通过 anytrain 对真实 train batch 统计动态 FLOPs，用于短
calibration。该模式有 profiler 开销；生产 timing run 保持关闭，并显式传入 calibration 得到的
`callback.performance.model_flops_per_step`，不能把 profiled step time 当作生产吞吐。

frame-aligned waveform decode 会裁掉连续的右侧 padding；一个 batch 内的有效 semantic 长度必须相同，
不同长度的请求需要先分组或逐行 decode。

## 当前状态

已完成 single-sample route smoke、32-sample LongCat FM checkpoint/resume smoke，以及一条真实
`speech-to-speech` semantic-only decode smoke。cross-text 四路线重跑、真实 BiCodec RVQ 32-slot 验证、
大规模 screening 和 held-out fixed eval 仍在 [实验 TODO](docs/experiments/todo.md) 中。
