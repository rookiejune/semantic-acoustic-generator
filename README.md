# semantic-acoustic-generator

`semantic-acoustic-generator` 是 reference-optional 的 semantic-to-acoustic 生成扩展：输入 semantic
codes 和可选 reference acoustic features，生成外部 codec backend 解码所需的 acoustic features 或 codes。
codec 的编码、解码、码本结构和 backend 实现不属于本仓库；主力实现由 AnyCodec 持有，当前代码通过
`anytrain.codec.SemanticAcousticCodec` 契约接入 frame-aligned LongCat backend。

训练主线消费 workspace 离线生成的 Qwen speaker grid codec units；`dataset=qwen` 配合
`pairing=cross_text` 为每个 target 选择同 speaker、不同 utterance 和不同文本的 reference。推理可以只使用 semantic codes 和 learned null
condition，也可以额外提供 reference acoustic features。

仓库边界：

- 不实现或封装新的 codec；只消费外部 backend 契约，并提供薄 runtime composition。
- 支持 FM 与 RVQ 两条 frame-aligned generator 路线；BiCodec 的 semantic/global layout 不属于本工程。
- 持有 generator 的数据适配、训练、with-reference / without-reference 评估和 artifact 导出。
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
`DYNAMIC_HOME`，或直接设置 `SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT`。Python import、环境变量、job 标识和
新 artifact 均使用 generator 命名；旧 schema-7 artifact 和 checkpoint metadata 仅通过显式 reader 兼容。

## 快速验证

不依赖 prepared data 的本地 contract smoke：

```bash
python scripts/smoke.py
```

这个命令默认使用内置 fake backend，并跳过数据 smoke。验证真实 prepared data 时显式传入数据根目录：

```bash
python scripts/smoke.py \
  --codec longcat \
  --data-pairing cross_text \
  --data-root /path/to/prepared/codec-grid \
  --require-data
```

测试和 lint 以 `pyproject.toml` 为准：

```bash
python -m pytest
ruff check .
basedpyright
```

## 训练

`scripts/train.py` 是薄 Hydra 入口，只负责配置组合并委托给
`semantic_acoustic_generator.training.run()`。实际训练组装位于
`semantic_acoustic_generator.training`：调用方和测试可以用 `build_session()` 构造可检查、可复用的
`TrainingSession`。`configs/backend`、`datamodule`、`model`、`loss`、`pl_module`、`runtime` 和
`callback` 对应 `src/semantic_acoustic_generator` 的模块边界；`configs/experiment/` 显式组合这些 preset
并持有数据范围与训练预算：

```bash
# 最小真实 DataModule smoke
python scripts/train.py experiment=smoke

# 固定 pair overfit
python scripts/train.py experiment=overfit backend=longcat model/route=fm

# 984-pair screening FLOPs calibration
python scripts/train.py experiment=screening backend=longcat model/route=fm \
  callback.performance.enabled=true callback.performance.profile_flops=true

# LongCat-FM REPA / EMA ablation（screening 预算）
python scripts/train.py experiment=ablation_fm_baseline
python scripts/train.py experiment=ablation_fm_repa
python scripts/train.py experiment=ablation_fm_ema
```

`pl_module.ema_decay` 非空时启用 `anytrain.lightning.EMACallback`；sample / artifact 导出使用 EMA
权重。`loss.repa_loss_weight>0` 时训练服务构造 `WavLMTeacher`（当前仅 LongCat frame-aligned FM）。
`callback.debug.enabled=true` 时启用 `anytrain.lightning.DebugCallback`，逐步检查 loss、参数和梯度
中的 NaN / Inf；正式训练默认关闭，smoke / overfit 等诊断实验显式开启。

LongCat-FM 可以显式启用第一 acoustic codebook 的低维 feature adapter：

```bash
python scripts/train.py experiment=overfit \
  model.feature_adapter=longcat_first_codebook \
  output_subdir=overfit/longcat/fm-8l-first-codebook
```

该 adapter 把每帧第一个 `8100=90×90` composite code 拆成两个冻结的 8-D codebook embedding，FM
因此预测 16-D target；waveform decode 前再通过 LongCat stage-0 的 `out_proj_a/b` 还原为 1024-D
decoder 输入。默认推理直接解码连续输出；oracle 和 fixed-set evaluator 会额外导出两个 factor 分别按
cosine nearest-codebook snapping 的对照。默认配置为
`model.feature_adapter=none`，且 adapter 目前只允许用于 FM；选择会写入 artifact，加载原始 LongCat
backend 时由 `GeneratorRuntime` 自动恢复。

在第一码本 adapter 上还可以显式选择严格逐帧对齐的 deterministic anchor，或 anchor 加 residual FM：

```bash
# 单次前向、有限局部上下文的 aligned anchor
python scripts/train.py experiment=010_longcat_anchor_overfit

# 仅在 anchor 可懂后使用；FM 拟合 stop-gradient anchor 未覆盖的 residual
python scripts/train.py experiment=010_longcat_anchor_overfit \
  model.decoder.fm_mode=residual
```

`model.decoder.fm_mode=flow` 是默认旧路径；`anchor|residual` 必须与
`model.feature_adapter=longcat_first_codebook` 一起使用。anchor loss 组合 normalized feature MSE、raw
feature cosine 和两个 90-way factor cosine-CE；reference 在 010 fixed-speaker overfit 中固定 dropout 为 1。

010 的 representation oracle、整段 artifact raw/snap 评估和 Anytrain Whisper/UTMOS 汇总入口分别为：

```text
jobs/010/01_representation_oracle.sh
jobs/010/02_anchor_overfit.sh
jobs/010/03_eval_artifact_set.sh
```

两条正式路线分别由 `experiment=001_longcat_fm|001_longcat_rvq` 完整组合，也有对应的 job wrapper：

```text
jobs/001/01_longcat_dit.sh
jobs/001/02_longcat_rvq.sh
```

训练输出目录默认由 `SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT` 和 `output_subdir` 决定，也可以用
`output_dir=/path/to/output` 覆盖。启用 checkpoint 时，周期 checkpoint 位于
`<output>/checkpoints/`，训练结束导出的 runtime artifact 位于 `<output>/artifact/`。

FM 正式训练的 feature normalization 会遍历有效训练 subset 计算；不会只使用固定示例或首个 batch。
需要 held-out 验证时设置 `datamodule.validation_split=<split>`，并可用
`datamodule.validation_sample_limit=<count>` 限制固定评估集。验证以固定顺序分别记录有/无 reference 的
FM feature MSE 或 RVQ code error。

变长训练通过 anydataset `MapStyleABC.dataloader(...)`，按 manifest audio duration 推导的近似
semantic frame cost 规划 batch；`datamodule.batching.max_batch_seconds` 转换为 additive proxy-frame
budget，`batch_size` 只限制单批最大样本数。planner 在 worker 物化 codec Tensor 前完成，不依赖已归档的
length-based batching adapter，也不会为了规划反序列化 codec Tensor。

## Artifact 评估

使用 `scripts/eval_artifact.py` 在一个 target/reference pair 上同时评估有 reference 和无 reference 路径：

```bash
python scripts/eval_artifact.py \
  --artifact /path/to/output/artifact \
  --codec longcat \
  --data-root /path/to/prepared/codec-grid \
  --output-json /tmp/semantic-acoustic-generator-eval.json \
  --without-reference-wav /tmp/without-reference.wav \
  --with-reference-wav /tmp/with-reference.wav
```

artifact 目录只包含 `generator.json` 和 `model.ckpt`；当前 schema 为 `8`。manifest 只保存 generator
配置、backend 兼容性 metadata、sampling 和 feature normalization，不包含 codec 实例或 backend decoder。
`GeneratorRuntime` 负责加载 generator artifact、生成缺失 acoustic units 并调用外部 backend 解码 waveform。
导出配置只取自 support 的构造配置；字段缺失、类型不精确或包含非有限数值时会直接拒绝，不补默认值，也不
把字符串或 bool 强转为数值。兼容例外是早期 schema-8 artifact 缺少后加的 `feature_adapter` 和
anchor 配置时，分别按 `none` 与 `fm_mode=flow` 的默认值读取；旧 schema-7 `codec.json` 仍可由显式
legacy reader 加载。

同时设置 `callback.performance.enabled=true callback.performance.profile_flops=true` 会通过 anytrain
对真实 train batch 统计动态 FLOPs，用于短 calibration。该模式有 profiler 开销；生产 timing run
保持 profiling 关闭，并显式传入 calibration 得到的
`callback.performance.enabled=true callback.performance.model_flops_per_step=<value>`，不能把 profiled
step time 当作生产吞吐。

frame-aligned waveform decode 会裁掉连续的右侧 padding；一个 batch 内的有效 semantic 长度必须相同，
不同长度的请求需要先分组或逐行 decode。

## 当前状态

已完成 single-sample route smoke、32-sample LongCat FM checkpoint/resume smoke、一条真实
`speech-to-speech` semantic-only decode smoke，以及 LongCat-FM 100k 数值稳定性门禁和 16-pair held-out
fixed eval。当前配置不继续到 200k；结论见
[009 result](docs/experiments/results/009_longcat_fm_numerical_stability.md)。006/007 fixed-eval 人工复核等
未完成事项仍以 [实验 TODO](docs/experiments/todo.md) 为准。
