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
```

## 训练

`scripts/train.py` 是 Hydra 训练入口。生产默认配置位于 `configs/train.yaml`，smoke/overfit 配置位于
`configs/experiment/`：

```bash
# 最小真实 DataModule smoke
python scripts/train.py experiment=smoke

# 固定 pair overfit
python scripts/train.py experiment=overfit codec=longcat route=fm
```

四条常用路线也有 job wrapper：

```text
jobs/001/01_longcat_dit.sh
jobs/001/02_longcat_rvq.sh
jobs/001/03_bicodec_dit.sh
jobs/001/04_bicodec_rvq.sh
```

训练输出目录默认由 `SEMANTIC_ACOUSTIC_CODEC_TRAIN_ROOT` 和 `output_subdir` 决定，也可以用
`output_dir=/path/to/output` 覆盖。启用 checkpoint 时，周期 checkpoint 位于
`<output>/checkpoints/`，训练结束导出的 runtime artifact 位于 `<output>/artifact/`。

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

artifact 目录只包含 `codec.json` 和 `model.ckpt`；`codec.json` 内保存 schema、generator 配置、backend
兼容性 metadata、sampling 和 feature normalization。`SemanticCodecRuntime` 负责加载 artifact、生成缺失
acoustic units 并调用 backend 解码 waveform。

## 当前状态

已完成 single-sample route smoke、32-sample LongCat FM checkpoint/resume smoke，以及一条真实
`speech-to-speech` semantic-only decode smoke。cross-text 四路线重跑、真实 BiCodec RVQ 32-slot 验证、
大规模 screening 和 held-out fixed eval 仍在 [实验 TODO](docs/experiments/todo.md) 中。
