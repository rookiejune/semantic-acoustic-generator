# 001 Runtime Scaffold

本次先落地 semantic-only codec 的公开边界，目标是让后续 `speech-to-speech` 依赖本仓库，而不是继续持有
codec oracle 细节。

已完成：

- 建立 `semantic_acoustic_codec` package、README 和设计文档。
- 迁移/改写 `speech-to-speech` codec oracle 中的 shared semantic conditioner、DiT/FM backbone、RVQ decoder 和三类 masked loss。
- 添加 LongCat prepared code parser：按 `[frame, codebook]` 拆分 `codes[..., :1]` semantic 和
  `codes[..., 1:]` acoustic RVQ codebooks。
- 训练和 runtime 直接消费 anytrain 的 `SemanticAcousticCodec` backend；LongCat 本地代码只保留 prepared
  `[frame, codebook]` parser 和 batch helper。
- 训练入口新增 `codec` 配置，通过 `anytrain.codec.load_semantic_acoustic(codec)` 加载 backend。
- 补齐 001 实验矩阵 job wrapper：LongCat/BiCodec × DiT/RVQ；`qwen_fixed_speaker`
  数据源现在通过 workspace prepared codec store 提供 backend-native structured units。
- 修正 `configs/experiment/smoke.yaml` 的 Hydra package，使 smoke 覆盖能真正作用到顶层训练配置。
- 添加 `SemanticCodecSupport` runtime：`encode(audio)` 只导出 semantic codes，
  `decode(semantic_codes)` 在本仓库内预测 acoustic side channel 后重建 waveform。
- 添加最小 artifact contract：`codec.json` + `model.ckpt`，并验证同 backend 下保存/加载后输出一致。

## 复旦 145 smoke

- 环境：145，`/home/zhuyin/anaconda3/envs/py312/bin/python`，代码位于
  `/mnt/pami202/zhuyin/repos/semantic-acoustic-codec`。
- 静态和单测：`ruff check` 通过；`tests/test_model_contracts.py`、`tests/test_training.py`、
  `tests/test_fixed_layout.py`、`tests/test_structured_data.py`、`tests/test_data.py`、
  `tests/test_qwen_fixed_source.py` 通过。
- LongCat prepared data smoke：`wmt19_tts_codec(longcat)`，`root=/mnt/pami202/zhuyin/datasets/wmt19_tts`，
  `split=train`，`index=0`，semantic shape `(1, 36, 1)`，acoustic shape `(1, 36, 3)`。
- LongCat FM smoke：GPU0，`experiment=smoke`，真实 prepared data，2 steps 通过，
  `train/flow_loss≈2.150`；输出目录
  `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/fdu-longcat-smoke`。
- LongCat RVQ smoke：145 当时 GPU 被其他任务占用，默认 8-layer RVQ 在 GPU0 OOM；改用 CPU +
  tiny decoder（`condition_dim=64`、`decoder.hidden_dim=64`、`decoder.layers=1`、`decoder.heads=2`）
  跑真实 prepared data 1 step 通过，`train/rvq_loss≈9.165`；输出目录
  `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/fdu-longcat-rvq-cpu-smoke`。
- Qwen fixed-speaker LongCat prepared store：使用 workspace 离线入口从
  `wmt19_qwen_tts_role_speaker/train_0_1000` 选择 `role=target`、`speaker_id=vivian`，写出
  `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/qwen-fixed-speaker-codec-tiny/longcat/train`。
- Qwen fixed-speaker LongCat FM smoke：CPU tiny decoder，`data.source=qwen_fixed_speaker`，
  真实 backend + prepared structured sample 1 step 通过，`train/flow_loss≈2.340`；输出目录
  `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/fdu-qwen-fixed-longcat-fm-cpu-smoke`。
- BiCodec HF 权重：复旦 145 上已有 `/mnt/pami202/zhuyin/huggingface/bicodec`，可由
  anytrain 的 `SparkAudio/Spark-TTS-0.5B` snapshot 逻辑校验；缺口不是权重，而是可 import 的
  Spark-TTS Python 源码和 `einx` 依赖。
- BiCodec backend load：补入 `third_party/Spark-TTS` 后，`load_semantic_acoustic("bicodec", device="cpu")`
  通过；layout 为 `AcousticLayout.FIXED_LENGTH`，semantic codebook `(8192,)`，acoustic codebook
  `(4096,)`，frame rate `50.0`。
- Qwen fixed-speaker BiCodec prepared store：GPU0，1 条样本离线编码通过，写出
  `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/qwen-fixed-speaker-bicodec-tiny/bicodec/train`。
- Qwen fixed-speaker BiCodec FM smoke：CPU tiny decoder，1 step 通过，`train/flow_loss≈2.463`；输出目录
  `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/fdu-qwen-fixed-bicodec-fm-cpu-smoke`。
- Qwen fixed-speaker BiCodec RVQ smoke：CPU tiny decoder，fixed-length acoustic layout 分支 1 step 通过，
  `train/rvq_loss≈8.481`；输出目录
  `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/fdu-qwen-fixed-bicodec-rvq-cpu-smoke`。
- Qwen fixed-speaker LongCat RVQ 默认 8-layer smoke：GPU0，`jobs/001/02_longcat_rvq.sh`，
  `decoder.layers=8`、`trainer.accelerator=gpu`、`train.max_steps=1`，1 step 通过，
  `train/rvq_loss≈9.208`；输出目录
  `/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/fdu-qwen-fixed-longcat-rvq-8l-gpu-smoke`。

## 修复记录

- 修复 `third_party/anytrain` LongCat backend：真实 LongCat encoder 的 `hop_length` 是
  `np.int64`，现在 `semantic_frame_rate` 接受 `numbers.Integral` 并转成 Python `int`；
  本地和 145 上 `tests/test_longcat_codec.py` 均通过。
- 修复本仓库 GPU feature masking：backend acoustic features 在 CUDA、batch mask 在 CPU 时，mask 先移动到
  feature device 再 `masked_fill`。
- 修复 `single_batch_loader` 的运行时 `Tensor` 引用错误。
- `SemanticCodecModule` 现在显式 freeze/eval backend；最终 Lightning summary 中 backend 为 eval /
  non-trainable，只有 support 参与优化。

## 001 fixed-speaker overfit

四条路线的 single-sample overfit 已完成，结果和 artifact decode 验收见
[001 fixed-speaker result](001_fixed_speaker_codec_generators.md)。LongCat/BiCodec × FM/RVQ
均完成 20 steps，四个 semantic-only waveform 均 finite。

已知阻断：

- 还没有开始 1000 样本主线 screening 和 fixed eval。
- 32-sample checkpoint/resume smoke 已完成，见 [002 32-sample smoke](002_32_sample_smoke.md)。
