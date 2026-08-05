# 004 Speech-to-Speech Semantic-Only Decode Smoke

## 范围

- 日期：2026-07-27。
- 机器：复旦 145，NVIDIA RTX 4090 D，GPU1。
- Python：`/home/zhuyin/anaconda3/envs/py312/bin/python`。
- speech-to-speech entry：`scripts/overfit.py`。
- Artifact：`/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/overfit-longcat-fm-8l-sample0/artifact`。
- 输出：`/tmp/s2s-semantic-codec-tts-smoke-1785149639053`。

## 配置

运行使用以下关键 override：

- `experiment=longcat_decoupled_semantic_only_smoke`
- `runtime.semantic_codec_artifact=<artifact>`
- `runtime.device=cuda`
- `trainer.accelerator=gpu`
- `trainer.devices=1`
- `trainer.strategy=auto`
- `trainer.precision=32-true`
- `logging=tensorboard`
- `callbacks.task_sample.enabled=true`
- `callbacks.task_sample.every_n_steps=1`
- `train.max_steps=1`

experiment 组合显式使用真实 LongCat backend、`model/acoustic=none`、toy token model、toy data、
TTS task 和 `semantic_only` parameter policy。

## 训练结果

- 远端进程正常结束；`run.log` 无 traceback，并记录训练因达到 `max_steps=1` 正常停止。
- `train/loss=9.419875144958496`。
- 总参数量 `15,435,328`，trainable 参数量 `10,571,552`。
- acoustic decoder 参数量为 `0`。

## Decode 结果

TensorBoard event：

`/tmp/s2s-semantic-codec-tts-smoke-1785149639053/tensorboard/longcat-decoupled-semantic-only-smoke/tts/longcat-decoupled-semantic-only/version_0/events.out.tfevents.1785149769.145.pami.group.1295783.0`

| 字段 | 结果 |
| --- | --- |
| Metadata tag | `task_sample/0/metadata/text_summary` |
| Audio tag | `task_sample/0/generated` |
| Event step | `0` |
| Status | `ok` |
| Task / dataset index | `tts` / `0` |
| Result features | `null` |
| Waveform shape | `[1, 245760]` |
| Sample rate | `16000` Hz |
| Duration | `15.36` s |
| Waveform finite | `true` |
| Response tokens | `256` |
| Reached max new tokens | `true` |

event 中的 WAV 另外通过 Python `wave` 和 NumPy 解码验证：单声道、16-bit PCM、245760 samples、
全部 finite；采样率、通道数和采样数均与 metadata 一致。

## 结论边界

`speech-to-speech` 已在 `model/acoustic=none` 下通过 semantic-acoustic-generator artifact 完成一条真实
TTS generation waveform decode。`features=null`、acoustic decoder 参数量为 0，且 event 内 WAV
可独立解码，验证了 semantic-only runtime 接入闭环。

本次使用随机 toy token model，生成达到 `max_new_tokens=256`；15.36 秒音频只用于 decode smoke，
不支持语音可懂度、音色、音质、收敛或泛化结论。
