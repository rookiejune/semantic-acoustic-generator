# 004 Speech-to-Speech Semantic-Only Decode Smoke

## 目标

在 `speech-to-speech` 的真实 generation 路径中加载 semantic-acoustic-codec artifact，验证
`model/acoustic=none` 时只依赖生成的 LongCat semantic codes 也能完成 waveform decode。

本实验只验证两个仓库之间的 runtime 契约，不验证 token model 收敛、语音可懂度或音质。

## 输入

- semantic-acoustic-codec artifact：LongCat FM 8-layer single-sample overfit artifact。
- speech-to-speech codec：LongCat native semantic/acoustic backend。
- speech-to-speech model/data：toy token model 与 toy sample。
- task：TTS。
- parameter policy：`semantic_only`。
- acoustic composition：`none`。

## 执行

在复旦 145 的单张 GPU 上运行 `speech-to-speech/scripts/overfit.py`：

- `experiment=longcat_decoupled_semantic_only_smoke`
- `runtime.semantic_codec_artifact=<artifact>`
- `runtime.device=cuda`
- `trainer.accelerator=gpu`
- `logging=tensorboard`
- `callbacks.task_sample.enabled=true`
- `callbacks.task_sample.every_n_steps=1`
- `train.max_steps=1`

## 验收

1. 训练入口到达 `max_steps=1`，无 traceback。
2. `task_sample/0/metadata` 的状态为 `ok`。
3. generation result 中 `features` 为 `null`，确认没有 acoustic side channel。
4. metadata 报告 waveform finite，TensorBoard audio event 可独立解码且采样数、采样率一致。
5. 记录 token 数和是否达到 `max_new_tokens`，避免把随机 toy generation 当成质量结果。

结果写入 `docs/experiments/results/004_speech_to_speech_decode_smoke.md`。
