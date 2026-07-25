# 001 Fixed Speaker Codec Generator Screening

## 目标

第一版实验以一个固定 speaker 的 Qwen 合成数据集为主线，先在约 1000 条样本上测试
LongCat / BiCodec 两个 codec backend 与 DiT / RVQ 两条 generator route 的生成效果。核心产物是 4
条可重复运行的训练脚本，每条脚本只改变 codec backend 和 generator route，底层 decoder 统一使用
8 层 Qwen decoder 配置。

这版实验同时保留 `wmt19_tts` 和 `wmt19_tts_codec` 真实数据入口，用来验证训练 / codec encode /
prepared data 的真实链路；固定 speaker 合成数据只是 001 screening 的默认实验数据，不是项目唯一数据源。

结果应写入 `docs/experiments/results/001_fixed_speaker_codec_generators.md`，已验证结论再同步到
`docs/experiments/conclusion.md`。

## 判断

这个方向适合先做 screening：

- 固定 speaker 的 1000 条合成样本能压低 speaker、录音条件和文本风格方差，适合先判断 route 是否能学会缺失 codec units。
- LongCat / BiCodec 同时跑可以尽早暴露 backend unit layout 差异，避免只围绕 LongCat 的 RVQ frame layout 设计接口。
- DiT / RVQ 共用 8 层 Qwen decoder 便于控制容量变量，第一版对比更干净。
- 复旦 145 开 8 卡可以顺便验证真实训练入口、分布式 dataloader、checkpoint、artifact export 和 job wrapper。
- 1000 条样本足够看能否跑通和是否有明显生成差异，但不足以声明最终音质结论；结论文档只记录可复现 screening 结果。

## 数据入口

训练层只消费统一 prepared sample contract，不关心样本来自合成数据、真实 waveform，还是已有 codec prepared data：

| Source | 用途 | Codec units 来源 | 第一版角色 |
| --- | --- | --- | --- |
| Qwen fixed-speaker synthetic | 001 主实验数据 | 生成 waveform 后分别 encode LongCat / BiCodec | 默认 screening 数据 |
| `wmt19_tts` | 真实 waveform 入口 | 训练前或准备阶段 encode codec units | 真实链路验证，必须保留 |
| `wmt19_tts_codec` | 已有 codec prepared 入口 | 直接读取 prepared units | LongCat/BiCodec prepared data 快速验证 |

统一样本字段至少包括：

- `utterance_id`
- `speaker_id`
- `text`
- `audio`
- `sample_rate`
- `codec`
- `semantic_units`
- `side_units` 或 `acoustic_units`
- `duration`
- `split`

## 固定 Speaker 合成数据

固定 speaker 数据应先通过 workspace 入口离线落盘，再被训练入口消费。不要在训练 dataloader 里临时合成，避免
TTS 生成耗时、失败和随机性污染训练速度与复现实验。

建议流程：

1. 通过 workspace 的 Qwen 合成入口选择一个固定 speaker。
2. 准备约 1000 条文本，生成对应 waveform。
3. 固定合成参数、采样率、speaker id 和后处理策略。
4. 对同一批 waveform 分别 encode LongCat / BiCodec backend-native units。
5. 写出 train / validation / fixed-eval split；同一 speaker，文本不重叠。
6. 记录数据版本、合成命令、workspace commit、codec backend 版本和样本清单。

如果 workspace 还没有 Qwen fixed-speaker 合成入口，先新增 workspace 数据准备能力，再跑本实验；训练脚本不承担合成职责。

## wmt19_tts / wmt19_tts_codec

`wmt19_tts` 入口要保留，因为它验证真实 waveform -> codec units -> training batch 的链路；`wmt19_tts_codec`
入口要保留，因为它可以跳过 encode，快速验证已有 prepared units。

建议在 001 中做两个轻量验证：

- `wmt19_tts`：抽少量真实 waveform，分别 encode LongCat / BiCodec，验证 unit layout、mask、duration 和 decode finite。
- `wmt19_tts_codec`：读取已有 prepared units，验证能复用同一 collate、route target 和训练 smoke。

这两个入口不参与 1000 fixed-speaker 的主对比，除非后续单独开真实数据实验。

## 实验矩阵

| Script | Codec | Route | Target | Decoder |
| --- | --- | --- | --- | --- |
| `jobs/001/01_longcat_dit.sh` | LongCat | DiT / flow matching | backend acoustic features | 8-layer Qwen decoder blocks |
| `jobs/001/02_longcat_rvq.sh` | LongCat | RVQ | acoustic codebook IDs | 8-layer Qwen decoder |
| `jobs/001/03_bicodec_dit.sh` | BiCodec | DiT / flow matching | backend acoustic / residual features | 8-layer Qwen decoder blocks |
| `jobs/001/04_bicodec_rvq.sh` | BiCodec | RVQ | backend side-unit IDs | 8-layer Qwen decoder |

命名里的 DiT 表示 continuous denoising / flow route。配置层可以继续使用现有 `route=fm`，但脚本和文档使用
`dit` 命名，以便和 RVQ 离散路线区分。

## 统一模型假设

- `decoder.layers=8` 作为第一版固定容量，不在 001 实验里做深度消融。
- `decoder.hidden_dim` 优先跟随 codec semantic embedding / condition adapter 后的主维度；如果两个 backend
  维度差异过大，显式记录各自 hidden dim，不把它作为质量结论变量。
- RVQ 路线使用 Qwen decoder 的 causal/codebook autoregressive 能力，输出每个 side-unit codebook 的分类 head。
- DiT 路线复用 Qwen decoder block 作为 denoiser backbone，但必须支持 timestep embedding、semantic condition 和非
  codebook 分类输出。
- 正式训练默认 bf16；smoke/overfit 只通过 `configs/experiment/` 覆盖，不反向污染默认训练配置。

## 脚本契约

4 个脚本都应只设置必要环境和 Hydra override，最终调用真实 Python 入口，并保留 `"$@"`：

- `codec=longcat route=fm data.source=qwen_fixed_speaker output_subdir=longcat/dit-8l/fixed-speaker`
- `codec=longcat route=rvq data.source=qwen_fixed_speaker output_subdir=longcat/rvq-8l/fixed-speaker`
- `codec=bicodec route=fm data.source=qwen_fixed_speaker output_subdir=bicodec/dit-8l/fixed-speaker`
- `codec=bicodec route=rvq data.source=qwen_fixed_speaker output_subdir=bicodec/rvq-8l/fixed-speaker`

实现前需要先把当前单一 `LongCatBackend.from_pretrained` 入口提升为 codec backend factory，否则 BiCodec 脚本会只是空壳。

## 执行顺序

1. **Workspace data smoke**：在 workspace 上生成或读取少量固定 speaker 样本，确认 text / audio / speaker_id 稳定。
2. **Codec encode smoke**：同一批 waveform 分别产出 LongCat / BiCodec units，验证 frame 对齐、dtype、mask 和 finite decode。
3. **wmt19_tts smoke**：跑真实 waveform 入口的小样本 codec encode，确认真实入口没有被合成数据路径绕开。
4. **wmt19_tts_codec smoke**：读取已有 prepared units，确认同一训练 batch contract 可用。
5. **Route smoke**：4 条路线各跑 1 batch forward/backward/optimizer，确认 finite loss 和 artifact export。
6. **Single-sample overfit**：4 条路线各用同一条样本 overfit，记录 loss 曲线、decode finite、音频长度和基本频谱异常。
7. **145 8-card screening**：在复旦 145 上用 1000 fixed-speaker 样本跑 4 条路线，记录训练稳定性、吞吐和 checkpoint。
8. **Fixed eval**：固定 16 条 held-out 文本，导出 4 组 audio、原始合成 audio 和 backend full-unit reconstruction baseline。
9. **结论归档**：只把可复现结果写入 results；没有完成 fixed eval 前，不在 conclusion 写质量判断。

## 指标

- 通用：train loss、validation loss、finite loss ratio、decode success ratio、waveform duration error。
- DiT：feature MSE / velocity loss、采样 steps、RTF、不同 flow steps 下的稳定性。
- RVQ：codebook CE、top-1 accuracy、per-codebook accuracy、采样温度 / top-p 敏感性。
- 资源：tokens 或 frames per second、peak memory、MFU、checkpoint size。
- 人工检查：固定 eval 音频的爆音、静音、重复、音色一致性和文本可懂度。

## 第一版完成标准

- workspace 能产出或定位 1000 条 fixed-speaker synthetic samples，并生成 LongCat / BiCodec prepared units。
- `wmt19_tts` 和 `wmt19_tts_codec` 都至少通过小样本 smoke，确认真实入口仍可用。
- 4 个脚本均可从空输出目录启动训练，并产出 checkpoint / support artifact。
- 每条路线 smoke 与 single-sample overfit 都无 `NaN` / `Inf` / decode error。
- 145 8-card screening 至少完成一次 1000 样本主线运行，记录吞吐、显存、MFU 和失败样本。
- fixed eval 至少导出 16 条样本的 4 组生成音频与 baseline。
- results 文档记录命令、git commit、数据集版本、主要指标和失败样本。
- 如果 BiCodec adapter、workspace fixed-speaker data 或复旦 145 环境未完成，明确标记为 blocked，不用 LongCat 或本地 smoke 冒充完整对比。

## 待确认事项

- workspace 里 Qwen fixed-speaker synthetic 入口的函数名、参数和输出位置。
- 固定 speaker 的 speaker id、文本来源和 1000 条样本 split 比例。
- BiCodec backend 是否已有可调用 encode/decode 与 side-unit layout 文档。
- DiT 路线是否沿用配置名 `route=fm`，还是新增 `route=dit` 枚举来匹配实验命名。
- 第一版 hidden dim 是否完全固定，还是只固定 `decoder.layers=8`。

