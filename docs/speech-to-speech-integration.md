# speech-to-speech 集成边界

`semantic-acoustic-codec` 后续应作为 `speech-to-speech` 的下游依赖，而不是把新 codec 逻辑继续放在
`speech-to-speech` 内部。本页定义依赖方向和迁移边界。

## 1. 依赖方向

目标方向：

```text
speech-to-speech
    -> semantic-acoustic-codec
        -> anytrain / anydataset / workspace training entry
```

禁止方向：

```text
semantic-acoustic-codec -> speech-to-speech
```

原因：

- reference-optional semantic codec 是通用音频 codec 能力，不应绑定某个 S2S token model。
- `speech-to-speech` 只通过公开 artifact API 消费完整 semantic support 或可复用 generator，不需要知道
  本仓库的 backend feature distillation 细节。
- 避免两个仓库互相 import 模型、runtime singleton 或 Hydra schema。

## 2. speech-to-speech 需要消费什么

`speech-to-speech` 保留负责数据和 token layout 的 backend capability，并按 capability 选择解码路径：

```python
frame_codec.decode(full_codes)         # FrameCodec path
semantic_runtime.decode(semantic_codes)  # SemanticAcousticCodec + SAC artifact
```

接入 `SemanticAcousticCodec` 时，`speech-to-speech` 应选择 acoustic side channel 为 none 的组合：

```text
semantic audio tokens -> SemanticCodecRuntime.decode(semantic_codes) -> waveform
```

`FrameCodec` 不接收 semantic-only codes；需要完整 codebook 展开时使用
`FULL_CODEC_SEQUENCE`，由 frame backend 自己执行 `decode(full_codes)`。

structured path 中 S2S token model 只预测 semantic audio tokens，waveform reconstruction 由本仓库
checkpoint 完成；FrameCodec path 则预测完整展开后的 frame/codebook token 序列。

同一个 SAC artifact 还有一条独立的联合训练用途：

```text
aligned S2S backbone hidden state
    -> S2S HiddenConditionAdapter
    -> initialized SAC generator
    -> acoustic features/codes
```

这条路径使用 `model/acoustic=flow|rvq` 和 `acoustic.init_artifact`。它不调用 SAC semantic conditioner，
也不使用 `runtime.semantic_codec_artifact`；artifact I/O 与 compatibility validation 位于 S2S composition，
model 构造器只接收已经加载好的 `AcousticGeneratorArtifact`。

### Optional cross-text reference

`SemanticCodecRuntime.decode()` 的 target semantic codes 是必需输入；cross-text reference 是请求级可选输入：

```python
runtime.decode(target_semantic_codes, generator=without_reference_generator)
runtime.decode(
    target_semantic_codes,
    reference_features=reference_features,
    reference_mask=reference_mask,
    generator=with_reference_generator,
)
```

省略 reference 时 runtime 使用 artifact 内训练得到的 learned null condition，保持 semantic-only 基线路径；
提供 reference 时，`speech-to-speech` 负责用同一 backend 准备 acoustic features 和对应 mask。reference
必须与 target 来自同一 speaker，同时使用不同 utterance 和不同文本，不能把 target audio 本身作为
reference。

有/无 reference 的质量对照必须固定同一 target，并分别创建以同一个 seed 初始化的独立 generator；不能
复用一个已经被第一条路径推进过 RNG state 的 generator。这样两条路径共享相同初始随机噪声，差异才可归因于
reference condition。reference 是可选增强，不得成为 `speech-to-speech` semantic decode 的必需输入。

## 3. 需要从 speech-to-speech 迁出的能力

当前 `speech-to-speech` 的 codec oracle 已经有可参考实现，但它是 screening experiment，不是
独立 codec runtime。迁移时只迁出稳定能力，不迁出上层任务逻辑：

- semantic embedding initialization；
- semantic token/frame condition repeat；
- RVQ acoustic code decoder；
- FM acoustic feature decoder；
- acoustic feature normalization；
- LongCat backend feature/decode adapter；
- feature/audio logging 的通用部分。

不迁出：

- Qwen backbone；
- text tokenizer/chat template；
- S2S task/stage schedule；
- S2S generation service；
- S2S layout/global token id 逻辑。

## 4. 下沉归属

从 S2S oracle 拆出的逻辑按复用层级归属：

- `semantic-acoustic-codec`：semantic codebook 初始化、frame-level semantic condition、
  reference acoustic condition、LongCat backend feature/feature normalization、Flow/RVQ acoustic decoder、
  `SemanticCodecRuntime.decode(semantic_codes)` artifact runtime 和 codec artifact export/import。
- `anytrain`：不含音频/codec 语义的通用训练积木，例如 sequence DiT backbone、attention backend、
  condition cache、通用 Qwen/MTP codebook predictor、task-agnostic flow matching runtime、MFU/性能统计。
- `speech-to-speech`：只保留 text/audio token model、task datamodule、joint S2S training stage、
  generation service、S2S-specific evaluation/logger，以及 FrameCodec 的 full-code 展开适配。

判断规则是：只要需要解释 LongCat code layout、semantic/acoustic codebook、backend feature、
artifact schema 或 semantic-only waveform reconstruction，就属于本仓库；只有完全不依赖这些领域名词、
能被其它训练任务直接组合的模块，才考虑进入 `anytrain`。`anytrain` 是共享 `third_party` 组件，
迁移前需要先以独立 PR/commit 明确通用契约和测试，不能从本仓库直接复制 project-specific API。

## 5. Artifact 的两种用途

本仓库应产出一个可独立加载的 runtime artifact：

```text
checkpoint_dir/
    model.ckpt
    codec.json
```

当前 artifact schema 为 `7`。`codec.json` 的顶层结构为：

```json
{
  "schema_version": 7,
  "config": {"route": "fm", "...": "..."},
  "backend": {"...": "..."},
  "checkpoint": "model.ckpt"
}
```

其中 `config` 保存 generator 和 sampling 配置，`backend` 保存 runtime 兼容性 metadata。`backend` 至少
包含：

- acoustic layout：`frame_aligned` / `fixed_length`；
- backend name、sample rate、frame rate 与 semantic frame rate；
- semantic vocab size 与 embedding dim；
- acoustic codebook sizes 与 feature dim；
- fixed-length layout 的 acoustic unit length（如适用）。

训练目录中的 `sample_metrics.json`、TensorBoard events 和周期 checkpoint 属于训练产物，不是 runtime
artifact 的必需文件。

artifact 保存生成时的 backend identity 和 rate metadata；`SemanticCodecRuntime` 将其与绑定的 anytrain
backend 严格校验。artifact 不复制 codec 实例或 backend decoder。

semantic-only waveform runtime 指向完整 support artifact：

```yaml
runtime:
  codec: longcat
  semantic_codec_artifact: /path/to/semantic-acoustic-codec/checkpoint_dir
model/acoustic: none
```

字段由当前 `speech-to-speech` runtime schema 持有；S2S 不重建本仓库模型配置，只加载 artifact。

联合训练初始化则把路径放在 acoustic composition：

```yaml
model/acoustic: flow  # 或 rvq
acoustic:
  init_artifact: /path/to/semantic-acoustic-codec/checkpoint_dir
```

两者不能混为同一配置语义：

- `runtime.semantic_codec_artifact` 是推理期完整 support，要求 `model/acoustic=none`，并严格校验 semantic
  与 acoustic backend metadata。
- `acoustic.init_artifact` 是训练开始时的一次性 generator 初始化，要求 Flow/RVQ route 与 decoder topology
  匹配，只以 acoustic metadata 约束目标 backend；artifact 的 semantic vocab/embedding 仅作为来源记录。

当前联合初始化只支持 `FRAME_ALIGNED`。Flow 迁移 decoder 权重与 feature normalization；RVQ 只支持
`codebook_ar` generator，SAC 默认的 `MTP` artifact 会被显式拒绝。fixed-length condition 如何从 S2S
hidden sequence 映射到 acoustic slots 仍需单独设计，不能通过 repeat 或静默截断兼容。

## 6. 两阶段训练与推理期差异

本仓库训练期：

```text
semantic codes + backend acoustic codes
    -> backend acoustic features
    -> train decoder
```

S2S 联合训练期：

```text
semantic token objective -> aligned causal backbone hidden state
    -> trainable HiddenConditionAdapter
    -> initialized acoustic generator
    -> acoustic objective
```

这两个训练阶段只通过 artifact 中的 generator contract 衔接。Phase A 的 semantic/reference conditioner
不注册到 S2S model；Phase B 的 hidden adapter 也不写回 SAC artifact。

`speech-to-speech` 推理期：

```text
generated semantic tokens
    -> semantic codes
    -> SemanticCodecRuntime.decode(reference_features=optional)
    -> waveform
```

因此 backend acoustic codes、backend feature loss、overfit dataloader、训练期 dynamic batch planner 都不进入
`speech-to-speech` runtime。请求提供 reference audio 时，S2S 只额外准备 backend acoustic features 和
mask；reference 的配对约束和有/无 reference A/B 口径仍遵循第 2 节契约。

## 7. 接入验收

semantic-only 基线已完成以下验收：

1. `decode(semantic_codes)` 对真实 LongCat semantic codes 生成 finite waveform。
2. artifact load 后输出与保存前一致，至少同 seed 下 deterministic route 完全一致。
3. route metadata 不匹配时严格报错。
4. sample rate/frame rate 与 LongCat backend 一致。
5. 单条 TTS request 在 `speech-to-speech` 中使用 `model/acoustic=none` 完成 decode。

远端 generation、metadata 和 TensorBoard WAV 证据见
[004 speech-to-speech semantic-only decode smoke](experiments/results/004_speech_to_speech_decode_smoke.md)。
该结果只证明无 reference 的 semantic-only 真实 generation 链路，不构成 optional-reference 下游集成
证据。

optional-reference 的 `speech-to-speech` 接入尚未真实验收，后续必须完成：

1. 在 S2S 真实 generation 请求中同时跑通省略 reference 和提供 reference 的两条 finite waveform 路径。
2. reference 使用与 target 同 speaker、不同 utterance 和不同文本的音频，并保留可核对的 pair metadata。
3. 对同一 target 使用同 seed 初始化的两个独立 generator，导出 with-reference / without-reference 音频和
   对应指标，不能串行复用同一个 generator。
4. 验证省略 reference 时仍走 learned null condition，且不要求调用方构造空 reference tensor 或伪造 mask。

上述四项完成并留下 S2S 侧 generation、metadata 和音频证据前，不得把 optional-reference 标记为已接入。

generator joint initialization 的本地 contract 已覆盖 artifact 加载、hidden adapter、Flow/RVQ state transfer
和不兼容配置失败。真实 frame-aligned Flow 与 RVQ `codebook_ar` artifact 均已完成 S2S 单步 joint smoke、
finite loss/梯度/generation；Flow 另已完成正式 checkpoint resume。该结果只验证第二阶段的初始化和
执行契约，不支持质量或收敛结论，RVQ checkpoint resume 仍未单独验证。
