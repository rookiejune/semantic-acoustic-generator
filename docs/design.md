# Semantic Acoustic Codec Design

## 目标

本项目实现一个 reference-optional semantic codec：输入音频时只导出语义码；解码时可以只接收
语义码，也可以额外接收 reference acoustic features。当前训练主线使用 Qwen speaker grid 离线生成的
LongCat / BiCodec units，并由 `qwen_cross_text` 构造显式 target/reference pair。每个 reference 必须和
target 属于同一 speaker，同时 sample index、utterance id 和文本均不同；无法配对时在 dataset 构造阶段
直接报错。

保留的 LongCat prepared source 按现有约定解释 code layout：

- `codes[..., :1]`：semantic codebook。
- `codes[..., 1:]`：acoustic RVQ codebooks。
- semantic 和 acoustic codebooks 共享 frame 轴。

抽象层不能绑定 LongCat 的 frame-level acoustic RVQ layout。LongCat backend 的缺失部分可以是
frame-level acoustic features 或 RVQ codebooks；BiCodec backend 的缺失部分可能是 semantic 外的 global /
acoustic / residual units。公共 runtime 稳定 semantic 必需输入与 optional-reference 输入，backend adapter
和 generator 负责解释各 codec 自己的 side-unit layout。

这个仓库要把 `speech-to-speech` 当前 codec oracle 中可复用的 acoustic decoder 能力沉淀为
独立包。`speech-to-speech` 之后只依赖本仓库的公开 codec/runtime/model 契约，不再持有
LongCat semantic-only reconstruction 的实现细节。

## 非目标

- 不在训练进程内在线合成 TTS 数据；Qwen speaker grid 必须先在 workspace 离线物化。
- 不把 `speech-to-speech` 的 token model、generation service 或 task datamodule 搬进本仓库。
- 不让本仓库 import `speech-to-speech`，避免循环依赖。
- 不把 acoustic codes 暴露成调用方必须提供的推理输入。
- 不用兼容逻辑静默吞掉 codec capability 缺失；缺少 semantic codebook、acoustic feature decode 或
  codebook size 时直接报错。

## 顶层契约

公共接口优先稳定在真实 codec backend 和 semantic-only support wrapper 边界，而不是训练脚本边界。
codec backend contract 直接使用 `anytrain.codec.SemanticAcousticCodec` 和
`anytrain.codec.load_semantic_acoustic()`，本仓库不再重新定义或转发 codec protocol：

```python
class SemanticCodecSupport(nn.Module):
    generator: CodecUnitGenerator

    def sample_features(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...

    def sample_acoustic_codes(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


class SemanticCodecRuntime:
    support: SemanticCodecSupport
    backend: anytrain.codec.SemanticAcousticCodec

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...
    def sample_features(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...
    def decode(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...
```

`SemanticCodecSupport` 不持有 codec backend：FM 路线可直接生成 features，RVQ 路线的 code-to-feature
转换必须通过 `SemanticCodecRuntime` 完成。`mask` 用于 semantic padding，`output_length` 只适用于
`FIXED_LENGTH` acoustic layout，`generator` 用于固定 seed 的采样对照。

训练侧另外定义 codec unit generator，不要求调用方知道 FM/RVQ 或 codec-specific side-unit 路线：

```python
class CodecUnitGenerator(Protocol):
    def loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_features: Tensor | None = None,
        target_codes: Tensor | None = None,
        repa_features: Tensor | None = None,
    ) -> DecoderLoss: ...

    def loss(self, batch: SemanticCodecBatch) -> Mapping[str, Tensor]: ...
    def sample_features(
        self,
        semantic_codes: Tensor,
        *,
        output_length: int | None = None,
    ) -> Tensor: ...
```

`loss_from_condition()` 是训练 route 的复用边界：调用方可以直接提供外部模型产生的
frame/slot condition，FM 接收 acoustic features，RVQ 接收 acoustic code IDs。`loss(batch, ...)`
只负责把本仓库的 `SemanticCodecBatch` 和 layout/mask 规则适配到这个入口；它不要求外部调用方
构造语义 code batch。feature normalization、REPA teacher features 和 flow runtime 都在 route
参数中显式传入，缺失或 shape 不一致时直接报错。

`SemanticCodecBatch` 使用严格结构表达数据：

- `semantic_codes: Tensor`，shape `[B, F, 1]`，signed integer dtype。
- `acoustic_codes: Tensor`，shape `[B, U, K]`，其中 `U` 可以是 frame 或固定 acoustic slot。
- `mask: Tensor`，shape `[B, F]`，标记 semantic 时间轴的有效位置。
- `acoustic_mask: Tensor`，shape `[B, U]`，标记 acoustic unit 轴的有效位置。
- `acoustic_layout: AcousticLayout`，`FRAME_ALIGNED` 要求 `F == U` 且 mask 相同；
  `FIXED_LENGTH` 允许 semantic 时间轴和 acoustic slot 轴独立变化。
- `reference_semantic_codes`、`reference_acoustic_codes`、`reference_mask` 和
  `reference_acoustic_mask`：`qwen_cross_text` batch 中完整保留 cross-text reference side。
- `metadata: tuple[SemanticCodecPairMetadata, ...]`：逐行记录 target/reference store index、grid
  `text_index`、原始 `source_index`、role、utterance id、speaker id 和文本，并在构造时维护
  same-speaker / cross-text 不变量。
- `semantic_tokens: Tensor | None` 和 `semantic_token_spans: Tensor | None`，仅在显式使用
  semantic BPE tokenizer 时存在；native LongCat 路线默认直接使用 frame-level semantic codes。

`collate_structured_codes()` 直接接收 anytrain 的 `SemanticAcousticCodes`，按两个轴分别 padding。
因此 BiCodec 的 `[time, 1]` semantic 与 `[slot, codebook]` acoustic 不会被拼成一个伪造的
`[frame, codebook]` Tensor。`FIXED_LENGTH` 不对 semantic 做 mean-pool 后复制：codec 声明的每个
acoustic slot 都有 learned query，通过 masked cross-attention 读取完整、带位置信息的 semantic memory。
BiCodec RVQ 的 temporal MTP 再沿 32-slot 轴自回归生成；FM 保留为在同一 32-slot condition 上联合生成的
对照路线。`FRAME_ALIGNED` 仍保持逐帧 condition。

## 模块边界

建议源码组织：

```text
src/semantic_acoustic_codec/
  backend/        # LongCat prepared-code parser and structured-unit collation helpers
  runtime/        # codec-free support, codec runtime composition and artifact loader
  types.py        # SemanticCodecBatch and other cross-backend training contracts
  datamodule/     # paired Qwen/legacy WMT prepared data, filtering and Lightning data modules
  model/
    condition.py  # semantic embedding, BPE span repeat, linear projection
    dit.py        # FM decoder backbone
    rvq.py        # acoustic RVQ code predictor
  loss/           # route-specific losses
  pl_module/      # LightningModule and training factories
  callback/       # artifact export and sample/audio logging callbacks
scripts/
  train.py        # production train entry
  smoke.py        # minimal local validation
  eval_artifact.py # one-pair artifact evaluation and WAV export
configs/
  train.yaml
  experiment/
jobs/
docs/
```

`runtime/`、`backend/`、`types.py` 和 `model/` 是 `speech-to-speech` 未来依赖的稳定层；
`datamodule/`、`pl_module/`、`callback/`、`scripts/` 是本仓库训练实现，不应被
`speech-to-speech` 直接 import。

codec backend 的加载和 capability 分类由 anytrain 统一提供；本仓库训练入口通过
`anytrain.codec.load_semantic_acoustic(codec)` 消费 structured backend，不再增加
一层只转发属性和方法的 LongCat/BiCodec wrapper。固定长度 acoustic units 的训练适配属于本仓库的
batch/generator contract，不属于 codec loader。

## 复用约定

模型内部优先复用已有通用模块，不先为本仓库重写 Transformer、attention、cache、MTP 或 codebook AR
基础结构。当前 RVQ 路线明确复用 anytrain 的 Qwen/Qwen3 model builder，以 temporal backbone 处理
acoustic unit 轴的 causal dependency，并以 MTP backbone 处理 unit 内 codebook dependency；本仓库只负责
codec 语义、condition、head、loss 和 runtime 组合。

后续如果需要优化 RVQ 的专用结构，应先用当前 Qwen 路线跑通质量/效率闭环，再单独设计和验证替代模块；
不要在 P0/P1 阶段为了“看起来更小”引入自定义基础网络。

## 数据路线

正式训练默认使用 `qwen_cross_text`：

1. workspace 从 Qwen speaker grid 离线生成 waveform，并分别物化 LongCat / BiCodec structured units。
2. dataset 按 source order 为每个 target 确定性选择同 speaker、不同 utterance 和文本的 reference。
3. datamodule 分别 padding target/reference 的 semantic 与 acoustic 轴，并保留 pair metadata。
4. target acoustic units 只构造 FM continuous target 或 RVQ labels；reference acoustic units 只构造
   reference condition，不能把 target features 回流到 condition。
5. 训练逐样本以 `reference_dropout=0.5` 在显式 reference 与 learned null condition 之间切换。

`wmt19_tts_codec(longcat)` 与单侧 `qwen_fixed_speaker` 仍作为显式 smoke source 保留；后者只是从统一
Qwen codec grid 选择一个 role/speaker 列，codec view 与网格的行、列、整网格访问契约保持一致，不再依赖
独立的固定 speaker prepared dataset。默认训练数据契约仍是 `qwen_cross_text`。

## 两条 generator 路线

### RVQ

RVQ 路线预测 backend acoustic codebook IDs：

- 条件：同样使用 frame-level semantic condition。
- 目标：`codes[..., 1:]` 中的 acoustic codebooks。
- 训练：按 codebook 计算 causal cross entropy；padding frame 不参与 loss。
- 推理：采样 acoustic codebook IDs，再通过 codec 的 code-to-feature 路径转换为 acoustic features，
  最后 decode waveform。

RVQ 的优势是目标离散、与 LongCat acoustic representation 对齐；风险是多 codebook 采样误差会逐层累积。
generator 需要显式持有每个 codebook size，不能假设所有 acoustic codebook 共用相同 vocab size。
当前默认 predictor 是 MTP：temporal Qwen backbone 对 acoustic unit 轴做 causal teacher forcing 和
cached autoregressive generation；一个 unit 内存在多个 codebook 时，再由小型 MTP backbone 逐 codebook
补全。对 BiCodec 的 `[B, 32, 1]` acoustic units，这意味着严格沿 32-slot 轴生成，不是把一个 pooled
condition 复制 32 次。`FIXED_LENGTH + CODEBOOK_AR` 在构造时直接拒绝。

### FM

FM 路线复用 DiT backbone，但 objective 使用 continuous flow matching：

- 条件：semantic condition，可选叠加 reference acoustic condition。
- 目标：backend acoustic features。
- 训练：预测从 noise 到 target acoustic feature 的 velocity。
- 推理：ODE sampler 生成 acoustic features，再调用 `decode_features`。

这条路线最接近 `speech-to-speech` 当前 codec oracle 的 Flow screening，可以作为 P0
最小闭环优先落地；但实现应放在本仓库，oracle 只作为迁移参考。

## 共享 condition 层

两条路线共用 semantic condition：

1. native LongCat：`semantic_codes[..., 0]` 直接查 `codec.semantic_codebook` 初始化的 embedding。
2. CodecBPE：BPE token embedding 按 `semantic_token_spans` repeat 到 frame 轴。
3. projection：仅在 codec embedding dim 和 decoder condition dim 不一致时做线性投影。

semantic condition 层只负责 semantic 表示，不读取 acoustic target，也不构造 text/audio vocabulary head。
初始化策略由枚举控制，例如 codec initialization 和 matched random initialization，字符串只在配置边界解析一次。

reference condition 是独立支路：只读取 paired cross-text reference acoustic features，masked pooling 后映射到
condition space，作为全局 speaker/reference 条件广播到 target semantic 时间轴。这里的 pooling 只发生在
reference 支路；target semantic memory 不做 pooling。无 reference 或逐样本 dropout 时直接使用 condition-space
的 learned null condition，不先伪造 feature-space reference。

## 缺失 unit 生成

用户侧 runtime 的唯一必需输入是 semantic codes；`decode()`、`sample_features()` 和
`sample_acoustic_codes()` 额外接受可选的 `reference_features/reference_mask`。LongCat 的 acoustic
features / RVQ codes、BiCodec 的 semantic 外 global / acoustic / residual units，都属于缺失 codec units，
必须由 `CodecUnitGenerator` 在 runtime 内部生成，不能成为调用方必须传入的 target 输入。

训练时可以使用 target side units 构造 backend feature、loss target 或 condition dropout；这些只属于
`datamodule/`、`pl_module/` 和 `model/` 的训练内部契约。推理 artifact 必须保存生成缺失 units 所需的
模型权重、normalization 和 backend compatibility metadata。`SemanticCodecSupport` 本身不持有 codec；
加载后由 `SemanticCodecRuntime(support, backend).decode(semantic_codes)` 完成端到端 waveform
reconstruction。

## 训练与验收

当前已完成的闭环包括：

- paired data contract、FM/RVQ route contract 和 artifact roundtrip 的本地测试；
- LongCat / BiCodec 四条路线的 fixed-sample overfit 与 finite waveform decode；
- 32-sample LongCat FM 的 checkpoint/resume、sample metrics、TensorBoard audio 和 artifact export；
- `speech-to-speech` 中一条真实 semantic-only TTS decode smoke。

验收证据见 [experiments/results/](experiments/results/)。当前仍待验证的是：

1. 用当前 `qwen_cross_text` pair contract 重跑 LongCat / BiCodec × FM / RVQ single-pair overfit，分别记录
   with-reference 与 without-reference 的 loss、feature MSE、音频和 `reference_gain`。
2. 对 BiCodec RVQ 的真实 backend 运行 32-slot temporal AR generation，确认 finite decode 和推理耗时。
3. 在 `speech-to-speech` 真实 generation 中同时跑通省略 reference 和提供 reference 的两条路径，并保留
   可核对的 pair metadata 与同 seed A/B 指标。
4. 完成约 1000 样本的四路线 screening，以及至少 16 条 held-out cross-text fixed eval；记录 loss、双路
   feature MSE、waveform finite、RTF、显存、MFU 和失败样本。
5. 检查 reference 是否泄漏文本内容；验证完成前不把 reference gain 或音色保持写入 conclusion。

其中 fixed-speaker 结果不等同于当前默认的 `qwen_cross_text` 训练契约；cross-text 重跑、真实 BiCodec
RVQ 32-slot generation、held-out fixed eval 和 reference leakage 检查以
[experiments/todo.md](experiments/todo.md) 为准。

正式训练默认面向完整数据和长预算；smoke/overfit 配置只放在 `configs/experiment/`，不反向污染
生产 preset。

## 与 speech-to-speech 的关系

迁移目标是：

- `semantic-acoustic-codec` 消费 anytrain 提供的 LongCat/BiCodec backend、拥有 codec-free `SemanticCodecSupport`、
  FM/RVQ unit generator、sampling 和 `SemanticCodecRuntime.decode(semantic_codes)` 实现。
- `speech-to-speech` 拥有 text/audio token model、task datamodule、generation service 和
  evaluation。
- `speech-to-speech` 只通过公开 `SemanticCodecSupport`、anytrain 的
  `SemanticAcousticCodec`、`CodecUnitGenerator` 和
  artifact loading API 依赖本仓库。

避免循环依赖的规则：

- 本仓库不 import `speech_to_speech`。
- 可复用的 oracle model、Flow/RVQ decoder 和 condition 初始化逻辑从 `speech-to-speech` 迁移或复制到
  本仓库后，`speech-to-speech` 再删除本地重复实现。
- generation 中 semantic-only decode 使用 `SemanticCodecRuntime(support, backend).decode(...)`；训练只用
  backend 将 prepared acoustic codes 转成 target/reference features，support 不持有 backend。
- checkpoints 使用稳定前缀区分 semantic condition、unit generator 和 support wrapper，不依赖调用方包名。

## 关键风险

- 无 reference 输入缺少明确 speaker/prosody/channel 信息，其质量上限由训练分布和 learned null condition
  决定；不能声称支持任意 speaker 保真重建。
- LongCat backend capability 必须显式存在：semantic codebook、acoustic codebook sizes、
  acoustic code-to-feature、feature decode 缺一不可。
- RVQ codebook size 可能不一致，不能用单一 vocab size 近似。
- reference condition 仍可能泄漏内容；pair contract 强制同 speaker、不同 utterance 和不同文本，但是否
  消除内容泄漏必须由 held-out cross-text A/B 实验验证。
- `speech-to-speech` 依赖迁移前会短期存在重复实现；每次迁移都应以公开接口和测试为边界，不做隐式兼容。
