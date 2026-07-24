# Semantic Acoustic Codec Design

## 目标

本项目实现一个 semantic-only codec：输入音频时只导出语义码，解码时只接收语义码并重建波形。
第一版不重新合成数据，直接使用 `wmt19_tts_codec(longcat)` prepared dataset 中的 LongCat
codes。LongCat code layout 按现有约定解释为：

- `codes[..., :1]`：semantic codebook。
- `codes[..., 1:]`：acoustic RVQ codebooks。
- semantic 和 acoustic codebooks 共享 frame 轴。

抽象层不能绑定 LongCat 的 frame-level acoustic RVQ layout。LongCat backend 的缺失部分可以是
frame-level acoustic features 或 RVQ codebooks；BiCodec backend 的缺失部分可能是 semantic 外的 global /
acoustic / residual units。公共 runtime 只稳定 semantic-only 输入输出，backend adapter 和 generator 负责
解释各 codec 自己的 side-unit layout。

这个仓库要把 `speech-to-speech` 当前 codec oracle 中可复用的 acoustic decoder 能力沉淀为
独立包。`speech-to-speech` 之后只依赖本仓库的公开 codec/runtime/model 契约，不再持有
LongCat semantic-only reconstruction 的实现细节。

## 非目标

- 不在第一阶段新增指定 speaker TTS 合成数据。
- 不把 `speech-to-speech` 的 token model、generation service 或 task datamodule 搬进本仓库。
- 不让本仓库 import `speech-to-speech`，避免循环依赖。
- 不把 acoustic codes 暴露成调用方必须提供的推理输入。
- 不用兼容逻辑静默吞掉 codec capability 缺失；缺少 semantic codebook、acoustic feature decode 或
  codebook size 时直接报错。

## 顶层契约

公共接口优先稳定在真实 codec backend 和 semantic-only support wrapper 边界，而不是训练脚本边界：

```python
class CodecBackend(Protocol):
    name: str
    sample_rate: int
    frame_rate: float
    semantic_codebook: Tensor

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Return backend-native full units; support wrapper extracts semantic codes."""

    def decode(self, units: Tensor) -> Tensor:
        """Decode backend-native full/generated units."""

    def decode_features(self, semantic_codes: Tensor, features: Tensor) -> Tensor:
        """Decode explicit acoustic features; used by training, logging and baselines."""

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        """Convert backend acoustic RVQ codes into codec acoustic features."""


class SemanticCodecSupport(nn.Module):
    backend: CodecBackend
    generator: CodecUnitGenerator

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Return semantic codes with shape [batch, frame, 1]."""

    def decode(self, semantic_codes: Tensor) -> Tensor:
        """Generate missing codec units internally and return waveform."""
```

训练侧另外定义 codec unit generator，不要求调用方知道 FM/RVQ 或 codec-specific side-unit 路线：

```python
class CodecUnitGenerator(Protocol):
    def loss(self, batch: SemanticCodecBatch) -> Mapping[str, Tensor]: ...
    def sample_features(self, semantic_codes: Tensor, *, frames: int | None = None) -> Tensor: ...
```

`SemanticCodecBatch` 使用严格结构表达数据：

- `semantic_codes: Tensor`，shape `[B, F, 1]`，signed integer dtype。
- `acoustic_codes: Tensor | None`，shape `[B, F, K]`，训练 RVQ/backend feature 时存在。
- `mask: Tensor`，shape `[B, F]`，只标记有效 frame。
- `semantic_tokens: Tensor | None` 和 `semantic_token_spans: Tensor | None`，仅在显式使用
  semantic BPE tokenizer 时存在；native LongCat 路线默认直接使用 frame-level semantic codes。

## 模块边界

建议源码组织：

```text
src/semantic_acoustic_codec/
  backend/        # LongCatBackend / BiCodecBackend adapters around real codec instances
  runtime/        # SemanticCodecSupport, artifact loader and public Protocols
  datamodule/     # wmt19_tts_codec(longcat) batch extraction, collate and Lightning data modules
  model/
    condition.py  # semantic embedding, BPE span repeat, adapter
    dit.py        # FM decoder backbone
    rvq.py        # acoustic RVQ code predictor
  loss/           # route-specific losses
  pl_module/      # LightningModule and training factories
  callback/       # artifact export and sample/audio logging callbacks
scripts/
  train.py        # production train entry
  smoke.py        # minimal local validation
configs/
  codec/
  model/
  experiment/
jobs/
docs/
```

`runtime/`、`backend/` 和 `model/` 是 `speech-to-speech` 未来依赖的稳定层；
`datamodule/`、`pl_module/`、`callback/`、`scripts/` 是本仓库训练实现，不应被
`speech-to-speech` 直接 import。

## 数据路线

当前只使用 `wmt19_tts_codec(longcat)`：

1. 从 anydataset/anytrain workspace 读取 prepared sample。
2. 取 target audio 的 LongCat view。
3. 将完整 codes 拆成 semantic/acoustic 两组。
4. 用 acoustic codes 通过 LongCat backend 得到 acoustic features，作为 FM 的连续目标。
5. RVQ 路线直接监督 acoustic codebook IDs。

第一阶段默认只消费 target audio。source/target 双侧扩展、固定 speaker 合成和数据过滤策略留到
semantic-only 重建质量证明后再加入。

## 两条 generator 路线

### RVQ

RVQ 路线预测 LongCat acoustic codebook IDs：

- 条件：同样使用 frame-level semantic condition。
- 目标：`codes[..., 1:]` 中的 acoustic codebooks。
- 训练：按 codebook 计算 causal cross entropy；padding frame 不参与 loss。
- 推理：采样 acoustic codebook IDs，再通过 codec 的 code-to-feature 路径转换为 acoustic features，
  最后 decode waveform。

RVQ 的优势是目标离散、与 LongCat acoustic representation 对齐；风险是多 codebook 采样误差会逐层累积。
generator 需要显式持有每个 codebook size，不能假设所有 acoustic codebook 共用相同 vocab size。

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
3. adapter：将 codec embedding dim 转换到 decoder condition dim。

semantic condition 层只负责 semantic 表示，不读取 acoustic target，也不构造 text/audio vocabulary head。
初始化策略由枚举控制，例如 codec initialization 和 matched random initialization，字符串只在配置边界解析一次。

## 缺失 unit 生成

用户侧 runtime 契约严格保持 semantic-only：`decode()` 只接收 semantic codes。LongCat 的 acoustic
features / RVQ codes、BiCodec 的 semantic 外 global / acoustic / residual units，都属于缺失 codec units，
必须由 `CodecUnitGenerator` 在 runtime 内部生成，不能成为调用方必须传入的输入。

训练时可以使用 target side units 构造 backend feature、loss target 或 condition dropout；这些只属于
`datamodule/`、`pl_module/` 和 `model/` 的训练内部契约。推理 artifact 必须保存生成缺失 units 所需的
模型权重、normalization 和 backend metadata，加载后由 `SemanticCodecSupport.decode(semantic_codes)` 完成
端到端 waveform reconstruction。

## 训练与验收

P0 文档落地后，代码实现按以下闭环推进：

1. data smoke：加载一条 `wmt19_tts_codec(longcat)` sample，校验 `[F, K]` layout、dtype、
   semantic/acoustic frame 对齐和 finite backend features。
2. FM overfit：单样本训练到 feature MSE 明显下降，sample waveform finite。
3. RVQ overfit：单样本训练到 acoustic codebook CE 和 accuracy 明显改善，采样后 waveform finite。
4. semantic-only decode smoke：确认缺省路径只用 semantic codes 也能生成 finite waveform。
5. 32/1000 sample smoke：记录 loss、feature MSE、waveform finite、RTF、显存和 MFU。

正式训练默认面向完整数据和长预算；smoke/overfit 配置只放在 `configs/experiment/`，不反向污染
生产 preset。

## 与 speech-to-speech 的关系

迁移目标是：

- `semantic-acoustic-codec` 拥有 LongCat/BiCodec backend adapter、SemanticCodecSupport、FM/RVQ unit generator、sampling 和
  `decode(semantic_codes)` 实现。
- `speech-to-speech` 拥有 text/audio token model、task datamodule、generation service 和
  evaluation。
- `speech-to-speech` 只通过公开 `SemanticCodecSupport`、`CodecBackend`、`CodecUnitGenerator` 和
  artifact loading API 依赖本仓库。

避免循环依赖的规则：

- 本仓库不 import `speech_to_speech`。
- 可复用的 oracle model、Flow/RVQ decoder 和 condition 初始化逻辑从 `speech-to-speech` 迁移或复制到
  本仓库后，`speech-to-speech` 再删除本地重复实现。
- generation 中 semantic-only decode 使用 `support.decode(semantic_codes)`；有 acoustic features 的训练/日志路径使用
  `support.decode_features(semantic_codes, features)`。
- checkpoints 使用稳定前缀区分 semantic condition、unit generator 和 support wrapper，不依赖调用方包名。

## 关键风险

- semantic-only 输入缺少 speaker/prosody/channel 信息。当前使用 WMT19 TTS LongCat prepared data 时，
  质量上限由数据分布和 generator 默认条件决定，不能声称支持任意 speaker 保真重建。
- LongCat backend capability 必须显式存在：semantic codebook、acoustic codebook sizes、
  acoustic code-to-feature、feature decode 缺一不可。
- RVQ codebook size 可能不一致，不能用单一 vocab size 近似。
- reference condition 可能泄漏内容；训练采样必须优先使用同 speaker 的不同音频作为 reference。
- `speech-to-speech` 依赖迁移前会短期存在重复实现；每次迁移都应以公开接口和测试为边界，不做隐式兼容。
