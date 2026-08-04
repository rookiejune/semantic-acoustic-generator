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

backend 的可移植结构统一通过 `anytrain.codec.SemanticAcousticCodecSpec` 表达；训练构建从真实
backend 使用 `semantic_acoustic_spec()` 检查得到，artifact load 则从严格 metadata 重建同一 spec。
spec 只持有码本、feature dim、frame rate 和 acoustic unit layout，不持有 backend 权重或加载策略。

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
`FIXED_LENGTH` acoustic layout，`generator` 用于固定 seed 的采样对照。waveform decode 只接受每行
有效长度相同的 batch，mask 必须表示连续的右侧 padding；runtime 会在进入 backend 前裁掉 padding。
不同长度的请求应先按有效长度分组，或逐行 decode。

训练侧的公共基类只持有两条路线共有的 target-axis condition 适配；采样和 loss 按实际能力拆分：

```python
class CodecUnitGenerator(nn.Module):
    route: Route


class FeatureSampler(Protocol):
    def sample_features(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        flow_steps: int,
        unconditional_condition: Tensor | None = None,
        cfg_scale: float = 1.0,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


class AcousticCodeSampler(Protocol):
    def sample_acoustic_codes(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        top_p: float,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


class FMFeatureGenerator(CodecUnitGenerator):
    def feature_loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_features: Tensor,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        repa_features: Tensor | None = None,
        flow_runtime: FlowRuntime | None = None,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss: ...

    def loss(
        self,
        batch: SemanticCodecBatch,
        condition: Tensor,
        target_features: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
    ) -> DecoderLoss: ...


class RVQCodeGenerator(CodecUnitGenerator):
    def code_loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_codes: Tensor,
        include_top1: bool = False,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss: ...

    def loss(self, batch: SemanticCodecBatch, condition: Tensor) -> DecoderLoss: ...
```

`feature_loss_from_condition()` 和 `code_loss_from_condition()` 是两条训练路线各自的复用边界：FM 只接收
acoustic features，RVQ 只接收 acoustic code IDs；两者不再通过 nullable target 参数模拟一个万能接口。
`SemanticCodecSupport` 在 runtime 侧按 `FeatureSampler` / `AcousticCodeSampler` capability 调用，错误路线直接
报错。`loss(batch, ...)` 只负责把本仓库的 batch 和 layout/mask 规则适配到对应入口。FM objective 统一由
`loss/flow.py` 的 `FlowLoss` 持有，feature normalization、REPA teacher features 和 flow runtime 都显式传入。

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

`mask` 和 `acoustic_mask` 都是构造时必填字段；batch 不提供 `semantic_mask`、`target_mask`、
`target_acoustic_mask` 或 `target_*` tensor alias。同一含义只保留一个名称，reference 侧通过
`batch.reference` 返回四个已校验的必需 tensor。

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
  backend/        # strict backend config and anytrain codec loading
  evaluation.py   # shared seeded with/without-reference feature evaluation
  runtime/        # codec-free support, codec runtime composition and artifact loader
  types.py        # SemanticCodecBatch and other cross-backend training contracts
  datamodule/     # source adapters, paired Qwen data, batching and Lightning data modules
  model/
    condition.py  # semantic embedding, BPE span repeat, linear projection
    dit.py        # FM decoder backbone
    rvq.py        # acoustic RVQ code predictor
  loss/           # route-specific losses
  pl_module/      # LightningModule and training factories
  callback/       # artifact export and sample/audio logging callbacks
  training/       # strict train config, component factories and TrainingSession
scripts/
  train.py        # thin Hydra entry delegating to semantic_acoustic_codec.training
  smoke.py        # minimal local validation
  eval_artifact.py # one-pair artifact evaluation and WAV export
configs/
  backend/        # codec backend selection
  datamodule/     # prepared data source and loader policy
  model/          # decoder and FM/RVQ route
  loss/           # route loss options such as REPA
  pl_module/      # optimization and training-module options
  runtime/        # support initialization and sampling
  callback/       # sample, performance and checkpoint callbacks
  trainer/        # Lightning execution preset
  train.yaml      # production defaults and entry composition
  experiment/     # explicit end-to-end compositions and budgets
jobs/
docs/
```

Hydra 配置按源码模块命名空间组织；入口不维护 `data`、`optimizer`、`sample` 等平铺别名。
`train.yaml` 提供生产默认组合，`experiment/` 中的每个文件显式选择 backend、datamodule、model route、
loss、pl_module、runtime、callback 和 trainer，再覆盖该实验独有的数据范围、预算和输出目录。
外部 Hydra/JSON 输入只在边界解析一次，随后使用 frozen dataclass；`BackendConfig`、`DecoderConfig`、
`SamplingConfig` 和 `SemanticSupportConfig` 严格校验枚举、布尔值、整数及有限浮点数，不把字符串或 bool
静默强转为数值。

`semantic_acoustic_codec.training` 是仓库内可复用的训练服务边界。`parse_train_config()` 负责严格配置
解析，`build_session()` 组装 backend、data module、support、Lightning module、callbacks 和 trainer，
并返回持有单次 `fit()` 生命周期的 `TrainingSession`；`run()` 只执行这两个阶段。Hydra 装饰器与命令行
配置路径只存在于 `scripts/train.py`，因此测试、job 或其他仓库内调用方不需要 import 脚本模块。

`runtime/`、`backend/`、`types.py` 和 `model/` 是 `speech-to-speech` 未来依赖的稳定层；
`datamodule/`、`pl_module/`、`callback/`、`training/` 和 `scripts/` 是本仓库训练实现，不应被
`speech-to-speech` 直接 import。

codec backend 的 capability 由 anytrain 统一提供；本仓库的 `load_backend(BackendConfig, device)` 是严格
配置边界：普通 backend 转交 `anytrain.codec.load_semantic_acoustic()`，BiCodec 则显式传递其模型目录、
revision 和本地加载策略。它不接受松散 mapping，也不增加只转发属性和方法的 codec wrapper。固定长度
acoustic units 的训练适配属于本仓库的 batch/generator contract，不属于 codec loader。

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
2. dataset 在 `__getitem__` 中按 source order 为每个 target 懒加载并确定性选择同 speaker、不同 utterance
   和文本的 reference；构造 dataset 不读取或缓存全量 codec tensor。
3. datamodule 分别 padding target/reference 的 semantic 与 acoustic 轴，并保留 pair metadata。
4. target acoustic units 只构造 FM continuous target 或 RVQ labels；reference acoustic units 只构造
   reference condition，不能把 target features 回流到 condition。
5. 训练逐样本以 `reference_dropout=0.5` 在显式 reference 与 learned null condition 之间切换。

单侧 `qwen_fixed_speaker` 仍作为显式 smoke source 保留：它通过
`zhuyin.datasets.wmt19.qwen_tts` 从统一 Qwen codec grid 选择一个 role/speaker 列，codec view
与网格的行、列、整网格访问契约保持一致，不再依赖 `zhuyin.datasets.wmt19_tts` 或独立的固定 speaker
prepared dataset。默认训练数据契约仍是 `qwen_cross_text`。

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
2. projection：仅在 codec embedding dim 和 decoder condition dim 不一致时做线性投影。

semantic condition 层只负责 semantic 表示，不读取 acoustic target，也不构造 text/audio vocabulary head。
初始化策略由枚举控制，例如 codec initialization 和 matched random initialization，字符串只在配置边界解析一次。

reference condition 是独立支路：只读取 paired cross-text reference acoustic features，masked pooling 后映射到
condition space，作为全局 speaker/reference 条件广播到 target semantic 时间轴。这里的 pooling 只发生在
reference 支路；target semantic memory 不做 pooling。无 reference 或逐样本 dropout 时直接使用 condition-space
的 learned null condition，不先伪造 feature-space reference。

## 两阶段生成器复用

semantic-only 主线包含两个跨仓库阶段，不等同于 `speech-to-speech` 的 Stage 0-4 数据/任务日程：

```text
Phase A: semantic codes -> SAC conditioner -> CodecUnitGenerator -> acoustic target
Phase B: aligned S2S hidden state -> HiddenConditionAdapter -> initialized generator -> acoustic target
```

Phase A 由本仓库训练完整的 `SemanticCodecSupport`。Phase B 只复用其中的 generator、decoder 配置、
condition dimension、feature normalization 和 acoustic backend metadata；semantic conditioner 与 reference
conditioner 不进入 S2S model。S2S 自己拥有 hidden-state 对齐和 `LayerNorm + Linear` adapter，本仓库不读取
Qwen hidden state，也不 import S2S。

同一 artifact 因此有两个显式加载视图：

- `load_artifact()` 返回完整 `SemanticCodecSupport`，用于 semantic-only waveform runtime；
  `SemanticCodecRuntime` 对 semantic vocab/embedding 与 acoustic metadata 做完整 backend 校验。
- `load_generator_artifact()` 返回 `AcousticGeneratorArtifact(generator, spec)`，用于外部 condition producer；
  `validate_acoustic_backend()` 只校验 acoustic feature dimension、codebook sizes、layout 与 unit length。

`AcousticGeneratorSpec` 仍记录 semantic vocab/embedding 作为 artifact 来源信息，但 Phase B 不用 semantic codes，
因此它们不是 generator 初始化的兼容条件。route、condition dimension、decoder topology、REPA 配置和 acoustic
metadata 仍必须严格匹配，不能由调用方静默覆盖。

`save_artifact(path, support, backend=...)` 只从 `support.config` 序列化构造配置，不接受第二份调用方 config。
schema 7 的已声明字段必须完整且类型精确；loader 不补默认值、不做字符串/布尔数值强转，并先在 CPU 上以
`weights_only=True` 加载 state dict，再把已构造模块移动到目标设备。

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

FM 的 feature mean/std 在正式训练开始前流式遍历有效训练 subset 计算，使用 float64 sum/square sum，
不会从 `sample_index` 或首个 batch 推断全数据分布；fixed-batch overfit 则显式只统计该固定 batch。
`datamodule.validation_split` 可绑定独立 held-out split，validation loader 固定顺序且不使用动态
batching。训练 loader 通过 anydataset map-style `dataloader`，先按 manifest audio duration 推导的
index-level frame proxy 规划 batch，再由 worker 物化 codec Tensor；planner 不反序列化 codec payload，
也不缓存已物化 pair 或依赖归档 batching adapter。验证对同一 target 用相同 seed 的独立 generator 计算
with-reference / without-reference：FM 记录 feature MSE，RVQ 记录 code error，并记录两者差值作为
reference gain。

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
  `SemanticAcousticCodec`、route-specific sampler capability、`AcousticGeneratorArtifact` 和 artifact loading API
  依赖本仓库。

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
