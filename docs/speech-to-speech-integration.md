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

- semantic-only codec 是通用音频 codec 能力，不应绑定某个 S2S token model。
- `speech-to-speech` 只需要 semantic-only support artifact 和可选训练 checkpoint，不需要知道本仓库的
  backend feature distillation 细节。
- 避免两个仓库互相 import 模型、runtime singleton 或 Hydra schema。

## 2. speech-to-speech 需要消费什么

`speech-to-speech` 保留负责数据和 token layout 的完整 backend codec，并额外消费一个满足
semantic-only decode contract 的对象：

```python
codec.sample_rate
codec.frame_rate
codec.decode(semantic_codes)           # -> waveform
```

如果接入为 semantic-only codec，`speech-to-speech` 应选择 acoustic side channel 为 none 的组合：

```text
semantic audio tokens -> codec.decode(semantic_codes) -> waveform
```

也就是说，S2S token model 只预测 semantic audio tokens；waveform reconstruction 由本仓库 checkpoint
完成。

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

- `semantic-acoustic-codec`：semantic codebook 初始化、semantic token/span 到 frame condition 的展开、
  reference acoustic condition、LongCat backend feature/feature normalization、Flow/RVQ acoustic decoder、
  `decode(semantic_codes)` artifact runtime 和 codec artifact export/import。
- `anytrain`：不含音频/codec 语义的通用训练积木，例如 sequence DiT backbone、attention backend、
  condition cache、通用 Qwen/MTP codebook predictor、task-agnostic flow matching runtime、MFU/性能统计。
- `speech-to-speech`：只保留 text/audio token model、task datamodule、joint S2S training stage、
  generation service、S2S-specific evaluation/logger。

判断规则是：只要需要解释 LongCat code layout、semantic/acoustic codebook、backend feature、
artifact schema 或 semantic-only waveform reconstruction，就属于本仓库；只有完全不依赖这些领域名词、
能被其它训练任务直接组合的模块，才考虑进入 `anytrain`。`anytrain` 是共享 `third_party` 组件，
迁移前需要先以独立 PR/commit 明确通用契约和测试，不能从本仓库直接复制 project-specific API。

## 5. Runtime artifact

本仓库应产出一个可独立加载的 runtime artifact：

```text
checkpoint_dir/
    model.ckpt
    codec.json
    config.json
    metrics.json
```

`codec.json` 至少记录：

- route：`rvq` / `fm`；
- backend：`longcat`；
- sample rate and frame rate；
- semantic vocab size；
- acoustic feature dim；
- decode sampling config；
- feature normalization；
- checkpoint schema version。

`speech-to-speech` runtime preset 只需要指向这个 artifact：

```yaml
runtime:
  codec: longcat
  semantic_codec_artifact: /path/to/semantic-acoustic-codec/checkpoint_dir
model/acoustic: none
```

具体字段名后续以 `speech-to-speech` runtime schema 为准，但原则是：S2S 不重建本仓库模型配置，
只加载 artifact。

## 6. 训练期与推理期差异

本仓库训练期：

```text
semantic codes + backend acoustic codes
    -> backend acoustic features
    -> train decoder
```

`speech-to-speech` 推理期：

```text
generated semantic tokens
    -> semantic codes
    -> semantic_acoustic_codec.decode()
    -> waveform
```

因此 backend acoustic codes、backend feature loss、overfit dataloader、LBA planner 都不进入
`speech-to-speech` runtime。

## 7. 接入验收

在 `speech-to-speech` 依赖本仓库前，必须先在本仓库完成：

1. `decode(semantic_codes)` 对真实 LongCat semantic codes 生成 finite waveform。
2. artifact load 后输出与保存前一致，至少同 seed 下 deterministic route 完全一致。
3. route metadata 不匹配时严格报错。
4. sample rate/frame rate 与 LongCat backend 一致。
5. 单条 TTS request 在 `speech-to-speech` 中使用 `model/acoustic=none` 完成 decode。

接入初期不应把本仓库 checkpoint 反向 import 进 S2S oracle；oracle 可以被删除、保留作历史实验，
或改成调用本仓库 decoder。
