# 012 LongCat First-Codebook Factor Objective

对应计划：`docs/experiments/schedules/012_longcat_factor_objective.md`。

## 实现与本地门禁

- code: `990da8a`
- contract: strict frame-preserving local anchor；每个 semantic frame 并行输出两个 90-way logits；纯
  factor CE；推理 argmax 后查原始两个 `90 x 8` factor codebook 并拼成 16-D；不使用时间自回归、
  stride、pooling、插值或长度预测
- default_boundary: `anchor_target=feature`，默认 `feature_adapter=none`、`fm_mode=flow` 和
  `anchor_context=local` 均未改变
- artifact: 两个原始 factor codebook 作为 buffer 随 artifact 保存；加载 generator artifact 不依赖 codec
  权重来隐式重建 codebook
- local: ruff、basedpyright、Hydra composition 与全量 `225 passed`；factor CE backward、padding、原始
  embedding lookup、direct factor validation error、artifact roundtrip 和 legacy schema 默认值均有回归测试

## 真数据 2-step smoke

- status: passed
- host: `144`
- cuda_visible_devices: `2`
- sample_limit / batch_size / max_steps: `16 / 16 / 2`
- result: LongCat encoder/decoder、factor codebook 读取、bf16 CE/backward 与 artifact 导出通过；起始 CE
  约 `4.533`，符合随机 90-way 分类器的 `ln(90)`；artifact 恢复得到两个 `90 x 8` buffer 和 16-D
  feature contract
- memory: training peak allocated/reserved 约 `3.53/3.75 GB`；包含 LongCat backend 的整卡观测约
  `9 GB`，按 buffer 将任务单卡最小显存记录为 `11 GB`

## 128-sample overfit gate

- status: passed
- sample_limit / batch_size / max_steps: `128 / 16 / 2000`
- final training: CE `0.00443`；最后一批 factor A/B top-1 `0.99769/0.99808`
- fixed train-128: factor A/B accuracy `0.997685/0.997706`；raw/snap feature MSE 相同，均为
  `0.19684`。factor classifier 输出已经是原始码本 embedding，因此 raw 与 snap 路径逐值等价
- train speech: raw=snap WER `0.02374`，UTMOS `3.79909`；target reconstruction WER
  `0.02110`，UTMOS `4.34768`
- fixed heldout-16: factor A/B accuracy `0.09847/0.08562`；raw/snap feature MSE 相同，均为
  `92.40475`
- heldout speech: raw=snap WER `0.05585`，UTMOS `2.16138`；target reconstruction WER
  `0.01064`，UTMOS `4.26940`
- interpretation: 纯离散目标把 train factor accuracy 从 011 的十几个百分点提高到接近 100%，确认
  连续 16-D regression 的条件均值/目标参数化是训练瓶颈；heldout accuracy 仍与 011 Transformer 的
  `0.0959/0.0785` 同量级，说明 semantic-only 条件对 unseen 第一 acoustic codebook 的信息上限或泛化
  仍是主瓶颈。合法码本 embedding 把 heldout UTMOS 提高到 `2.16`，但没有恢复正确 factor 或 target
  级音质
- infra: 144 的系统盘 `/tmp` 和默认 pami202 output 均已满；不删除共享产物、不修改 anytrain，gate
  通过实验级 output/TMPDIR override 写到 pami201，并关闭短跑的周期 checkpoint，只保留最终 artifact

## 10k/2k

- status: completed
- host / cuda_visible_devices: `144 / 2`
- sample_limit / batch_size / max_steps: `10000 / 16 / 2000`
- invariant: 与 128 gate 相同的 local frame-preserving factor classifier、optimizer、无 reference 和纯 CE；
  只扩大训练数据
- training: 2k 正常结束并导出自包含 artifact；最后在线日志 CE 约 `3.13`，factor A/B top-1 约
  `0.177/0.165`，step-time window 约 `91 ms`
- fixed train-128: factor A/B accuracy `0.19415/0.16562`；raw/snap feature MSE 相同，均为
  `70.00482`；raw=snap WER `0.02704`，UTMOS `3.19180`；target reconstruction WER/UTMOS
  `0.02110/4.34768`
- fixed heldout-16: factor A/B accuracy `0.17418/0.14374`；raw/snap feature MSE 相同，均为
  `73.86406`；raw=snap WER `0.02394`，UTMOS `3.29026`；target reconstruction WER/UTMOS
  `0.01064/4.26940`
- comparison: train 与 heldout 的 factor accuracy、WER 和 UTMOS 接近，不存在 128 gate 那种记忆/泛化
  分叉。相比 011 同为 10k/2k 的 Transformer continuous anchor，heldout factor A/B accuracy 从
  `0.0959/0.0785` 提高到 `0.1742/0.1437`，snap UTMOS 从 `1.3127` 提高到 `3.2903`，WER 从
  `0.0559` 降到 `0.0239`。011 的 snap 也输出合法码本 embedding，因此收益不能只归因于 inference
  requantize，而来自离散 CE 目标避免 16-D 条件均值回归
- interpretation: exact factor top-1 仍低，但整段可懂度已接近 target，且音质相对 continuous anchor
  大幅恢复。这说明“逐帧精确还原录音中的唯一 acoustic code”不是合适的最终目标：semantic-only 条件无法
  唯一决定局部声学细节，而多个非 target factor 对 LongCat decoder 仍是可用的声学近邻；feature MSE 和 exact
  code accuracy 只能作为结构诊断，不能代替 WER/UTMOS 决策
- decision: 012 完成，不继续增加 deterministic classifier 的 context、深度或当前训练预算，也不切回
  16-D continuous regression。下一步先在同一逐帧 logits 上诊断 top-k、entropy 和 transition statistics，
  再验证固定输出长度的 factor sampling/temporal prior；每个 semantic frame 仍严格对应一个 acoustic frame，
  不引入 stride、插值、长度预测或时间轴漂移。只有该离散 aligned base 稳定后，再考虑把 FM 限制为其未覆盖的
  stochastic residual
