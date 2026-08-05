# 013 LongCat Factor Classifier Scaling

## 目标

012 已证明第一 acoustic codebook 的严格逐帧 `2 x 90` factor CE 明显优于 16-D continuous anchor，且
生成语音的 WER/UTMOS 已接近第一码本可达上界。013 不再保留 FM 作为主路线，直接验证相同的因子分类能否
扩展到 LongCat 前 2/3 个 acoustic codebooks。

LongCat 每个 acoustic stage 都由两个 `90 x 8` factor codebook 组成；每个 stage 的两个投影拼成
decoder-ready 1024-D 表示，多个 stage 的表示在 decoder 前逐项求和。因此 N-stage 目标严格保持 frame
轴不变，模型每帧输出 `2N x 90` logits，查表得到 `N x 16-D`，再执行原 codec 的 stage projections。

## 对照

- N=1：复用 012 的 10k/2k artifact 作为已完成 baseline，不重复训练
- N=2：`feature_adapter=longcat_codebooks`、`feature_codebooks=2`，纯 factor CE
- N=3：同一配置，`feature_codebooks=3`
- 所有路线保持 local frame-preserving anchor、condition_dim/hidden_dim/layers、optimizer、batch 16、
  10k samples、2k steps、无 reference；不使用 FM、MSE/cosine、stride、pooling、插值、长度预测或时间
  自回归移位

## 门禁与评估

1. N=3 真数据 16-sample/2-step smoke，确认六个 logits、48-D target、三段 projection 求和和 artifact
   roundtrip
2. N=3 128-sample/2k overfit；若每个 factor 的 train top-1 能明显脱离 `1/90` 且 loss finite，再进入
   10k。N=2 与 N=3 共用同一个实现门禁
3. N=2/N=3 各完成 10k/2k，固定评估 train-128 与 heldout-16：每个 factor accuracy、raw/snap feature
   MSE、selected-codebook oracle WER/UTMOS、generated WER/UTMOS、full-target reconstruction 上界
4. 结果以整段 Whisper WER 和 UTMOS 为主；exact factor accuracy 只做结构诊断。selected-codebook oracle
   用于区分“更多 codebook 本身带来的 decoder 上界”与“semantic-to-acoustic 预测能力”

## 判定

- N 从 1 到 2/3 时 generated UTMOS 单调提升且 WER 不恶化：直接因子分类可以替代 FM，继续做更多
  codebook 的预算/概率解码优化
- factor accuracy 提升但 UTMOS 不提升：目标容量已扩展，但需要 factor-level perceptual/temporal prior，
  不回到全维 FM
- N=2/3 在 train 和 heldout 都接近随机或明显 OOM：停止盲目扩大 codebook 数，先缩小 head、做渐进式
  codebook dropout 或分阶段训练

默认全局边界不变：`feature_adapter=none`、`fm_mode=flow`；013 只在显式 experiment 中启用。
