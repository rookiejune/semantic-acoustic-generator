# 012 LongCat First-Codebook Factor Objective

## 目标

011 已排除“local anchor 感受野太窄”作为主瓶颈，但当前 16-D anchor 同时受 normalized MSE、cosine 和
tied-codebook CE 约束。对 semantic-only 的多模态 target，MSE 会偏向条件均值；即使 snap 到最近码本，
也可能落到错误且时间不连贯的 factor。012 用严格逐帧、非自回归的离散分类诊断区分目标参数化问题与
semantic-only 信息上限。

## A. Parallel Factor Classifier

- 保持第一 acoustic codebook 分解得到的两个 90-way factor target。
- 保持 local frame-preserving context、10k 数据、batch 16、2k steps、无 reference 和相同 optimizer。
- 输出改为同索引 frame 的 `2 x 90` logits，只训练 factor CE；不回归 16-D feature，不使用 MSE/cosine。
- inference 对两个 factor 取 argmax，查原始 LongCat factor codebook embedding，拼成 16-D 后走现有
  stage-0 projection/decode。不得 stride、pooling、插值、长度预测或时间自回归移位。

先跑 128 条 overfit gate；无法显著超过 011 的 train factor accuracy 时不进入 10k。overfit 通过后跑
10k/2k，并复用 train-128/held-out-16 的 feature/factor、Whisper 与 UTMOS evaluator。

## 判定

- train/held-out factor accuracy 与 snap UTMOS 同时提高：连续回归的条件均值冲突是主瓶颈，继续研究
  factor sampling/temporal prior。
- train accuracy 提高但 held-out 不提高：数据覆盖或语义条件泛化是主瓶颈。
- train accuracy 仍低：停止第一码本精确预测路线，semantic-only 条件或当前 conditioner 无法提供足够信息。
- factor accuracy 提高但 UTMOS 不提高：逐帧 top-1 不是正确感知目标，下一步只考虑保持 frame 对齐的
  概率生成与序列先验，不再增加 deterministic regression 容量。

默认 `feature_adapter=none`、`fm_mode=flow`、`anchor_context=local` 不变；012 只在显式 experiment 中启用。
