# 011 LongCat First-Codebook Anchor Scale

## 目标

010 已验证第一 acoustic codebook 的 16-D factor space 是有效 target，局部 aligned anchor 在训练集可精确
预测两个 factor，并在 held-out 保住内容；但 128 条训练后的 held-out factor A/B accuracy 只有
`0.0798/0.0763`，snap UTMOS 只有 `1.8477`。本实验先区分这是训练数据多样性不足，还是 9-frame 局部
感受野不足。严格逐帧对齐、第一码本 target、loss、优化器和 2k 更新预算保持不变。

默认配置继续保持 `model.feature_adapter=none`、`model.decoder.fm_mode=flow`；011 只在显式 experiment 中
启用 first-codebook anchor。

## A. 数据规模单变量

入口：`jobs/011/01_anchor_scale.sh`，配置：`configs/experiment/011_longcat_anchor_scale.yaml`。

- train：固定说话人的完整 10k store，batch size 16，2k steps；不使用 reference。
- model：condition/anchor hidden 512，4 个 kernel-3 local blocks，感受野 9 frames。
- loss：与 010 相同的 normalized MSE、cosine 和两个 90-way tied-codebook factor CE。
- decode：同一输出同时评估 continuous raw 与 cosine argmax snap。
- eval：训练集固定 128 条，加与 009/010 相同的独立 held-out 16 条整段语音。

与 010 anchor-128 基线比较 factor A/B accuracy、raw/snap feature MSE、Whisper WER/chrF、UTMOS、step
time 和显存。只有 held-out factor accuracy 与 snap UTMOS 同时提高，才把收益归因于数据规模；训练集指标
本身不作为泛化门禁。

若 2k 时 train-128 与 held-out factor accuracy 同时较低且接近，表示模型整体欠拟合，不把低 UTMOS 归因
于 local context。此时从完整 optimizer checkpoint 原样续到 5k，再重复同一评估；只有 train 指标已明显
收敛而 held-out 仍无收益，才进入 B。

## B. Frame-Preserving Transformer Anchor

仅当 A 的训练 loss/accuracy 正常而 held-out 没有实质改善时实现。用保持 `[batch, frame]` 轴不变的
Transformer encoder 替换 local Conv anchor：不做 stride、pooling、插值、长度预测或自回归移位；每个输出
仍对应同索引 semantic frame。先在与 A 相同的 10k/2k 预算下比较，再决定是否延长训练。

Transformer 仍输出 16-D，并使用同一 tied-codebook factor CE 和 snap decode。该门禁失败前不增加独立
`2 x 90` logits head，避免同时改变上下文范围和分类器参数化。

## C. Reference 后置

现有 cross-text reference 是同说话人、不同文本的全局池化条件，只能提供音色/全局风格，不能提供目标句
韵律；当前数据又固定为 Vivian，因此先不加入。A/B 仍不足且确认多说话人或音色保持是主要缺口后，再用
paired source 单独做 reference ablation。

## 停止条件

- A 若 2k 时 train/held-out 同时欠拟合，先原样续到 5k。
- A 若显著改善 held-out factor accuracy 与 UTMOS，继续扩大训练预算，不实现 Transformer。
- A 若 train 已明显改善而 held-out 不改善，进入 B。
- B 若仍不能改善 snap UTMOS，停止第一码本精确预测路线，重新评估 semantic-only 条件的信息上限。
- 任一路线 WER 明显退化，停止扩大。

实验路径、checkpoint、WAV、转写和逐样本明细留在 debug 或制品存储；schedule/result 可以提交
复现入口、汇总指标和结论。
