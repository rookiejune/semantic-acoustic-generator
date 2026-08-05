# 010 LongCat First-Codebook Aligned Anchor

## 背景与目标

009 已证明当前 LongCat-FM 比当前 RVQ 设置更好，但二者都没有达到可懂、可用语音；继续沿原始 1024-D
全 acoustic latent 和同一 velocity MSE 堆步数不成立。本实验把 LongCat semantic/acoustic 严格逐帧对齐的
先验写进模型，并先验证用户已观察到的“第一 acoustic codebook 单独解码质量足够可用”。

目标不是证明 16-D 一定优于 1024-D，而是按门禁回答：

1. 第一 codebook 的两个 90-way、各 8-D embedding 是否是可用且稳定的生成 target；
2. 严格对齐的 deterministic anchor 是否能先学会可懂内容；
3. 只有 anchor 通过后，residual FM 是否能补充声学质量而不破坏内容；
4. 映射成立后，模型深度、宽度和 flow steps 能否显著下降。

默认配置仍保持 `model.feature_adapter=none`、`model.decoder.fm_mode=flow`。010 显式启用实验路径，未通过
整段指标前不改变默认值。

## A. Representation oracle

入口：`jobs/010/01_representation_oracle.sh`。

在与 009 相同的 16 条 held-out 整段语音上导出：

1. full 3-stage codec reconstruction；
2. native stage-0 code reconstruction；
3. 精确 16-D factor embedding 经 stage-0 `out_proj_a/b` 的 reconstruction；
4. 对 16-D target 加按 codebook 各维标准差缩放的噪声，默认
   `sigma=0.05,0.1,0.2,0.5,1.0`；
5. 每个扰动同时导出 continuous raw decode 和两个 factor 分别做 cosine nearest-codebook snapping 后的
   decode。

记录：

- native stage-0 与精确 16-D projection 的 max-abs/MSE；
- raw/snap feature MSE；
- 两个 90-way factor 的 top-1 accuracy；
- Anytrain Whisper `large-v3` 的整段 BLEU/WER/chrF；
- Anytrain UTMOS；
- 同一组 WAV 的人工听评。

Oracle gate：

- native stage-0 和精确 16-D projection 数值等价；
- 精确 16-D reconstruction 的可懂度接近 full reconstruction，音质达到后续生成 target 的可用下界；
- 扰动曲线能区分 raw 与 snap 的收益区间，而不是在轻微扰动下立即整体崩溃。

若精确 stage-0 本身不可懂或 UTMOS/ASR 明显不够，停止 anchor 训练，默认 adapter 保持关闭。

## B. Deterministic aligned anchor

入口：`jobs/010/02_anchor_overfit.sh`，基础配置：
`configs/experiment/010_longcat_anchor_overfit.yaml`。

模型：

```text
semantic_t -> trainable semantic embedding/projection
           -> bounded local Conv1d anchor(semantic_{t-k:t+k})
           -> mu_t in 16-D stage-0 factor space
```

每个输出 frame 与同一 semantic frame 严格一一对应；局部卷积只提供有限邻域，不引入时间重采样、全局
pooling 或自回归 exposure bias。第一版使用 condition/anchor hidden 512、4 个 local block、kernel 3；
reference 固定关闭（`reference_dropout=1.0`）。

Loss：

```text
L_anchor = MSE(normalized(mu), normalized(e0))
         + 0.1 * cosine_distance(mu, e0)
         + 0.1 * mean(
             CE(cosine(mu_a, C_a) / 0.1, a_id),
             CE(cosine(mu_b, C_b) / 0.1, b_id)
           )
```

这里是两个 90-way factor classification，不是三个 8100-way temporal RVQ AR。

按以下顺序运行，前一级失败则不扩大：

1. 1 utterance，默认 2k steps；
2. 16 utterances；
3. 128 utterances；
4. 同一 16 条 held-out 整段 fixed eval。

每一级记录 anchor feature MSE/cosine、factor A/B top-1、raw/snap decode、Whisper WER/chrF、UTMOS 和训练
吞吐。1-sample gate 要求训练样本整段 raw 或 snap 至少一条达到稳定可懂；16/128 gate 要求训练集内容保持
随规模扩大不发生系统性崩溃。未通过 1/16/128 overfit 不进入长跑或泛化结论。

## C. Anchor + residual FM

只有 B 的 anchor 整段可懂后，才把 `model.decoder.fm_mode=residual`：

```text
r_t = normalized(e0_t) - stop_gradient(normalized(mu_t))
e0_t = denormalize(mu_t + residual_fm(noise | semantic))
```

训练总 loss 为 `L_anchor + L_flow(residual)`。stop-gradient 防止 residual objective 把已经承担内容对齐的
anchor 拉回原始 FM shortcut。先复现 1/16/128 overfit，再做 held-out；只有 UTMOS 提升且 WER/chrF 不退化
才接受 residual FM。

若 16-step residual 在 16 条训练语音上保持 WER、降低 feature MSE，但 UTMOS 没有提升，先固定权重、噪声
seed 和评估集，把 Euler steps 提高到 32/64 做一次积分误差诊断。高步数仍不能提升 UTMOS 时停止 residual
路线，不延长训练或扩大到 128；高步数有效时再区分是继续训练降低 velocity error，还是替换积分器。

## D. 性能降配

映射门禁通过后再测：

- anchor hidden `512 -> 256`，local blocks `4 -> 2`；
- residual DiT hidden `1024 -> 512`，layers `8 -> 4`；
- flow steps `16 -> 8 -> 4`；
- anchor-only 单次前向作为最低延迟路线。

每个降配必须同时报告参数量、FLOPs、step time、推理时延、显存、WER/chrF 和 UTMOS。若 anchor-only 已满足
质量门禁，优先停止使用 FM，而不是为了保留 FM 结构继续付出多步采样成本。

## 运行与记录边界

- 复旦单卡 probe 优先 144/145；启动前复查 `nvidia-smi`。
- 代码只通过本地 commit/push 和远端 `git pull --ff-only` 同步。
- held-out 路径、checkpoint、WAV、转写和逐样本明细写入 debug 或制品存储；实验 schedule/result
  可以提交复现入口、汇总指标和结论。
- 整段 artifact 评估入口为 `jobs/010/03_eval_artifact_set.sh`，同一生成 feature 同时导出 raw/snap。
