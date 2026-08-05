# 009 LongCat FM Numerical-Stability Continuation

对应计划：[`../schedules/009_longcat_fm_numerical_stability.md`](../schedules/009_longcat_fm_numerical_stability.md)

## 训练结果

- 65k checkpoint 以 `1e-4` 正常 resume，训练到 `max_steps=100000` 后由 Lightning 正常停止。
- 70k、80k、90k、100k checkpoint 均完整写出，100k runtime artifact 导出成功。
- 日志中没有 traceback、OOM、non-finite loss 或异常退出；训练进程结束后 GPU 正常释放。
- 本次只证明当前较低学习率 continuation 的数值稳定性，不证明继续增加训练步数必然改善质量。

## Held-out 数据修复

16 条 held-out LongCat store 缺少 `TextMeta.LANG`，新版 workspace loader 按契约拒绝读取。评估前在私有
debug/staging 中复制 store，并运行 workspace 的 manifest-only migration：

- dry run 检出 16 个待补 text item、0 个冲突；
- 正式迁移更新 16 个 text item；
- 原 `samples.parquet` 已备份；
- codec payload 未解码、未重编码，正式 store 未修改。

## Fixed eval 完整性

- 65k 与 100k 使用同一组 16 条 held-out cross-text pairs、同一 seed 和 sampling 配置。
- 每个 checkpoint 均产出 16 份 metrics，以及 without-reference、with-reference、target reconstruction、
  reference reconstruction 四组 WAV。
- 100k 共 16/16 条 metrics、64/64 个 WAV，所有 waveform 均 finite；评估日志无异常。

## 65k→100k 对照

| 指标 | 65k | 100k | 变化 |
| --- | ---: | ---: | ---: |
| without-reference feature MSE mean | 1.9200 | 1.8867 | -0.0333（-1.73%） |
| with-reference feature MSE mean | 1.9199 | 1.8864 | -0.0335（-1.75%） |
| without-reference ASR target distance mean | 0.4889 | 0.4607 | -0.0283 |
| with-reference ASR target distance mean | 0.4941 | 0.4576 | -0.0365 |

两路 feature MSE 都在 14/16 条样本上改善。离线 ASR 的 target/reference reconstruction 校准距离均约
0.030，说明同一 evaluator 能正确识别 codec reconstruction；因此生成音频约 0.46 的 target distance 主要
反映 generator 内容保持仍明显弱于 full-unit reconstruction，而不是 ASR 完全失效。

## Reference condition 与泄漏检查

- 100k reference gain mean 为 `0.00030`，median 为 `0.00022`，只有 9/16 条为正；相对约 `1.89` 的
  feature MSE，这个收益很小且不稳定。
- 65k→100k 的 reference gain 只在 8/16 条样本上改善，说明续跑主要改善共同 reconstruction objective，
  没有形成可靠的 reference-specific 增益。
- 100k with-reference / without-reference waveform 仍有变化，但平均相关系数约 `0.987`；reference 并非被
  完全忽略，变化却没有稳定转化为更低 target feature error。
- 离线 ASR 中，with-reference 和 without-reference 都是 0/16 条更接近 reference 文本；提供 reference 后
  target distance 平均只改善约 `0.003`。当前 fixed set 没有发现 reference 文本泄漏，也没有证明 reference
  能稳定改善内容保持。

ASR 检查只覆盖英文 held-out 文本和字符距离，不替代更大样本集或主观音质/音色听评。

## 决策

**当前配置停在 100k，不继续到 200k。**

理由：65k→100k 的数值稳定性和共同 reconstruction 指标有小幅、较一致改善，但生成内容距离 codec
reconstruction 仍有明显差距，reference gain 接近零且正负混杂。继续使用同一 objective 堆训练步数，更可能
继续缓慢改善共同 reconstruction，而不是解决 reference conditioning 的核心问题。

如果后续重启该方向，优先重新设计 reference objective、condition 注入或配对监督，再用同一 fixed-eval
协议比较；不要直接从 100k 无条件续跑到 200k。

## 训练集 128-step / requantize 诊断

为区分 held-out 泛化、flow 积分步数和 continuous latent 偏离 codebook manifold 三种可能原因，使用同一
100k artifact 对训练集合内四条样本做追加诊断。四条样本分布在训练列的不同位置，时长分别为
3.96、4.92、6.36 和 17.76 秒；评估固定无 reference、seed 0，并把 flow sampling 从 artifact 默认的
16 steps 提高到 128 steps。

每条样本只生成一次 continuous acoustic features，再从完全相同的 features 导出两路 waveform：

1. raw：直接把 FM features 交给 LongCat decoder；
2. requantized：先执行 latent → 3 个 acoustic codebooks → latent，再交给同一 decoder。

| 训练样本 | Raw feature MSE | Requantized feature MSE | Raw → requantized MSE | Waveform correlation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.7104 | 1.7438 | 0.2792 | 0.5392 |
| 2 | 1.7097 | 1.7420 | 0.2320 | 0.7771 |
| 3 | 1.7473 | 1.7676 | 0.2689 | 0.7507 |
| 4 | 1.8800 | 1.8614 | 0.2732 | 0.6533 |

requantized codes 与训练 target codes 的逐 codebook accuracy 仅为：

- codebook 0：`0.0%`--`3.04%`；
- codebook 1：`0.94%`--`2.44%`；
- codebook 2：`0.0%`--`0.94%`。

人工盲听 raw / requantized A/B 与 full-unit target reconstruction 后得到：

- 四条 full-unit target reconstruction 的声学质量正常；其中一条原始 Qwen TTS 本身存在不自然的数字读法，
  LongCat reconstruction 与数据源一致，因此不能用该内容现象解释 generator 的整体失败；
- 四条训练样本的 raw 与 requantized generation 均未达到可用、可懂的语音质量；
- requantize 没有产生一致听感收益，并在代表样本上明显劣于 raw；其 feature MSE 也只在最长样本上小幅改善。

这组结果排除了“主要只是 held-out 泛化不足”“16-step flow 积分过粗”和“简单 requantize 即可修复
off-manifold latent”三种解释。模型能够生成 finite、长度和能量范围合理的 waveform，说明它学到了一定的
acoustic prior 与粗粒度统计；但即使在训练样本上，也没有学到足以支持可懂语音重建的
semantic-conditioned acoustic mapping。

因此，对当前 100k artifact 的质量结论是：**LongCat-FM 当前训练配置失败，不应继续使用相同
objective、架构和数据配置堆训练步数。** 后续讨论应先重新界定 target representation、semantic condition
利用方式和训练 objective，再建立新的实验计划。

## Anytrain 整段可懂度与质量评估

对同一组 16 条 held-out 整段 WAV 使用 anytrain evaluator 重新评估：ASR 固定为
OpenAI Whisper `large-v3`、English、temperature 0；文本比较忽略大小写和标点；质量使用
`tarepan/SpeechMOS:v1.2.0` 的 `utmos22_strong`。65k 与 100k 的 target/reference reconstruction
逐文件完全一致，因此 reconstruction 只评估一次。

| Checkpoint / route | BLEU | WER | chrF | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| codec target reconstruction | 98.20 | 0.0106 | 99.73 | 4.269 |
| 65k without reference | 34.75 | 0.5718 | 60.46 | 1.380 |
| 65k with reference | 35.33 | 0.5612 | 60.11 | 1.374 |
| 100k without reference | 41.99 | 0.4681 | 64.01 | 1.357 |
| 100k with reference | 40.60 | 0.5479 | 59.79 | 1.362 |

这组整段指标补充了三个结论：

- 65k→100k 的 without-reference WER 改善 `0.1037`，说明 FM 继续训练确实增加了一部分内容映射能力；
  但 UTMOS 同时下降 `0.0228`，不能把该 continuation 描述为整体语音质量提升。
- 100k 提供 reference 后 WER 反而增加 `0.0798`、chrF 下降 `4.22`，UTMOS 只增加 `0.0050`；
  reference condition 没有形成可用收益。
- generator 与 codec 可达上界仍相差约 `+0.457` WER 和 `-2.912` UTMOS。瓶颈不在 LongCat decoder
  或 evaluator，而在 semantic-conditioned acoustic generation。

## Semantic perturbation 与 flow-time 诊断

使用 100k artifact、相同初始噪声和原始 target semantic decode，比较 acoustic generator 收到的三种
condition：正确逐帧 semantic、同一句内随机打乱 semantic frame、semantic 置零。正确路线复现已有 fixed-eval
WAV 的平均相关系数大于 `0.999999`，最大逐采样差小于 `6.2e-5`。

| Generator condition | Feature MSE | BLEU | WER | chrF | UTMOS |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct semantic | 1.8867 | 42.15 | 0.4681 | 64.34 | 1.357 |
| shuffled semantic | 2.2404 | 0.07 | 1.7021 | 3.47 | 1.247 |
| zero semantic | 2.2113 | 0.09 | 1.4069 | 5.58 | 1.236 |

即使 LongCat decoder 仍接收原始 target semantic，只破坏 acoustic generator 的 semantic condition 也会使
内容基本崩溃。这排除了“generator 完全忽略 semantic、可懂度只来自 codec decoder semantic 输入”的解释；
当前 semantic condition 是必要的，但学到的 mapping 仍不足以产生高保真 acoustic features。

固定同一 target、noise 和 `x_t` 后，直接比较 velocity prediction 的 masked MSE：

| flow time | correct | shuffled | zero | shuffled - correct | zero - correct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.9432 | 1.1311 | 1.0517 | +0.1879 | +0.1085 |
| 0.20 | 0.4222 | 0.5041 | 0.4652 | +0.0819 | +0.0430 |
| 0.50 | 0.0219 | 0.0230 | 0.0226 | +0.0010 | +0.0006 |
| 0.80 | 0.0193 | 0.0203 | 0.0193 | +0.0010 | +0.0000 |
| 0.95 | 0.0707 | 0.0756 | 0.0694 | +0.0049 | -0.0013 |

正确 semantic 的优势主要集中在 noise 端；到 flow 中段后，正确、打乱和置零 condition 的误差几乎收敛。
因此当前模型不是完全没有 semantic 信号，而是没有让严格逐帧对齐约束贯穿整条生成轨迹：早期 semantic
提供粗内容锚点，中后段主要依赖当前 acoustic state 和 learned acoustic prior。

## 模型设计与训练通路归因

1. **数据对齐没有丢。** LongCat semantic/acoustic 都是 `16.67 Hz` frame-aligned，batch contract 也要求两者
   共享 frame 轴；semantic condition 以 frame FiLM 注入 DiT。问题不是数据 loader 做了错误的时间重采样。
2. **FM 目标允许中后段 condition shortcut。** 当前直接对归一化后的 1024 维 acoustic latent 做 Gaussian
   noise→target velocity MSE，并均匀采样 flow time。随着 `t` 增大，`x_t` 自身已包含越来越多 target acoustic
   信息；实测模型在中后段几乎不再区分正确和错误 semantic，训练目标也没有额外的 aligned content loss
   强迫每个 frame 持续服从 semantic。
3. **semantic 与 target 的一对多部分全部交给同一个 FM。** acoustic latent 同时包含内容、音色、韵律和
   codebook 组合细节。单一 velocity MSE 要同时学 deterministic aligned content mapping 和 stochastic residual，
   容易先得到 acoustic prior 与平均轨迹，却长期保留较高的内容错误。
4. **reference 分支在当前数据上缺少可识别监督。** `qwen_cross_text` 固定 `speaker_id=vivian`，target/reference
   又要求同 speaker、不同文本；reference dropout 为 0.5。模型不需要读取 reference 就能知道唯一 speaker，
   而不同文本 reference 的局部韵律也不是 target 的监督信号，所以忽略 reference 是当前训练分布下的近优解。
5. **reference 表示和尺度进一步削弱了支路。** 整段 reference acoustic features 只做 masked mean pooling，
   再经零初始化 gate 后广播并与 semantic 相加。100k gate 平均绝对值只有 `0.0106`；实测 paired-reference
   condition RMS 为 `0.00080`，只有 semantic RMS `0.03052` 的约 `2.6%`，不足以稳定改变生成结果。
6. **当前 RVQ 失败不推翻 FM 相对优势。** 当前 RVQ 对三个 8100-way codebook 做 temporal causal teacher
   forcing 和帧内 MTP，既引入 temporal exposure bias，又要求精确分类离散组合；它没有把严格对齐先验实现成
   简单的逐帧确定性预测。因此结论应保持为“当前 FM 优于当前 RVQ，但两者设置都不合格”，而不是切回 RVQ。

## 后续方向

保留 FM 主线，但先把严格对齐先验写进模型分解：

```text
mu_t = aligned_anchor(semantic_t, local_context)
residual_t = target_acoustic_t - mu_t
target_acoustic_t = mu_t + residual_fm(noise | semantic, reference)
```

- `aligned_anchor` 先承担可懂内容和逐帧确定性映射，用 feature regression 加内容相关验收单独证明能在
  1/16/128 样本上过拟合；FM 只拟合 anchor 未覆盖的 acoustic residual。
- reference 若要承担 speaker/style 控制，训练集必须包含多 speaker counterfactual，并用 speaker/style
  一致性目标建立可识别监督；condition 应作为独立 AdaLN/cross-attention 支路，而不是小尺度向量直接加到
  semantic 上。
- 新方案的最小门禁应同时包含 anchor feature MSE、整段 Whisper WER/chrF、UTMOS、semantic shuffle
  margin 和分 flow-time condition margin。未过这些门禁前不继续大步数训练。
