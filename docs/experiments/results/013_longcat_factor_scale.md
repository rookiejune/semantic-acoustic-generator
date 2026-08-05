# 013 LongCat Factor Classifier Scaling

对应计划：`docs/experiments/schedules/013_longcat_factor_scale.md`。

## 实现与本地门禁

- code: `d260a71`；artifact device-move fix: `60b6f99`；speech evaluator PCM16 fallback:
  `0c14571`
- contract: `feature_adapter=longcat_codebooks` 配合显式 `feature_codebooks=N`；每帧输出 `2N x 90`
  logits；argmax 后查每个 stage 的两个原始 `90 x 8` factor codebook；每个 stage 的两路 projection 拼成
  1024-D，N 个 stage projection 按原 LongCat `from_codes` 契约逐项求和
- alignment: semantic/acoustic frame 数严格相同；没有 stride、pooling、插值、长度预测或时间自回归移位
- compatibility: 默认仍为 `feature_adapter=none`、`feature_codebooks=1`、`fm_mode=flow`；旧
  `longcat_first_codebook` 明确只允许 N=1；早期 artifact 缺少新字段时默认 N=1
- artifact: N=3 保存六个 factor buffers；首次真实 artifact reload 暴露已注册 buffer 在 `.to(cuda)` 后缓存
  tuple 仍指向 CPU tensor，`60b6f99` 改为按稳定 buffer 名称读取当前设备 tensor并增加 dtype/device-move
  回归测试
- local: ruff、basedpyright、Hydra composition、wrapper `bash -n` 和全量 `233 passed`

## N=3 smoke 与 overfit gate

- smoke: 145 GPU0；16 samples/batch 16/2 steps；六因子起始平均 CE `4.533`，符合 `ln(90)`；
  14.9M trainable parameters；artifact 恢复得到 N=3、48-D feature contract 和六个 `90 x 8` buffers
- artifact decode: 修复后真实 LongCat reload、三段 projection sum、selected-codebook reconstruction、raw/snap
  整段 WAV 和六因子指标导出通过。145 缺 `openai-whisper`；最终 WER/UTMOS 在 125 运行。125 的
  `torch 2.12` / torchcodec 组合缺 CUDA 13 动态库，`0c14571` 为本脚本写出的 PCM16 WAV 增加标准库
  fallback，不改变 Anytrain Whisper/UTMOS evaluator
- overfit: 145 GPU0；128 samples/batch 16/2k steps；最终平均 factor CE 约 `0.007`
- fixed train-128 factor accuracy: stage 0 A/B `0.99754/0.99752`；stage 1 A/B
  `0.99744/0.99749`；stage 2 A/B `0.99744/0.99739`；48-D raw=snap feature MSE `0.14172`
- decision: N=3 六因子可以在相同主干容量下共同收敛，进入 N=2/N=3 10k/2k 泛化对照

## 10k/2k

- status: completed
- host / GPUs: 145 GPU5=N2，GPU6=N3
- sample_limit / batch_size / max_steps: `10000 / 16 / 2000`
- output: `$SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT/013/factor-{2,3}cb-10k-2000`
- runtime: 两路 peak allocated/reserved 约 `3.86/4.36 GB`，训练 step-time window 约 `88-90 ms`；
  10k store 初始化/取数占主要 wall time，不是 factor head 计算
- final online N2: 平均 CE `3.53298`；stage 0 A/B top-1 `0.18634/0.16306`，stage 1
  `0.10194/0.08783`
- final online N3: 平均 CE `3.70545`；stage 0 A/B top-1 `0.19225/0.18616`，stage 1
  `0.10646/0.10008`，stage 2 `0.05371/0.08333`

## Fixed train-128

| Route | Stage 0 A/B | Stage 1 A/B | Stage 2 A/B | Feature MSE |
| --- | --- | --- | --- | ---: |
| N2 | `0.19528/0.16501` | `0.10432/0.09082` | - | 52.52368 |
| N3 | `0.19361/0.16825` | `0.10469/0.09378` | `0.05157/0.07933` | 45.20565 |

10k 训练集头部与 heldout 的准确率差距有限，不存在 128-sample overfit gate 那种接近 100% train、
heldout 接近随机的记忆/泛化分叉。后续 stage 在 train 上也显著更难，主要不是 heldout domain shift。

## Fixed heldout-16

| Route | Stage 0 A/B | Stage 1 A/B | Stage 2 A/B | Generated WER | Generated UTMOS |
| --- | --- | --- | --- | ---: | ---: |
| N1 / 012 | `0.17418/0.14374` | - | - | 0.02394 | 3.29026 |
| N2 | `0.17301/0.15276` | `0.08134/0.07918` | - | 0.02128 | 3.14882 |
| N3 | `0.17299/0.15231` | `0.08610/0.07900` | `0.04549/0.06613` | 0.02394 | 2.73102 |

- N2 selected-codebook oracle: WER `0.01596`，UTMOS `4.27482`
- N3 selected-codebook oracle: WER `0.01064`，UTMOS `4.26947`，与 full target
  `0.01064/4.26946` 等价
- N2/N3 raw 与 snap 逐值等价；feature MSE 分别为 `54.79969/47.47086`

## 结论

- 结构和优化上可以 scale：N=3 过拟合门禁可把六因子全部做到约 99.7%，10k 联合训练也没有损害
  第一 stage；N2/N3 heldout 的第一 stage 准确率与 N1 基本相同
- 朴素的“所有 stage 等权 CE + 每因子独立 argmax + decoder projection sum”不能在感知质量上 scale：
  N2 WER 略好但 UTMOS 下降 `0.14`，N3 WER 无改善且 UTMOS 下降 `0.56`
- oracle 证明前两 stage 已有接近完整 codec 的质量上界，前三 stage 等同完整 target；瓶颈不是 decoder
  容量，而是 semantic-only 对后续高熵 acoustic factors 的可预测性，以及错误后续 stage 被无条件加到
  decoder 输入
- exact feature MSE 从 N2 到 N3 下降，但 UTMOS 反而显著下降，再次说明 embedding MSE/exact code
  accuracy 不能替代整段感知指标
- decision: 当前不回到 FM，也不继续盲目增加码本；保留 N1 factor classifier 作为稳定基线。若继续，
  优先验证渐进式 stage 训练、codebook dropout/置信度门控或以已预测前级为条件的残差分类，让不可靠的
  后续 stage 可以退化为 0，而不是无条件污染第一 stage 的可用生成
