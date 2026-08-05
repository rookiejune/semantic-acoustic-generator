# 014 LongCat Factor Depth Autoregression

## 目标

013 已验证同一逐帧主干可以同时优化 LongCat 前 2/3 个 acoustic stages，但把所有 factors 等权并行预测
时，后级 stage 的错误会作为完整 decoder residual 直接叠加，N=2/3 没有带来稳定的音质收益。本实验验证
失败是否来自目标因子分解遗漏了 LongCat 原始 residual quantization 的 depth dependency，而不是因子分类
本身不能扩展。

LongCat acoustic stage 按深度执行 residual quantization：stage 0 量化原始 acoustic latent，stage 1 量化
减去 stage 0 reconstruction 后的 residual。每个 stage 内 A/B 两个 factor 则对同一个 residual 并行搜索，
不构成 A 到 B 的自回归关系。因此 014 只沿 acoustic stage 深度条件化，在每个 semantic frame 内执行
`stage0 -> stage1`；frame 轴、mask 和输出长度保持不变。

## 实验单元

- baseline-N1：复用 012 的第一 stage factor classifier，不重复训练
- baseline-N2-parallel：复用 013 的两 stage 并行 factor classifier，不重复训练
- treatment-N2-depth：`feature_codebooks=2`、`factor_predictor=depth_ar`，stage 0 和 stage 1 各自同时
  输出两个 90-way factor logits；stage 1 条件于同帧 stage 0
- 训练时使用 ground-truth stage 0 factors 构造 stage 1 条件，推理时使用 stage 0 prediction；首轮不加入
  scheduled sampling、confidence gate、stage dropout 或额外感知损失，避免混淆 depth conditioning 的收益
- depth core 固定为 hidden 512、2 layers、8 heads、FFN ratio 4；逐帧 context trunk 延续 013 的 local
  anchor hidden 512、4 layers、kernel 3
- 数据与优化预算固定为 10k samples、batch 16、2k steps、无 reference、纯 factor CE；job 环境变量和末尾
  Hydra 参数允许 smoke、overfit 与诊断 override

模型仍从兼容边界选择 `route=fm`、`fm_mode=anchor`，但该实验不执行 flow matching。默认全局配置继续保持
`feature_adapter=none`、`fm_mode=flow`、parallel factor predictor；014 仅在显式 experiment 中启用。

## 门禁

1. 本地 Hydra composition、ruff/basedpyright、相关单测和 `bash -n` 通过；确认 artifact roundtrip 保存
   depth predictor 结构和全部四个 factor codebook buffers
2. 真实 LongCat 16-sample/2-step smoke：四路 CE finite，输出 shape 为 `[B,T,4]` codes / `[B,T,32]`
   features，projection sum 与原生前两 stage reconstruction 一致
3. 128-sample/2k overfit gate：stage 0/1 的 A/B teacher-forced top-1 均应接近 `99%`；同时报告
   free-running top-1 与 teacher-forced gap，并验证两条路径都保持 frame/mask 契约
4. 通过后运行 N=2 10k/2k，并在同一 fixed train-128 与 heldout-16 上完成 factor accuracy、feature MSE、
   selected-codebook oracle、Whisper WER 和 UTMOS 评估

## 判定

- 主判据：N2-depth generated UTMOS 高于 N2-parallel，并至少恢复到 N1 baseline；WER 不明显恶化
- 结构诊断：stage 1 A/B accuracy 相比 N2-parallel 提升，同时 stage 0 不下降；否则不能把变化归因于正确的
  residual dependency
- 若 teacher-forced stage 1 明显改善但 free-running 音质无收益，下一实验只加入 scheduled sampling 或
  previous-stage token dropout，暂不增加 N=3
- 若 free-running N=2 有收益，再扩展到 N=3；stage 2 条件于同帧 stage 0/1 prefix，仍不引入时间自回归
- 若 factor accuracy 提升而 UTMOS 仍下降，再单独验证 projection 后的 prefix confidence gate；不得通过把
  16-D factor embedding 置零来模拟零 residual，因为 LongCat output projection 含非零 bias

## 入口与监控

- entry：`jobs/014/01_factor_depth.sh`
- experiment：`configs/experiment/014_longcat_factor_depth.yaml`
- output：`$SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT/014/depth-ar-2cb-<samples>-<steps>`
- monitor：TensorBoard loss/per-factor top-1、step time、peak allocated/reserved VRAM、训练日志和
  `nvidia-smi`
