# 010 LongCat First-Codebook Aligned Anchor

对应计划：`docs/experiments/schedules/010_longcat_first_codebook_anchor.md`。

## 运行记录

### Representation oracle probe

- status: passed
- started_at: `2026-08-05 04:01 CST`
- host: `144`
- cuda_visible_devices: `2`
- entry: `jobs/010/01_representation_oracle.sh --sample-limit 1 --sigmas 0.1`
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/probe`
- purpose: 验证真实 LongCat `from_codes`、整段 WAV 导出，以及同一 manifest 的 Whisper/UTMOS evaluator
  闭环；通过后才扩大到 16 条和完整 sigma 曲线。
- numerical: native stage-0 与精确 16-D projection 的 max-abs/MSE 均为 `0`；`sigma=0.1` raw 的
  factor A/B top-1 均为 `1.0`，snap feature MSE 为 `0`。
- speech: full/stage-0/exact/raw-0.1/snap-0.1 的单样本 Whisper WER 均为 `0`；UTMOS 分别为
  `4.5551 / 4.1147 / 4.1147 / 4.0022 / 4.1147`。
- evaluator notes: 首次 Whisper 权重下载完成；UTMOS 必须显式允许固定 TorchHub repo code，并通过
  `TORCH_HOME=/mnt/pami202/zhuyin/torch` 复用已有 SpeechMOS v1.2.0 cache，避免复旦 GitHub 直连超时。

### Anchor 2-step smoke

- status: passed
- started_at: `2026-08-05 04:10 CST`
- host: `144`
- cuda_visible_devices: `3`
- pid: `3104311`
- entry: `jobs/010/02_anchor_overfit.sh`
- overrides: `sample_limit=1`、`batch_size=1`、`max_steps=2`，关闭 sample、performance、throughput、
  codebook usage 和 checkpoint callback
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/anchor-smoke-2step`
- purpose: 只验证真实数据、LongCat 16-D adapter、anchor loss、反传和 artifact 导出；不作为可懂度门禁。
- result: `max_steps=2` 正常结束；14.6M trainable parameters；artifact 的 `generator.json` 和
  `model.ckpt` 均成功导出。

### 16-utterance representation oracle

- status: passed
- started_at: `2026-08-05 04:26 CST`
- host: `144`
- cuda_visible_devices: `2`
- entry: `jobs/010/01_representation_oracle.sh`
- sample_limit: `16`
- sigmas: `0.05,0.1,0.2,0.5,1.0`
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/oracle-16`
- numerical: native stage-0 与 exact 16-D 仍逐值等价；`sigma=0.05/0.1` 两因子 top-1 近 `100%`，
  snap feature MSE 为 `0`；`sigma=0.2` 的 A/B top-1 为 `100%/99.54%`，snap feature MSE 为
  `0.0259`；`sigma=0.5` 才降至 `87.13%/72.72%`。
- speech: full codec 为 WER `0.0106` / UTMOS `4.2694`；exact 16-D/stage-0 为 WER `0.0186` /
  UTMOS `3.7657`。`sigma=0.2` raw 为 WER `0.0160` / UTMOS `3.5736`；`sigma=0.5` raw/snap
  UTMOS 为 `2.9832/3.4199`，说明 snapping 在中等扰动区间有明确收益。

### Anchor 1-utterance overfit

- status: passed
- started_at: `2026-08-05 04:29 CST`
- host: `144`
- cuda_visible_devices: `3`
- entry: `jobs/010/02_anchor_overfit.sh`
- sample_limit: `1`
- max_steps: `2000`
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/anchor-1-2k`
- final training: normalized anchor MSE 约 `0.0015`，factor CE 约 `0.532`，step time window 约
  `69 ms`，GPU peak allocated `3.20 GB`。
- artifact: raw feature MSE `0.1231`；factor A/B accuracy 均为 `1.0`；snap feature MSE 为 `0`。
- speech caveat: 样本 0 是 009 已记录的异常数字读法；full target reconstruction 自身 Whisper WER
  `1.6667`，因此绝对 WER 失去校准意义。raw/snap UTMOS 为 `2.9426/2.9493`，而 snap 与 target
  stage-0 features 逐值一致，表示过拟合门禁按 codec target 已通过。

### Anchor 16-utterance overfit

- status: passed
- started_at: `2026-08-05 04:46 CST`
- host: `144`
- cuda_visible_devices: `3`
- entry: `jobs/010/02_anchor_overfit.sh`
- sample_limit: `16`
- batch_size: `16`
- max_steps: `2000`
- first_attempt: batch padding 使用 8100 pad ID，但 factor target 在 mask 前拆分 composite code，首步以
  `LongCat stage-0 codes contain an ID outside the codebook` fail-fast；没有产生训练结果。
- fix: `_factor_targets()` 先把 acoustic mask 外的 code 置零，mask 内非法 ID 仍由 adapter 报错；
  `tests/test_feature_adapter.py` 回归测试通过，commit `6500ecf`。
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/anchor-16-2k-r1`
- final training: anchor MSE 约 `0.0020`，factor CE 约 `0.57`，step time window 约 `69 ms`，
  GPU peak allocated `3.45 GB`。
- artifact features: factor A/B accuracy `0.99978/0.99931`；raw/snap feature MSE
  `0.1530/0.0309`。
- speech: raw/snap WER `0.0412/0.0385`，full target reconstruction WER `0.0412`；raw/snap
  UTMOS `3.7501/3.7595`，target reconstruction UTMOS `4.3428`。

### Anchor 128-utterance overfit

- status: passed
- started_at: `2026-08-05 04:56 CST`
- host: `144`
- cuda_visible_devices: `3`
- entry: `jobs/010/02_anchor_overfit.sh`
- sample_limit: `128`
- batch_size: `16`
- max_steps: `2000`
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/anchor-128-2k`
- final training: step 899 的 anchor MSE/factor CE 为 `0.0135/0.615`，step 1499 为
  `0.0078/0.600`；step time window 约 `67 ms`，GPU peak allocated `3.52 GB`；2k 正常结束并导出
  artifact。
- train_eval: GPU 4；128 samples；started `2026-08-05 05:03 CST`；output
  `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/anchor-128-2k-train-eval`
- heldout_eval: GPU 2；16 samples；started `2026-08-05 05:03 CST`；output
  `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/anchor-128-2k-heldout-eval`
- heldout features: factor A/B accuracy `0.0798/0.0763`；raw/snap feature MSE `81.27/91.63`，
  表明预测没有泛化到 target stage-0 code manifold。
- heldout speech: raw/snap WER `0.0293/0.0239`，target reconstruction WER `0.0106`；raw/snap
  UTMOS `1.7090/1.8477`，target reconstruction UTMOS `4.2694`。严格对齐 anchor 已保住内容，但声学
  manifold/音质仍是主要缺口。
- train features: factor A/B accuracy `0.99683/0.99641`；raw/snap feature MSE `0.4704/0.1725`。
- train speech: raw/snap WER `0.0241/0.0237`，target reconstruction WER `0.0211`；raw/snap
  UTMOS `3.7336/3.7993`，target reconstruction UTMOS `4.3477`。

### Residual FM 1-utterance overfit

- status: passed
- started_at: `2026-08-05 05:07 CST`
- host: `144`
- cuda_visible_devices: `3`
- entry: `jobs/010/02_anchor_overfit.sh`
- fm_mode: `residual`
- sample_limit: `1`
- batch_size: `1`
- max_steps: `2000`
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/residual-1-2k`
- training: step 1379 的 anchor MSE/flow loss 为 `0.00169/0.0147`；step time window 约 `137 ms`，
  约为 anchor-only 的 2 倍。
- artifact: factor A/B accuracy 均为 `1.0`；raw/snap feature MSE `0.8936/0`。
- speech: raw UTMOS `3.0839`，高于同一样本 anchor raw 的 `2.9426`，接近 target reconstruction 的
  `3.2372`；snap UTMOS `2.9493`。该样本数字读法使绝对 WER 不可校准，但 residual 未改变 factor target。

### Residual FM 16-utterance overfit

- status: 2k finite, default 16-step quality gate failed
- started_at: `2026-08-05 05:27 CST`
- host: `144`
- cuda_visible_devices: `3`
- entry: `jobs/010/02_anchor_overfit.sh`
- fm_mode: `residual`
- sample_limit: `16`
- batch_size: `16`
- max_steps: `2000`
- first_attempt: 默认 learning rate `1e-3`；step 729 flow loss `0.9765`，step 841 finite guard 检出
  non-finite flow loss 并 fail-fast；anchor 分支仍 finite，只保留到 step 500 checkpoint，不作为质量结果。
- retry: 从头训练，learning rate `1e-4`，不复用首轮 optimizer/权重。
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/010-first-codebook-anchor-20260805/residual-16-2k-lr1e4`
- retry training: 2k 正常结束；step 1899 flow loss/anchor MSE 为 `0.00942/0.00203`，step 1999
  flow loss 为 `0.00651`；step time window 约 `146 ms`，GPU peak allocated `4.70 GB`。
- artifact: factor A/B accuracy `0.99978/0.99931`，与 anchor-16 相同；raw/snap feature MSE
  `0.0905/0.0142`，低于 anchor-16 的 `0.1530/0.0309`。
- speech: raw/snap WER `0.0412/0.0385`，与 anchor-16 完全相同；raw/snap UTMOS
  `3.7253/3.7541`，没有超过 anchor-16 的 `3.7501/3.7595`。默认 16-step residual 没有通过
  “WER 不退化且 UTMOS 提升”的质量门禁，不直接扩大到 128。
- integration diagnostic: 固定该 artifact、seed 和 16 条语音，只把 Euler steps 改为 32/64。raw feature
  MSE 分别为 `0.1032/0.1140`，高于 16-step 的 `0.0905`；raw UTMOS 分别为
  `3.7372/3.7337`，仍低于 anchor-16 的 `3.7501`；raw WER 均为 `0.0412`。三组 factor accuracy、
  snap MSE、snap WER 和 snap UTMOS 完全相同。
- decision: 更多积分步只累积速度场误差，没有恢复感知质量。010 停止 residual 续训和 128 条扩展；当前
  没有达到可接受质量的生成路线，因此不进入性能降配。默认 `feature_adapter=none`、`fm_mode=flow`
  保持不变。
