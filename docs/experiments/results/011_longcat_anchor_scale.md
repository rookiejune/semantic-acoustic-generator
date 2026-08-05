# 011 LongCat First-Codebook Anchor Scale

对应计划：`docs/experiments/schedules/011_longcat_anchor_scale.md`。

## 运行记录

### 2-step smoke

- status: passed
- host: `144`
- cuda_visible_devices: `2`
- entry: `jobs/011/01_anchor_scale.sh`
- sample_limit: `16`
- batch_size: `16`
- max_steps: `2`
- result: 真实 10k store、LongCat first-codebook adapter、anchor loss/反传和 artifact 导出全部通过；
  14.6M trainable parameters，loss finite

### 10k-data local anchor

- status: 2k held-out completed; resumed to 5k
- host: `144`
- cuda_visible_devices: `2`
- pid: `3291702`
- entry: `jobs/011/01_anchor_scale.sh`
- sample_limit: `10000`
- batch_size: `16`
- max_steps: `2000`
- output: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/011-longcat-anchor-scale-20260805/011/local-anchor-10000-2000`
- fixed invariants: 与 010 anchor-128 相同的 local anchor、loss、optimizer 和无 reference 设置
- training: 2k 正常结束；step 1969 factor CE/anchor MSE 为 `4.37/0.744`，step time window 约
  `97 ms`，GPU peak allocated `3.85 GB`；10k 分布在该预算下明显欠拟合
- train_eval: GPU 3；PID `3319111`；固定 128 samples
- heldout_eval: GPU 4；PID `3319115`；独立 16 samples
- 2k heldout features: factor A/B accuracy `0.1337/0.1145`，高于 010 anchor-128 的
  `0.0798/0.0763`；raw/snap feature MSE `45.97/65.52`，低于 `81.27/91.63`
- 2k heldout speech: raw/snap WER 均为 `0.0239`；raw/snap UTMOS `1.2739/1.4206`，低于 010 的
  `1.7090/1.8477`
- 2k train-128 features: factor A/B accuracy `0.1466/0.1202`，raw/snap feature MSE
  `43.79/62.22`。train 与 held-out 同时只有十几个百分点，表明 2k 整体欠拟合，不是泛化分叉
- 2k train-128 speech: raw/snap WER `0.0303/0.0336`，raw/snap UTMOS `1.2626/1.4514`，target
  reconstruction WER/UTMOS `0.0211/4.3477`；训练集音质与 held-out 同样低
- resume: 从 2k `last.ckpt` 恢复 model/optimizer/global-step，保持 `lr=1e-3`，续到 5k；GPU 2；PID
  `3324024`；output `$DYNAMIC_HOME/debug/semantic-acoustic-generator/011-longcat-anchor-scale-20260805/011/local-anchor-10000-5000`。
  Lightning 明确提示当前 dataloader 不支持 cursor resume，因此从当前 epoch 开头重新取数，会重复该 epoch
  前段样本；5k 只作为质量趋势门禁，不作为严格数据顺序复现结果
- 5k training: step 4999 factor CE/anchor MSE 为 `4.214/0.714`，相对 2k 只小幅下降并进入平台
- 5k train-128 features: factor A/B accuracy `0.1635/0.1380`，raw/snap feature MSE
  `41.93/59.72`；相比 2k 有限改善，仍是整体欠拟合
- 5k train-128 speech: raw/snap WER `0.0310/0.0284`，raw/snap UTMOS `1.2842/1.5148`；与
  held-out 同样没有恢复感知质量
- 5k heldout features: factor A/B accuracy `0.1447/0.1187`，raw/snap feature MSE
  `45.01/65.93`
- 5k heldout speech: raw/snap WER `0.0266/0.0213`，raw/snap UTMOS `1.2913/1.5524`；snap
  UTMOS 仍低于 010 anchor-128 的 `1.8477`
- decision: local anchor 停止，不续到 10k；train 与 held-out 同时低准确率，且 3k 额外更新未恢复感知质量

### Frame-preserving Transformer anchor

- code: `ec95c1c`; `anchor_context=local|transformer`，默认 local；Transformer 保持输入输出 frame 轴长度
  不变，不使用 stride、pooling、插值、长度预测或自回归移位
- smoke: passed；host `145` GPU 0；16 samples/batch 16/2 steps；24.0M trainable parameters；真实
  store、attention mask、bf16 loss/backward、checkpoint 与 artifact 导出通过
- run: completed；host `145` GPU 5；10k samples/batch 16/2k steps；入口
  `jobs/011/02_transformer_anchor.sh`；约 36 分钟完成并导出 artifact
- invariant: first-codebook 16-D target、normalized MSE、cosine、两个 tied-codebook factor CE、snap
  decode、learning rate 与 local 2k 对照保持不变；只改变 anchor context
- training: 最后 50 个日志点 factor CE/MSE 均值 `4.519/0.847`，step-time window 均值约
  `110 ms`；GPU peak allocated/reserved `4.61/5.49 GB`；收敛弱于 local anchor
- train-128 features: factor A/B accuracy `0.1077/0.0928`，raw/snap feature MSE
  `51.66/77.83`
- heldout-16 features: factor A/B accuracy `0.0959/0.0785`，raw/snap feature MSE
  `53.28/81.40`；train 与 held-out 同时劣于 local，排除“只因 9-frame context 太窄”的解释
- speech eval: 145 的 py312 缺 `openai-whisper`，feature 阶段完成后明确失败；manifest 已迁到
  已验证环境 144 的 GPU 3/4 继续跑 Whisper/UTMOS，不改变生成音频或 feature 结果
- heldout-16 speech: raw/snap WER `0.0399/0.0559`，raw/snap UTMOS `1.2482/1.3127`；内容与感知质量
  都劣于 local 2k/5k 和 010 anchor-128
- train-128 speech: raw/snap WER `0.0514/0.1088`，raw/snap UTMOS `1.2478/1.3279`；训练集同样
  退化，确认不是 held-out 泛化分叉
- decision: Transformer anchor 停止，不延长预算、不增加深度，也不继续把“局部上下文不足”作为主假设；
  现有证据更符合 semantic-only 条件无法唯一确定第一 acoustic codebook 细节，或连续 16-D 回归把多模态
  条件均值映射到非码本流形。后续若继续该 target，应先验证信息上限/离散分类参数化，而不是继续扩模型
