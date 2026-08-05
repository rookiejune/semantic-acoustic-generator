# TODO

本文只记录未完成事项。完成项应移动到 `docs/experiments/results/` 或由 git 历史保留。

## P0: 014 LongCat Factor Depth-AR Probe

- id: sac-014-longcat-factor-depth
- state: running
- entry: jobs/014/01_factor_depth.sh
- num_gpus: 1
- gpu: 1x4090-24GB
- min_vram_gb_per_gpu: 12
- preferred_hosts: 144,145,125
- estimated_hours: 2
- monitor: TensorBoard factor loss/top-1 and performance metrics + training log + nvidia-smi
- ready_gate: passed; 128-sample/2k overfit reached at least 99.7% teacher-forced top-1 on all factors, with about 0.3pp stage-1 free-running gap
- output_root: $SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT/014
- task: N2 depth-AR 10k/2k main 已完成；heldout factor/feature 已导出，Anytrain Whisper large-v3/UTMOS 正在 `121`/GPU1 评估 full N2 与 stage0-only，随后与 N1 和 013 parallel N2 对照。

## P0: 006/007 Fixed Eval Review

- id: sac-006-007-fixed-eval-review
- state: ready
- entry: review exported fixed-eval audio and JSON summaries
- num_gpus: 0
- gpu: none
- min_vram_gb_per_gpu: none
- preferred_hosts: local
- estimated_hours: 1
- monitor: manual audio review checklist; sanitized notes only
- ready_gate: 006 four-route screening completed; 007 A3 REPA+EMA screening completed; fixed eval artifacts exported for the held-out cross-text set
- task: 对 006 四路线和 007 A3 的 fixed eval 导出物做人工听评、reference 泄漏检查和结论整理；只记录可脱敏复述的现象，不在 todo 中写私密路径、MSE 明细或样本内容。
