# TODO

本文只记录未完成事项。完成项应移动到 `docs/experiments/results/` 或由 git 历史保留。

## P0: 015 LongCat Factor Depth Formal Comparison

- id: sac-015-longcat-factor-depth-formal
- state: running
- entry: jobs/015/01_factor_recurrent.sh; jobs/015/02_factor_qwen_baseline.sh
- num_gpus: 1
- gpu: 1x4090-24GB
- min_vram_gb_per_gpu: 18; Qwen batch96 probe measured 13.99GiB reserved peak with margin
- preferred_hosts: 145,144,125
- estimated_hours: 2
- monitor: TensorBoard factor loss/top-1 + valid frames/s/padding + training log + nvidia-smi
- ready_gate: passed; strict target-role 9984/16 split, recurrent-original and Qwen-original 2k full-N2 both beat their stage0-only UTMOS with unchanged WER; batch96/1152s probe passed with margin
- output_root: $SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT/015
- host: 145
- cuda_visible_devices: 0
- started_at: 2026-08-06T04:41:57+08:00
- run: recurrent-original 正在运行，成功后由同一 tmux controller 串行启动 Qwen-original
- task: 监控两组 20k 正式训练；checkpoint 先写本机盘，完成后用同一 heldout manifest 和 Anytrain evaluator 统一评估。

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
