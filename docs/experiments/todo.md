# TODO

本文只记录未完成事项。完成项应移动到 `docs/experiments/results/` 或由 git 历史保留。

## P0: 016 LongCat Qwen-FiLM Temporal Anchor

- id: sac-016-longcat-qwen-film-anchor
- state: running
- entry: jobs/016/01_qwen_film_recurrent.sh
- num_gpus: 1
- gpu: 1x4090-24GB
- min_vram_gb_per_gpu: 8; batch96 probe measured 5.95GiB reserved peak with margin
- preferred_hosts: 125,145,144
- estimated_hours: 2
- monitor: TensorBoard factor loss/top-1 + valid frames/s/padding + performance probe + nvidia-smi
- ready_gate: passed; 260 local tests, artifact reload/decode smoke, and matched batch96 probe passed; Qwen-FiLM retained 90.9% of local-anchor throughput with equal VRAM
- output_root: $SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT/016
- host: 125
- cuda_visible_devices: 0
- started_at: 2026-08-06T19:35:24+08:00
- run: Qwen-FiLM temporal anchor + recurrent depth N2 正在训练到 5k；保存每 1k checkpoint
- task: 监控 5k 完成；用 heldout target-role 9984--9999 导出 1k--5k fixed-set audio，并与 015 local-anchor 5k 做 Anytrain WER/UTMOS 严格比较。

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
