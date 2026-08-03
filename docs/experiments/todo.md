# TODO

本文只记录未完成事项。完成项应移动到 `docs/experiments/results/` 或由 git 历史保留。

## P0: 008 Staged 10k→50k Reference Probe

- id: sac-008-10k-reference-probe
- state: running
- entry: `scripts/train.py experiment=001_longcat_fm` and `scripts/train.py experiment=001_bicodec_rvq`; resume clean 10k runs from `checkpoints/last.ckpt` to 50k
- num_gpus: 2
- gpu: 2x4090-24GB single-card probes
- min_vram_gb_per_gpu: 24GB-class until probe completes
- preferred_hosts: 144
- estimated_hours: staged probe
- monitor: train logs; 10k checkpoint/sample every 1000 steps; watcher resumes clean completions to 50k with checkpoint/sample every 5000 steps; fixed eval curve after completion; manual listening review; reference leakage check
- ready_gate: code at pushed fixed-eval accessor commit; 0_10000 Qwen cross-text codec views ready; host 144 GPUs 0/1 assigned as independent single-card runs; watcher installed for clean 10k→50k resume
- task: 跑 LongCat FM 和 BiCodec RVQ 两条 10k-step 单卡 reference probe；若 10k 日志无 Traceback/OOM/Error 且正常完成，则自动从 last checkpoint 在同一卡续跑到 50k；完成后用同一 fixed eval set 汇总曲线，再做人工听评和 reference 泄漏检查。todo 不记录私密输出路径、指标明细或样本内容。

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
