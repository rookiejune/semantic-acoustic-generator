# TODO

本文只记录未完成事项。完成项应移动到 `docs/experiments/results/` 或由 git 历史保留。

## P0: 009 LongCat FM 50k→200k Continuation

- id: sac-009-longcat-fm-200k
- state: running
- entry: `jobs/001/01_longcat_dit.sh trainer.max_steps=100000 trainer.ckpt_path=<resume-checkpoint> datamodule.batch_size=48 datamodule.batching.max_batch_seconds=576.0`; fixed eval 后再决定是否继续到 200k
- num_gpus: 1
- gpu: 1x4090-24GB
- min_vram_gb_per_gpu: 24GB-class
- preferred_hosts: 144
- estimated_hours: staged longrun
- monitor: TensorBoard plus train logs and anytrain throughput/memory metrics; checkpoint/sample every 10000 steps; fixed eval at 100k and 200k
- ready_gate: before launch, require a resume artifact supplied outside the public repository, a pushed code revision, and one eligible 24GB-class GPU
- task: 验证 LongCat FM continuation 能否通过后续固定评测的预定义质量门禁；先运行到 100k，仅在门禁通过后再决定是否继续到 200k。

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
