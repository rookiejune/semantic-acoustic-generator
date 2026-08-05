# 014 LongCat Factor Depth Autoregression

## 2026-08-06 Real LongCat Smoke

- revision: `8433bbf281afa252b974d6469f82d2e3b9a7c53f`
- host/device: `145`, `CUDA_VISIBLE_DEVICES=0`, RTX 4090 D 24GB
- run: 16 samples, batch 16, 2 optimizer steps, bf16 mixed precision
- output: `$SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT/014/depth-ar-2cb-smoke-16-2`
- status: passed; LongCat encoder/decoder load, training, artifact export/reload and fixed-sample decode all completed without NaN or OOM

Contract probe on one 66-frame sample:

- free-running factor codes: `[1, 66, 4]`
- teacher-forced logits: four tensors of `[1, 66, 90]`
- generated factor features: `[1, 66, 32]`
- two-stage projection versus native LongCat reconstruction max error: `0.0`
- stage-0 prefix projection versus native stage-0 reconstruction max error: `0.0`
- full and stage-0-only generated WAVs were both exported as valid PCM16 files

Performance probe with callback warmup disabled:

- peak allocated: `4,369,546,752` bytes (`4.07 GiB`)
- peak reserved: `5,110,759,424` bytes (`4.76 GiB`)
- post-warm step time: `0.119 s`
- measured model FLOPs per step: `526,089,486,336`

The 2-step factor accuracies are only a path-health check and are not treated as a model-quality result. The next gate is the 128-sample/2k overfit run defined in the schedule.

## 2026-08-06 Overfit Gate Launch

- started at: `2026-08-06T01:02:23+08:00`
- host/device: `145`, `CUDA_VISIBLE_DEVICES=0`
- remote PID: `1810089`
- run: 128 samples, batch 16, 2,000 optimizer steps
- output: `$SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT/014/depth-ar-2cb-overfit-128-2000`
- log: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/sac014-overfit-128-2000-8433bbf.log`

The run completed normally at step 2,000. Final training metrics:

- loss: `0.00614`
- stage 0 A/B top-1: `99.33% / 99.39%`
- stage 1 A/B top-1: `100% / 100%`
- peak allocated: `4.39 GiB`
- peak reserved: `8.05 GiB`
- post-warm step-time window: `0.116 s`

Reloaded-artifact evaluation on the exact 128-sample training subset, covering 25,463 valid frames:

- teacher-forced A0/B0/A1/B1: `99.705% / 99.705% / 100% / 100%`
- free-running A0/B0/A1/B1: `99.705% / 99.705% / 99.674% / 99.690%`
- stage 1 free-running gap: `0.326 / 0.310` percentage points
- stage 0 pair accuracy: `99.662%` (86 wrong frames)
- stage 1 given a correct stage 0 pair: `100% / 100%`
- stage 1 given a wrong stage 0 pair: `3.49% / 8.14%`

The gate passes: the two-stage factor classifier scales on the memorization set, and its small overall free-running gap is directly localized to frames where the preceding stage pair is wrong.

## 2026-08-06 Main Launch

- started at: `2026-08-06T01:18:27+08:00`
- host/device: `145`, `CUDA_VISIBLE_DEVICES=0`
- remote PID: `1817353`
- run: 10,000 samples, batch 16, 2,000 optimizer steps
- output: `$SEMANTIC_ACOUSTIC_GENERATOR_TRAIN_ROOT/014/depth-ar-2cb-10000-2000`
- log: `$DYNAMIC_HOME/debug/semantic-acoustic-generator/sac014-main-10000-2000-8433bbf.log`

The run completed normally at step 2,000. Final 20-point training window:

- loss: `3.2793`
- stage 0 A/B top-1: `17.82% / 15.96%`
- stage 1 A/B top-1: `17.37% / 15.53%`
- peak allocated: `4.48 GiB`
- peak reserved: `9.35 GiB`; whole-process `nvidia-smi` peak was about `9.8 GiB`
- step-time window: `0.130 s`; end-to-end wall time was data/NAS limited

Fixed heldout-16 artifact evaluation:

- stage 0 free/teacher A/B: `17.80% / 15.53%`
- stage 1 teacher-forced A/B: `17.50% / 16.52%`
- stage 1 free-running A/B: `8.22% / 8.00%`
- raw/snap feature MSE: `55.4681`

The correct prefix lifts stage 1 to stage-0-level accuracy, but predicted-prefix exposure error removes that gain in free-running inference. Stage 1 free-running accuracy is effectively the same as the 013 parallel N2 baseline (`8.13% / 7.92%`).

Anytrain Whisper large-v3/UTMOS evaluation launched at `2026-08-06T01:57:17+08:00` on `121`, `CUDA_VISIBLE_DEVICES=1`, PID `2468697`. It evaluates target, selected-codebook oracle, full N2, snap and stage0-only groups from the same heldout manifest.
