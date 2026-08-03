# 007 LongCat-FM REPA / EMA Ablation

## 目标

在已确认 codec initialization 有效、FM 优于 RVQ 的前提下，验证 LongCat-FM 上：

1. REPA（WavLM teacher）是否相对纯 FM baseline 有提升；
2. generator 权重 EMA 是否相对纯 FM baseline 有提升。

不扫 reference_dropout、initialization、decoder depth 或 RVQ。

## 固定不变量

- `backend=longcat`、`model/route=fm`
- `runtime.initialization=codec`、`model.decoder.layers=8`
- `pl_module.reference_dropout=0.5`
- 数据：`qwen_cross_text`，`sample_limit=984`，`max_seconds=32`，`overlong=filter`
- batch budget：`max_batch_seconds=32`
- 同一 held-out fixed eval 协议（with/without-reference、同 seed 独立 generator）

## 矩阵

| Cell | Experiment | 变化 |
| --- | --- | --- |
| A0 | `experiment=ablation_fm_baseline` | REPA off, EMA off |
| A1 | `experiment=ablation_fm_repa` | `loss.repa_loss_weight=0.25`，WavLM-base |
| A2 | `experiment=ablation_fm_ema` | `pl_module.ema_decay=0.999` |

可选后续：`pl_module.ema_decay=0.999` 叠在 A1 上做 A3，不单独建 preset 时可 CLI override。

## 实现边界

- 权重 EMA：`anytrain.lightning.EMACallback`（task-agnostic）
- REPA loss / DiT feature hook：复用 anytrain
- `WavLMTeacher` 与 train 接线：本仓库

## 效率探针（填满 GPU）

短跑效率数字与 REPA 变慢原因见
[`../results/007_fm_repa_ema_ablation.md`](../results/007_fm_repa_ema_ablation.md)。

要点：cost batching 以 A1 为准（`max_batch_seconds=360` 可开 `profile_flops`；`420` 易 OOM）。
A1 墙钟约慢 9×、counted FLOPs 约 3×；MFU 大跌不能归因于“WavLM 同时进分子分母”，见 results 分析。

## 验收

1. A0–A2 均能完成 screening 预算并导出 artifact。
2. A1 训练日志出现 finite `train/repa_loss`。
3. A2 的 sample / artifact 走 EMA 权重（`EMACallback.average_parameters`）。
4. 同一 held-out pair 集上比较 without-ref / with-ref feature MSE；泄漏检查完成前不把 reference gain 写入 conclusion。

## 入口

```bash
python scripts/train.py experiment=ablation_fm_baseline
python scripts/train.py experiment=ablation_fm_repa
python scripts/train.py experiment=ablation_fm_ema
```
