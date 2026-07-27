# 003 Fixed Sample A/B

## 目标

使用同一条固定 Qwen speaker sample，导出 semantic-only generator waveform 与 LongCat full-unit reconstruction waveform。只记录可复现 baseline，不做主观音质结论。

## 输入

- Artifact：LongCat FM 8-layer single-sample overfit artifact。
- Data source：qwen_fixed_speaker。
- Speaker：vivian。
- Split / index：train / 0。
- Codec：LongCat。
- Seed：0。

## 执行

扩展 scripts/eval_artifact.py：

- 保留原 semantic-only generated waveform 输出。
- 新增 full-unit reconstruction 输出。
- 新增 --reconstruction-wav 和 --output-json。
- JSON 同时记录 generated 与 reconstruction 的 shape、duration、finite、min/max 和 RMS。

远端在复旦 145 GPU0 运行，输出写入 dynamic debug 目录。
