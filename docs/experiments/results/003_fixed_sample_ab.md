# 003 Fixed Sample A/B

## 范围

- 日期：2026-07-27。
- 机器：复旦 145，NVIDIA RTX 4090D，GPU0。
- Artifact：/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/overfit-longcat-fm-8l-sample0/artifact。
- Data root：/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/qwen-fixed-speaker-codec-32。
- Sample：qwen_fixed_speaker train index 0，speaker vivian。
- 输出：/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/fixed-ab-longcat-fm-overfit-sample0-1785148822340。

## 入口

scripts/eval_artifact.py 现在一次执行同时生成：

- generated.wav：只输入 semantic units，由 LongCat FM overfit artifact 预测 acoustic features 后 decode。
- reconstruction.wav：输入同一条样本的完整 semantic + acoustic units，由 LongCat backend reconstruction。
- summary.json：两条 waveform 的数值摘要和输出路径。

## 结果

| Metric | Semantic-only generated | Full-unit reconstruction |
| --- | ---: | ---: |
| Shape | [1, 1, 29760] | [1, 1, 29760] |
| Sample rate | 16000 Hz | 16000 Hz |
| Duration | 1.86s | 1.86s |
| Finite | yes | yes |
| RMS | 0.1370836 | 0.0993583 |
| Min | -0.9696943 | -0.5503832 |
| Max | 0.9986243 | 0.4604407 |
| WAV size | 59564 bytes | 59564 bytes |

Codec unit shapes：

- semantic: [1, 31, 1]
- acoustic: [1, 31, 3]

## 结论边界

两条 WAV 都成功导出、长度一致且数值 finite，固定样本 A/B baseline 已建立。这里不比较可懂度、音色、噪声或主观质量；后续 fixed eval 应对 held-out 文本导出更大样本集后再做人工检查。
