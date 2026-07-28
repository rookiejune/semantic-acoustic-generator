# 005 Cross-Text Codec Overfit

## 目标

用当前 `qwen_cross_text` pair contract 完成真实 LongCat / BiCodec structured view、四路线
single-pair overfit 和 paired A/B decode，替换 fixed-speaker 旧数据契约的训练证据。

## 环境

- 日期：2026-07-28。
- 机器：复旦 145，NVIDIA RTX 4090D；GPU6 / GPU7。
- Python：`/home/zhuyin/anaconda3/envs/py312/bin/python`。
- 代码快照：`/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/cross-text-20260728/code`。
- 运行根：`/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/cross-text-20260728`。

共享 Git 工作树存在未提交历史改动，本实验使用隔离 rsync 快照，不修改共享工作树。

## 数据

输入 waveform grid：

`/mnt/pami202/zhuyin/dynamic/datasets/wmt19_qwen_tts_role_speaker/train_0_1000`

manifest 包含 1000 个 source、source/target 两种 role、vivian/ryan 两个 speaker，共 4000 个 flat
samples。完整物化以下两个 codec views：

- `wmt19_qwen_tts_role_speaker_longcat/train_0_1000`
- `wmt19_qwen_tts_role_speaker_bicodec/train_0_1000`

训练固定使用 target role、speaker `vivian`、sample index 0；reference 必须是同 speaker、不同
text index、utterance id 和文本的下一条有效样本。

## 训练

运行 LongCat / BiCodec x FM / RVQ 四条路线：

- 8-layer decoder，condition dim 1024；
- batch size 1，20 optimizer steps；
- learning rate `1e-3`，weight decay 0；
- bf16 mixed precision；
- `reference_dropout=0.5`；
- 第 20 step 导出 paired sample metrics、with/without-reference audio、full-unit reconstruction；
- 训练结束导出 schema 7 artifact。

## 验收

1. 两个 codec view 完整包含 4000 个样本并通过 grid loader 校验。
2. 四条路线 loss finite 且有优化信号。
3. 四个 artifact 的 with-reference / without-reference waveform 均 finite。
4. paired metadata 满足 same-speaker / cross-text 不变量。
5. BiCodec RVQ 确认输出 32 个 acoustic slots，并记录 generation/decode 耗时。

本实验不声明泛化、音质、音色保持或 reference gain 有效；这些结论需要后续 screening、held-out
fixed eval 和 leakage 检查。
