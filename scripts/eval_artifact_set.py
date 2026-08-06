from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from semantic_acoustic_generator.backend import BackendConfig, load_backend
from semantic_acoustic_generator.datamodule import BatchingConfig, DataConfig, load_batch
from semantic_acoustic_generator.evaluation import evaluate_artifact_sample, write_pcm16_wav
from semantic_acoustic_generator.runtime import GeneratorRuntime
from semantic_acoustic_generator.runtime.artifact import load_artifact


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    backend = load_backend(BackendConfig(name=args.codec), device=device)
    support = load_artifact(args.artifact, device=device)
    runtime = GeneratorRuntime(support, backend)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for offset in range(args.sample_limit):
        sample_index = args.sample_start + offset
        batch = load_batch(
            DataConfig(
                source="qwen_cross_text",
                root=str(args.data_root),
                split=args.split,
                role=args.role,
                speaker_id=args.speaker_id,
                sample_index=sample_index,
                batching=BatchingConfig(enabled=False),
            ),
            codec=args.codec,
            frame_rate=runtime.frame_rate,
            acoustic_layout=runtime.backend.acoustic_layout,
            semantic_pad_id=int(runtime.backend.semantic_codebook.size(0)),
            acoustic_pad_ids=runtime.backend.acoustic_codebook_sizes,
        ).to(device)
        evaluation = evaluate_artifact_sample(
            runtime,
            batch,
            seed=args.seed + sample_index,
        )
        text = batch.metadata[0].target_text
        for group, waveform in evaluation.audio.items():
            path = output / group / f"sample-{sample_index:04d}.wav"
            write_pcm16_wav(path, waveform, sample_rate=runtime.sample_rate)
            manifest.append(
                {
                    "group": group,
                    "sample_index": sample_index,
                    "wav": str(path),
                    "target_text": text,
                }
            )
        sample = {"sample_index": sample_index, "target_text": text, **evaluation.metrics}
        samples.append(sample)
        print(json.dumps({key: value for key, value in sample.items() if key != "target_text"}))
    private = {"items": manifest, "samples": samples}
    (output / "manifest.private.json").write_text(
        json.dumps(private, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metric_names = sorted({key for sample in samples for key in sample if key not in {"sample_index", "target_text"}})
    summary = {
        "count": len(samples),
        "metrics": {
            name: sum(float(sample[name]) for sample in samples if name in sample)
            / sum(name in sample for sample in samples)
            for name in metric_names
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, sort_keys=True))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an artifact on a fixed utterance set.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec", default="longcat")
    parser.add_argument("--split", default="train")
    parser.add_argument("--role", choices=("source", "target"), default="target")
    parser.add_argument("--speaker-id", default="vivian")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.sample_start < 0 or args.sample_limit <= 0:
        parser.error("sample-start must be non-negative and sample-limit must be positive")
    return args
if __name__ == "__main__":
    main()
