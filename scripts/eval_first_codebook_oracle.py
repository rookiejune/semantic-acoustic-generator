from __future__ import annotations

import argparse
import json
import sys
import wave
from array import array
from pathlib import Path
from typing import Any

import torch

from semantic_acoustic_generator.backend import (
    BackendConfig,
    LongCatFirstCodebookAdapter,
    adapt_backend,
    load_backend,
)
from semantic_acoustic_generator.config import FeatureAdapter
from semantic_acoustic_generator.datamodule import BatchingConfig, DataConfig, load_batch
from semantic_acoustic_generator.evaluation import evaluate_first_codebook_oracle


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    raw_backend = load_backend(BackendConfig(name="longcat"), device=device)
    adapted = adapt_backend(raw_backend, FeatureAdapter.LONGCAT_FIRST_CODEBOOK)
    if not isinstance(adapted, LongCatFirstCodebookAdapter):
        raise TypeError("LongCat first-codebook oracle requires its feature adapter.")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    sample_results: list[dict[str, Any]] = []
    for offset in range(args.sample_limit):
        sample_index = args.sample_start + offset
        batch = load_batch(
            DataConfig(
                source="qwen_cross_text",
                root=str(args.data_root),
                split=args.split,
                sample_index=sample_index,
                batching=BatchingConfig(enabled=False),
            ),
            codec="longcat",
            frame_rate=adapted.frame_rate,
            acoustic_layout=adapted.acoustic_layout,
            semantic_pad_id=int(adapted.semantic_codebook.size(0)),
            acoustic_pad_ids=adapted.acoustic_codebook_sizes,
        ).to(device)
        evaluation = evaluate_first_codebook_oracle(
            adapted,
            batch,
            sigmas=args.sigmas,
            seed=args.seed + sample_index,
        )
        metadata = batch.metadata[0]
        for group, waveform in evaluation.audio.items():
            path = output / group / f"sample-{sample_index:04d}.wav"
            _write_wav(path, waveform, sample_rate=adapted.sample_rate)
            manifest.append(
                {
                    "group": group,
                    "sample_index": sample_index,
                    "wav": str(path),
                    "target_text": metadata.target_text,
                }
            )
        result = {
            "sample_index": sample_index,
            "target_text": metadata.target_text,
            **evaluation.metrics,
        }
        sample_results.append(result)
        print(json.dumps({"sample_index": sample_index, **evaluation.metrics}, sort_keys=True))
    private = {"sample_rate": adapted.sample_rate, "items": manifest, "samples": sample_results}
    (output / "manifest.private.json").write_text(
        json.dumps(private, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = _summary(sample_results, sigmas=args.sigmas)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, sort_keys=True))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LongCat stage-0 16-D oracle paths.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--sigmas", type=_sigmas, default=(0.05, 0.1, 0.2, 0.5, 1.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.sample_start < 0 or args.sample_limit <= 0:
        parser.error("sample-start must be non-negative and sample-limit must be positive")
    return args


def _sigmas(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("sigmas must be a comma-separated list of positives")
    return result


def _summary(samples: list[dict[str, Any]], *, sigmas: tuple[float, ...]) -> dict[str, Any]:
    projection = [float(item["native_projection_max_abs"]) for item in samples]
    exact_snap = [float(item["exact_snap_max_abs"]) for item in samples]
    groups: dict[str, dict[str, float]] = {}
    for sigma in sigmas:
        key = format(sigma, ".6g").replace("-", "m").replace(".", "p")
        for prefix in ("raw", "snap"):
            name = f"{prefix}_sigma_{key}"
            values = [item["groups"][name] for item in samples]
            groups[name] = {
                field: sum(float(value[field]) for value in values) / len(values)
                for field in ("feature_mse", "factor_a_accuracy", "factor_b_accuracy")
            }
    return {
        "count": len(samples),
        "sigmas": list(sigmas),
        "native_projection_max_abs": max(projection),
        "exact_snap_max_abs": max(exact_snap),
        "groups": groups,
    }


def _write_wav(path: Path, waveform: torch.Tensor, *, sample_rate: int) -> None:
    audio = waveform.detach().float().cpu()[0]
    if audio.dim() != 2:
        raise ValueError("decoded waveform must have shape [batch, channel, samples].")
    pcm = (
        audio.clamp(-1, 1)
        .mul(32_767)
        .round()
        .to(torch.int16)
        .transpose(0, 1)
        .contiguous()
        .reshape(-1)
    )
    frames = array("h", pcm.tolist())
    if sys.byteorder == "big":
        frames.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(audio.size(0))
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames.tobytes())


if __name__ == "__main__":
    main()
