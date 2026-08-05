from __future__ import annotations

import argparse
import json
import sys
import wave
from array import array
from pathlib import Path
from typing import Any

import torch
from anytrain.codec import SemanticAcousticCodes, masked_acoustic_features

from semantic_acoustic_generator.backend import (
    BackendConfig,
    LongCatCodebookAdapter,
    load_backend,
)
from semantic_acoustic_generator.datamodule import BatchingConfig, DataConfig, load_batch
from semantic_acoustic_generator.evaluation import masked_feature_mse, seeded_generator
from semantic_acoustic_generator.model import FMFeatureGenerator
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
                sample_index=sample_index,
                batching=BatchingConfig(enabled=False),
            ),
            codec=args.codec,
            frame_rate=runtime.frame_rate,
            acoustic_layout=runtime.backend.acoustic_layout,
            semantic_pad_id=int(runtime.backend.semantic_codebook.size(0)),
            acoustic_pad_ids=runtime.backend.acoustic_codebook_sizes,
        ).to(device)
        target = masked_acoustic_features(
            runtime.backend,
            batch.acoustic_codes,
            batch.acoustic_mask,
        )
        generated = runtime.sample_features(
            batch.semantic_codes,
            mask=batch.mask,
            generator=seeded_generator(device, args.seed + sample_index),
        )
        semantic = batch.semantic_codes.masked_fill(~batch.mask[..., None], 0)
        acoustic = batch.acoustic_codes.masked_fill(~batch.acoustic_mask[..., None], 0)
        audio = {
            "target_reconstruction": runtime.backend.detokenize(
                SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)
            ),
            "generated_without_reference_raw": runtime.decode_features(
                semantic,
                generated,
                mask=batch.mask,
            ),
        }
        metrics: dict[str, float] = {
            "raw_feature_mse": masked_feature_mse(
                generated,
                target,
                batch.acoustic_mask,
                name="artifact set raw",
            )
        }
        if isinstance(runtime.backend, LongCatCodebookAdapter):
            audio["selected_codebook_reconstruction"] = runtime.decode_features(
                semantic,
                target,
                mask=batch.mask,
            )
            snapped = runtime.backend.snap_features(generated)
            audio["generated_without_reference_snap"] = runtime.decode_features(
                semantic,
                snapped,
                mask=batch.mask,
            )
            metrics["snap_feature_mse"] = masked_feature_mse(
                snapped,
                target,
                batch.acoustic_mask,
                name="artifact set snap",
            )
            predicted = runtime.backend.features_to_factor_codes(generated)
            labels = runtime.backend.factor_codes(batch.acoustic_codes).to(predicted.device)
            valid = batch.acoustic_mask.to(predicted.device)
            _factor_accuracy(metrics, predicted, labels, valid)
            feature_generator = runtime.support.generator
            if (
                isinstance(feature_generator, FMFeatureGenerator)
                and feature_generator.factor_depth is not None
            ):
                condition = runtime.support.condition(batch.semantic_codes, mask=batch.mask)
                teacher_logits = feature_generator.factor_logits(
                    condition,
                    batch.mask,
                    factor_targets=labels,
                )
                teacher = torch.stack(
                    tuple(value.argmax(dim=-1) for value in teacher_logits),
                    dim=-1,
                )
                _factor_accuracy(
                    metrics,
                    teacher,
                    labels,
                    valid,
                    prefix="teacher_forced_",
                )
            if runtime.backend.feature_codebooks > 1:
                audio["generated_stage0_only"] = runtime.backend.decode_features(
                    semantic,
                    generated,
                    active_codebooks=1,
                )
        text = batch.metadata[0].target_text
        for group, waveform in audio.items():
            path = output / group / f"sample-{sample_index:04d}.wav"
            _write_wav(path, waveform, sample_rate=runtime.sample_rate)
            manifest.append(
                {
                    "group": group,
                    "sample_index": sample_index,
                    "wav": str(path),
                    "target_text": text,
                }
            )
        sample = {"sample_index": sample_index, "target_text": text, **metrics}
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


def _factor_accuracy(
    metrics: dict[str, float],
    predicted: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    *,
    prefix: str = "",
) -> None:
    accuracy = predicted[valid].eq(labels[valid]).float().mean(dim=0)
    for factor, value in enumerate(accuracy):
        codebook, local = divmod(factor, 2)
        suffix = "a" if local == 0 else "b"
        name = (
            f"factor_{suffix}_accuracy"
            if codebook == 0
            else f"codebook_{codebook}_factor_{suffix}_accuracy"
        )
        metrics[f"{prefix}{name}"] = float(value.cpu())


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an artifact on a fixed utterance set.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec", default="longcat")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.sample_start < 0 or args.sample_limit <= 0:
        parser.error("sample-start must be non-negative and sample-limit must be positive")
    return args


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
