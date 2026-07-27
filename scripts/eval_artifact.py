from __future__ import annotations

import argparse
import json
import sys
import wave
from array import array
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes, load_semantic_acoustic

from semantic_acoustic_codec.datamodule import (
    DataConfig,
    LBAConfig,
    load_batch,
)
from semantic_acoustic_codec.runtime import SemanticCodecRuntime, load_artifact

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec
    from torch import Tensor

    from semantic_acoustic_codec.types import SemanticCodecBatch


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    backend = load_semantic_acoustic(args.codec, device=device)
    support = load_artifact(args.artifact, device=device)
    data = DataConfig(
        source=args.data_source,
        root=None if args.data_root is None else str(args.data_root),
        split=args.split,
        sample_index=args.sample_index,
        max_seconds=args.max_seconds,
        overlong=args.overlong,
        lba=LBAConfig(enabled=False),
    )
    batch = load_batch(
        data,
        codec=args.codec,
        frame_rate=backend.frame_rate,
        acoustic_layout=backend.acoustic_layout,
        semantic_pad_id=int(backend.semantic_codebook.size(0)),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    runtime = SemanticCodecRuntime(support, backend)
    audio, metrics = _evaluate(
        runtime,
        backend,
        batch,
        device=device,
        seed=args.seed,
    )
    result = _summary(
        audio,
        metrics,
        sample_rate=int(runtime.sample_rate),
        args=args,
        batch=batch,
    )
    _write_outputs(audio, result, args=args, sample_rate=int(runtime.sample_rate))
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized, encoding="utf-8")
    print(serialized)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode a semantic-acoustic artifact on one sample.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--codec", default="longcat")
    parser.add_argument("--data-source", choices=("qwen_cross_text",), default="qwen_cross_text")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--overlong", choices=("error", "filter", "truncate"), default="error")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--without-reference-wav",
        "--output-wav",
        dest="without_reference_wav",
        type=Path,
        default=None,
    )
    parser.add_argument("--with-reference-wav", type=Path, default=None)
    parser.add_argument(
        "--target-reconstruction-wav",
        "--reconstruction-wav",
        dest="target_reconstruction_wav",
        type=Path,
        default=None,
    )
    parser.add_argument("--reference-reconstruction-wav", type=Path, default=None)
    parser.add_argument("--passthrough-wav", type=Path, default=None)
    return parser.parse_args()


def _generator(*, device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def _evaluate(
    runtime: SemanticCodecRuntime,
    backend: SemanticAcousticCodec,
    batch: SemanticCodecBatch,
    *,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    if not batch.has_reference or len(batch.metadata) != 1:
        raise ValueError("artifact evaluation requires one cross-text target/reference pair.")
    semantic = batch.semantic_codes.to(device=device)
    semantic_mask = batch.mask.to(device=device)
    acoustic = batch.acoustic_codes.to(device=device)
    acoustic_mask = batch.target_acoustic_mask.to(device=device)
    reference_semantic = _reference_semantic(batch).to(device=device)
    reference_semantic_mask = _reference_mask(batch).to(device=device)
    reference_acoustic = _reference_acoustic(batch).to(device=device)
    reference_acoustic_mask = _reference_acoustic_mask(batch).to(device=device)

    target_features = _codec_features(backend, acoustic, acoustic_mask)
    reference_features = _codec_features(
        backend,
        reference_acoustic,
        reference_acoustic_mask,
    )
    without_features = runtime.sample_features(
        semantic,
        mask=semantic_mask,
        reference_features=None,
        reference_mask=None,
        generator=_generator(device=device, seed=seed),
    )
    with_features = runtime.sample_features(
        semantic,
        mask=semantic_mask,
        reference_features=reference_features,
        reference_mask=reference_acoustic_mask,
        generator=_generator(device=device, seed=seed),
    )
    without_mse = _feature_mse(without_features, target_features, acoustic_mask)
    with_mse = _feature_mse(with_features, target_features, acoustic_mask)

    target_codes = SemanticAcousticCodes(
        semantic=semantic.masked_fill(~semantic_mask[..., None], 0),
        acoustic=acoustic.masked_fill(~acoustic_mask[..., None], 0),
    )
    reference_codes = SemanticAcousticCodes(
        semantic=reference_semantic.masked_fill(~reference_semantic_mask[..., None], 0),
        acoustic=reference_acoustic.masked_fill(~reference_acoustic_mask[..., None], 0),
    )
    audio = {
        "generated_without_reference": runtime.decode_features(
            target_codes.semantic,
            without_features,
        ),
        "generated_with_reference": runtime.decode_features(
            target_codes.semantic,
            with_features,
        ),
        "target_reconstruction": backend.detokenize(target_codes),
        "reference_reconstruction": backend.detokenize(reference_codes),
    }
    if batch.acoustic_layout is AcousticLayout.FIXED_LENGTH:
        audio["reference_token_passthrough"] = backend.detokenize(
            SemanticAcousticCodes(
                semantic=target_codes.semantic,
                acoustic=reference_codes.acoustic,
            )
        )
    return audio, {
        "feature_mse_without_reference": without_mse,
        "feature_mse_with_reference": with_mse,
        "reference_gain": without_mse - with_mse,
    }


def _summary(
    audio: dict[str, Tensor],
    metrics: dict[str, float],
    *,
    sample_rate: int,
    args: argparse.Namespace,
    batch: SemanticCodecBatch,
) -> dict[str, Any]:
    reference_semantic = _reference_semantic(batch)
    reference_acoustic = _reference_acoustic(batch)
    summaries = {
        name: _waveform_summary(value, sample_rate=sample_rate)
        for name, value in audio.items()
    }
    result = {
        "artifact": str(args.artifact),
        "codec": str(args.codec),
        "data_root": None if args.data_root is None else str(args.data_root),
        "data_source": str(args.data_source),
        "sample_index": int(args.sample_index),
        "sample_rate": sample_rate,
        "target_semantic_shape": list(batch.semantic_codes.shape),
        "target_acoustic_shape": list(batch.acoustic_codes.shape),
        "reference_semantic_shape": list(reference_semantic.shape),
        "reference_acoustic_shape": list(reference_acoustic.shape),
        "pair": asdict(batch.metadata[0]),
        **metrics,
        **summaries,
    }
    result.setdefault("reference_token_passthrough", None)
    return result


def _write_outputs(
    audio: dict[str, Tensor],
    result: dict[str, Any],
    *,
    args: argparse.Namespace,
    sample_rate: int,
) -> None:
    paths = {
        "generated_without_reference": args.without_reference_wav,
        "generated_with_reference": args.with_reference_wav,
        "target_reconstruction": args.target_reconstruction_wav,
        "reference_reconstruction": args.reference_reconstruction_wav,
        "reference_token_passthrough": args.passthrough_wav,
    }
    if args.passthrough_wav is not None and "reference_token_passthrough" not in audio:
        raise ValueError("--passthrough-wav requires a fixed-length acoustic layout.")
    for name, path in paths.items():
        if path is None:
            continue
        _write_wav(path, audio[name], sample_rate=sample_rate)
        cast(dict[str, Any], result[name])["output_wav"] = str(path)


def _codec_features(
    backend: SemanticAcousticCodec,
    codes: Tensor,
    mask: Tensor,
) -> Tensor:
    features = backend.acoustic_codes_to_features(codes.masked_fill(~mask[..., None], 0))
    feature_mask = mask.to(device=features.device)
    return features.masked_fill(~feature_mask[..., None], 0)


def _feature_mse(generated: Tensor, target: Tensor, mask: Tensor) -> float:
    if generated.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError(
            "artifact feature tensors must align: "
            f"generated={tuple(generated.shape)}, target={tuple(target.shape)}, "
            f"mask={tuple(mask.shape)}"
        )
    value = (generated.float() - target.float()).pow(2)[mask].mean()
    if not bool(torch.isfinite(value).detach().cpu()):
        raise ValueError("artifact feature MSE must be finite.")
    return float(value.detach().cpu())


def _reference_semantic(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_semantic_codes
    if value is None:
        raise RuntimeError("reference_semantic_codes are required for artifact evaluation.")
    return value


def _reference_acoustic(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_acoustic_codes
    if value is None:
        raise RuntimeError("reference_acoustic_codes are required for artifact evaluation.")
    return value


def _reference_mask(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_mask
    if value is None:
        raise RuntimeError("reference_mask is required for artifact evaluation.")
    return value


def _reference_acoustic_mask(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_acoustic_mask
    if value is None:
        raise RuntimeError("reference_acoustic_mask is required for artifact evaluation.")
    return value


def _waveform_summary(waveform: Tensor, *, sample_rate: int) -> dict[str, Any]:
    audio = waveform.detach().float().cpu()
    finite = bool(torch.isfinite(audio).all())
    if audio.dim() != 3:
        raise ValueError("decoded waveform must have shape [batch, channel, samples].")
    samples = int(audio.size(-1))
    return {
        "finite": finite,
        "seconds": samples / sample_rate,
        "waveform_max": float(audio.max()),
        "waveform_min": float(audio.min()),
        "waveform_rms": float(audio.square().mean().sqrt()),
        "waveform_shape": list(audio.shape),
    }


def _write_wav(path: Path, waveform: Tensor, *, sample_rate: int) -> None:
    audio = waveform.detach().float().cpu()[0]
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
