from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from anytrain.codec import SemanticAcousticCodes

from semantic_acoustic_generator.backend import BackendConfig, load_backend
from semantic_acoustic_generator.datamodule import (
    BatchingConfig,
    DataConfig,
    load_batch,
)
from semantic_acoustic_generator.evaluation import (
    evaluate_feature_pair,
    waveform_summary,
    write_pcm16_wav,
)
from semantic_acoustic_generator.runtime import GeneratorRuntime
from semantic_acoustic_generator.runtime.artifact import load_artifact

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec
    from torch import Tensor

    from semantic_acoustic_generator.types import GeneratorBatch


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    backend = load_backend(BackendConfig(name=str(args.codec)), device=device)
    support = load_artifact(args.artifact, device=device)
    data = DataConfig(
        source=args.data_source,
        root=None if args.data_root is None else str(args.data_root),
        split=args.split,
        sample_index=args.sample_index,
        max_seconds=args.max_seconds,
        overlong=args.overlong,
        batching=BatchingConfig(enabled=False),
    )
    batch = load_batch(
        data,
        codec=args.codec,
        frame_rate=backend.frame_rate,
        acoustic_layout=backend.acoustic_layout,
        semantic_pad_id=int(backend.semantic_codebook.size(0)),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    runtime = GeneratorRuntime(support, backend)
    audio, metrics = _evaluate(
        runtime,
        runtime.backend,
        batch,
        device=device,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
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
    parser = argparse.ArgumentParser(
        description="Decode a semantic-acoustic artifact on one sample."
    )
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
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for with-reference FM sampling.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--include-private-metadata",
        action="store_true",
        help="Write raw text, speaker IDs, utterance IDs, and local paths to JSON output.",
    )
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
    return parser.parse_args()


@torch.no_grad()
def _evaluate(
    runtime: GeneratorRuntime,
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    device: torch.device,
    seed: int,
    cfg_scale: float = 1.0,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    if not batch.has_reference or len(batch.metadata) != 1:
        raise ValueError("artifact evaluation requires one cross-text target/reference pair.")
    batch = batch.to(device)
    semantic = batch.semantic_codes
    semantic_mask = batch.mask
    acoustic = batch.acoustic_codes
    acoustic_mask = batch.acoustic_mask
    reference = batch.reference
    reference_semantic = reference.semantic_codes
    reference_semantic_mask = reference.mask
    reference_acoustic = reference.acoustic_codes
    reference_acoustic_mask = reference.acoustic_mask

    evaluation = evaluate_feature_pair(
        runtime,
        backend,
        batch,
        seed=seed,
        cfg_scale=cfg_scale,
        name="artifact",
    )

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
            evaluation.without_reference,
        ),
        "generated_with_reference": runtime.decode_features(
            target_codes.semantic,
            evaluation.with_reference,
        ),
        "target_reconstruction": backend.detokenize(target_codes),
        "reference_reconstruction": backend.detokenize(reference_codes),
    }
    return audio, {
        "feature_mse_without_reference": evaluation.mse_without_reference,
        "feature_mse_with_reference": evaluation.mse_with_reference,
        "reference_gain": evaluation.reference_gain,
    }


def _summary(
    audio: dict[str, Tensor],
    metrics: dict[str, float],
    *,
    sample_rate: int,
    args: argparse.Namespace,
    batch: GeneratorBatch,
) -> dict[str, Any]:
    reference = batch.reference
    reference_semantic = reference.semantic_codes
    reference_acoustic = reference.acoustic_codes
    summaries = {
        name: waveform_summary(value, sample_rate=sample_rate) for name, value in audio.items()
    }
    result = {
        "artifact": _path_value(args.artifact, include_private=_include_private(args)),
        "codec": str(args.codec),
        "data_root": (
            None
            if args.data_root is None
            else _path_value(args.data_root, include_private=_include_private(args))
        ),
        "data_source": str(args.data_source),
        "sample_index": int(args.sample_index),
        "sample_rate": sample_rate,
        "cfg_scale": float(args.cfg_scale),
        "target_semantic_shape": list(batch.semantic_codes.shape),
        "target_acoustic_shape": list(batch.acoustic_codes.shape),
        "reference_semantic_shape": list(reference_semantic.shape),
        "reference_acoustic_shape": list(reference_acoustic.shape),
        "pair": batch.metadata[0].as_dict(include_private=_include_private(args)),
        **metrics,
        **summaries,
    }
    return result


def _path_value(path: Path, *, include_private: bool) -> str:
    return str(path) if include_private else path.name


def _include_private(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "include_private_metadata", False))


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
    }
    for name, path in paths.items():
        if path is None:
            continue
        write_pcm16_wav(path, audio[name], sample_rate=sample_rate)
        cast(dict[str, Any], result[name])["output_wav"] = str(path)


if __name__ == "__main__":
    main()
