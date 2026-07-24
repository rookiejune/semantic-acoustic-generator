from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from semantic_acoustic_codec.backend import LongCatBackend
from semantic_acoustic_codec.config import DecoderConfig, Route
from semantic_acoustic_codec.data import SemanticCodecBatch, collate
from semantic_acoustic_codec.loss import FlowLoss, RVQLoss
from semantic_acoustic_codec.model import RectifiedFlowRuntime, backend_features, build_route
from semantic_acoustic_codec.runtime import (
    SemanticSupportConfig,
    build_support,
    load_artifact,
    save_artifact,
)


class FakeEncoder:
    input_sample_rate = 16_000
    hop_length = 320


class FakeDecoder:
    latent_dim = 4


class FakeCodec:
    sample_rate = 16_000
    encoder = FakeEncoder()
    decoders = {"default": FakeDecoder()}
    semantic_codebook = torch.randn(16, 8)
    codebook_sizes = (16, 7, 11)

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        del sample_rate
        return audio.new_zeros((audio.size(0), 2, 3), dtype=torch.long)

    def decode(self, codes: Tensor) -> Tensor:
        return codes.new_zeros((codes.size(0), 1, codes.size(1) * 320), dtype=torch.float32)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        return torch.nn.functional.pad(acoustic_codes.float(), (0, 2))[:, :, :4]

    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor:
        return acoustic_features.new_zeros((semantic_codes.size(0), 1, semantic_codes.size(1) * 320))


def main() -> None:
    args = _args()
    torch.manual_seed(args.seed)
    backend = LongCatBackend(FakeCodec())
    decoder = DecoderConfig(hidden_dim=12, layers=1, heads=2, ffn_ratio=2)

    _route_smoke(backend, decoder, routes=_routes(args.routes))
    _artifact_smoke(backend, decoder)
    if args.data_root is not None:
        _data_smoke(args.data_root, split=args.split, index=args.index)
    elif args.require_data:
        raise ValueError("real data smoke requires --data-root or WMT19_TTS_ROOT.")
    else:
        print("data smoke skipped: pass --data-root or set WMT19_TTS_ROOT")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic-acoustic-codec smoke checks.")
    parser.add_argument(
        "--routes",
        nargs="+",
        default=[route.value for route in Route],
        choices=[route.value for route in Route],
        help="Decoder routes to smoke.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_env_path("WMT19_TTS_ROOT"),
        help="Prepared WMT19 TTS root containing the LongCat store.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if value is None or not value else Path(value)


def _routes(values: Iterable[str]) -> tuple[Route, ...]:
    return tuple(Route(value) for value in values)


def _batch() -> tuple[Tensor, Tensor, Tensor]:
    semantic = torch.tensor(
        [
            [[1], [2], [3]],
            [[4], [5], [0]],
        ],
        dtype=torch.long,
    )
    acoustic = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[6, 10], [2, 1], [0, 0]],
        ],
        dtype=torch.long,
    )
    mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ],
        dtype=torch.bool,
    )
    return semantic, acoustic, mask


def _route_smoke(
    backend: LongCatBackend,
    decoder: DecoderConfig,
    *,
    routes: tuple[Route, ...],
) -> None:
    semantic, acoustic, mask = _batch()
    target = backend_features(backend, acoustic, mask)
    for route in routes:
        if route is Route.RVQ and importlib.util.find_spec("transformers") is None:
            print("rvq smoke skipped: transformers is not installed")
            continue
        modules = build_route(route, backend, condition_dim=12, decoder=decoder)
        optimizer = torch.optim.AdamW(
            list(modules.conditioner.parameters())
            + list(modules.reference_conditioner.parameters())
            + list(modules.generator.parameters()),
            lr=1e-3,
        )
        reference = modules.reference_conditioner(target, mask=mask, batch_size=semantic.size(0))
        condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
        if route is Route.FM:
            item = FlowLoss()(modules.generator.core, condition, target, mask, RectifiedFlowRuntime())
        elif route is Route.RVQ:
            logits = modules.generator.core(condition, acoustic, mask=mask)
            item = RVQLoss()(logits, acoustic, mask)
        else:
            raise AssertionError(f"unsupported route: {route}")
        loss = item.loss.mean()
        loss.backward()
        optimizer.step()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{route.value} smoke produced a non-finite loss.")
        print(f"{route.value} train smoke ok: loss={float(loss.detach()):.6f}")


def _artifact_smoke(backend: LongCatBackend, decoder: DecoderConfig) -> None:
    semantic, _, mask = _batch()
    config = SemanticSupportConfig(route=Route.FM, condition_dim=12, decoder=decoder)
    support = build_support(backend, config).eval()
    features = support.sample_features(semantic, mask=mask, generator=torch.Generator().manual_seed(0))
    waveform = support.decode(semantic, mask=mask, generator=torch.Generator().manual_seed(1))
    if features.shape != (*semantic.shape[:2], backend.acoustic_feature_dim):
        raise RuntimeError("artifact smoke produced an unexpected feature shape.")
    if waveform.dim() != 3 or not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("artifact smoke produced an invalid waveform.")
    with tempfile.TemporaryDirectory(prefix="semantic-acoustic-codec-") as tmp:
        save_artifact(tmp, support, config)
        loaded = load_artifact(tmp, backend=backend)
        loaded_features = loaded.sample_features(semantic, mask=mask, generator=torch.Generator().manual_seed(0))
    if not torch.allclose(features, loaded_features):
        raise RuntimeError("artifact smoke changed seeded FM features.")
    print(f"artifact smoke ok: waveform_shape={tuple(waveform.shape)}")


def _data_smoke(root: Path, *, split: str, index: int) -> None:
    from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

    dataset = wmt19_tts_codec(codec="longcat", root=root, split=split)
    sample = dataset[index]
    batch = collate([sample])
    _validate_batch(batch)
    print(
        "data smoke ok: "
        f"split={split} index={index} semantic_shape={tuple(batch.semantic_codes.shape)} "
        f"acoustic_shape={tuple(batch.acoustic_codes.shape)} frames={int(batch.mask.sum())}"
    )


def _validate_batch(batch: SemanticCodecBatch) -> None:
    if batch.semantic_codes.size(0) != 1:
        raise RuntimeError("single-sample data smoke should produce batch size 1.")
    if batch.acoustic_codes.size(-1) < 1:
        raise RuntimeError("real LongCat batch must contain acoustic codebooks.")
    if not bool(batch.mask.any()):
        raise RuntimeError("real LongCat batch contains no valid frames.")
    if not bool((batch.semantic_codes[batch.mask] >= 0).all()):
        raise RuntimeError("valid semantic codes must be non-negative.")
    if not bool((batch.acoustic_codes[batch.mask] >= 0).all()):
        raise RuntimeError("valid acoustic codes must be non-negative.")


if __name__ == "__main__":
    main()
