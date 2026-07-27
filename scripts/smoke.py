from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from anytrain.codec import (
    AcousticLayout,
    SemanticAcousticCodes,
    load_semantic_acoustic,
    masked_acoustic_features,
)
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from anytrain.loss import MaskedCodebookCrossEntropyLoss

from semantic_acoustic_codec.config import DecoderConfig, Route
from semantic_acoustic_codec.datamodule import DataConfig, LBAConfig, load_batch
from semantic_acoustic_codec.loss import FlowLoss
from semantic_acoustic_codec.model import RVQCodeGenerator, build_route
from semantic_acoustic_codec.runtime import (
    SemanticCodecRuntime,
    SemanticSupportConfig,
    build_support,
    load_artifact,
    save_artifact,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from semantic_acoustic_codec.types import SemanticCodecBatch


class FakeEncoder:
    input_sample_rate = 16_000
    hop_length = 320


class FakeDecoder:
    latent_dim = 4


class FakeCodec:
    name = "fake"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None
    encoder = FakeEncoder()
    decoders = {"default": FakeDecoder()}
    semantic_codebook = torch.randn(16, 8)
    semantic_codebook_sizes = (16,)
    acoustic_codebook_sizes = (7, 11)
    acoustic_feature_dim = 4

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        semantic = audio.new_zeros((audio.size(0), 2, 1), dtype=torch.long)
        acoustic = audio.new_zeros((audio.size(0), 2, 2), dtype=torch.long)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor:
        return codes.semantic.new_zeros(
            (codes.semantic.size(0), 1, codes.semantic.size(1) * 320),
            dtype=torch.float32,
        )

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        return torch.nn.functional.pad(acoustic_codes.float(), (0, 2))[:, :, :4]

    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor:
        return acoustic_features.new_zeros((semantic_codes.size(0), 1, semantic_codes.size(1) * 320))


def main() -> None:
    args = _args()
    torch.manual_seed(args.seed)
    backend = FakeCodec()
    decoder = DecoderConfig(hidden_dim=12, layers=1, heads=2, ffn_ratio=2)

    _route_smoke(backend, decoder, routes=_routes(args.routes))
    _artifact_smoke(backend, decoder)
    if args.data_root is not None or args.require_data:
        _data_smoke(
            args.data_root,
            codec=args.codec,
            source=args.data_source,
            split=args.split,
            index=args.index,
            device=args.device,
        )
    else:
        print("data smoke skipped: pass --data-root or --require-data")


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
        default=None,
        help="Prepared codec grid root; workspace resolves the standard root when omitted.",
    )
    parser.add_argument("--data-source", default="qwen_cross_text")
    parser.add_argument("--codec", default="longcat", choices=("longcat", "bicodec"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _routes(values: Iterable[str]) -> tuple[Route, ...]:
    return tuple(Route(value) for value in values)


def _batch() -> tuple[Tensor, Tensor, Tensor]:
    semantic = torch.tensor(
        [
            [[1], [2], [3]],
            [[4], [5], [16]],
        ],
        dtype=torch.long,
    )
    acoustic = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[6, 10], [2, 1], [7, 11]],
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


def _reference_batch() -> tuple[Tensor, Tensor, Tensor]:
    semantic = torch.tensor(
        [
            [[6], [7], [16]],
            [[8], [9], [10]],
        ],
        dtype=torch.long,
    )
    acoustic = torch.tensor(
        [
            [[2, 8], [4, 6], [7, 11]],
            [[0, 9], [5, 7], [3, 5]],
        ],
        dtype=torch.long,
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ],
        dtype=torch.bool,
    )
    return semantic, acoustic, mask


def _route_smoke(
    backend: FakeCodec,
    decoder: DecoderConfig,
    *,
    routes: tuple[Route, ...],
) -> None:
    semantic, acoustic, mask = _batch()
    _, reference_acoustic, reference_mask = _reference_batch()
    target = masked_acoustic_features(backend, acoustic, mask)
    reference_target = masked_acoustic_features(backend, reference_acoustic, reference_mask)
    for route in routes:
        if route is Route.RVQ and importlib.util.find_spec("transformers") is None:
            print("rvq smoke skipped: transformers is not installed")
            continue
        modules = build_route(
            route,
            backend.semantic_codebook,
            backend.acoustic_feature_dim,
            backend.acoustic_codebook_sizes,
            condition_dim=12,
            decoder=decoder,
        )
        optimizer = torch.optim.AdamW(
            list(modules.conditioner.parameters())
            + list(modules.reference_conditioner.parameters())
            + list(modules.generator.parameters()),
            lr=1e-3,
        )
        reference = modules.reference_conditioner(
            reference_target,
            mask=reference_mask,
            batch_size=semantic.size(0),
        )
        condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
        if route is Route.FM:
            item = FlowLoss()(
                modules.generator.core,
                condition,
                target,
                mask,
                ContinuousFlowRuntime(),
            )
        elif route is Route.RVQ:
            generator = modules.generator
            if not isinstance(generator, RVQCodeGenerator):
                raise RuntimeError("RVQ route built a non-RVQ generator.")
            logits = generator.core(condition, acoustic, mask=mask)
            item = MaskedCodebookCrossEntropyLoss()(logits, acoustic, mask)
        else:
            raise AssertionError(f"unsupported route: {route}")
        loss = item.loss.mean()
        loss.backward()
        optimizer.step()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{route.value} smoke produced a non-finite loss.")
        print(f"{route.value} train smoke ok: loss={float(loss.detach()):.6f}")


def _artifact_smoke(backend: FakeCodec, decoder: DecoderConfig) -> None:
    semantic, _, mask = _batch()
    _, reference_acoustic, reference_mask = _reference_batch()
    reference_features = masked_acoustic_features(backend, reference_acoustic, reference_mask)
    config = SemanticSupportConfig(route=Route.FM, condition_dim=12, decoder=decoder)
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    ).eval()
    runtime = SemanticCodecRuntime(support, backend)
    without_features = support.sample_features(
        semantic,
        mask=mask,
        reference_features=None,
        reference_mask=None,
        generator=_generator(0),
    )
    with_features = support.sample_features(
        semantic,
        mask=mask,
        reference_features=reference_features,
        reference_mask=reference_mask,
        generator=_generator(0),
    )
    without_waveform = runtime.decode(
        semantic,
        mask=mask,
        reference_features=None,
        reference_mask=None,
        generator=_generator(1),
    )
    with_waveform = runtime.decode(
        semantic,
        mask=mask,
        reference_features=reference_features,
        reference_mask=reference_mask,
        generator=_generator(1),
    )
    expected_shape = (*semantic.shape[:2], backend.acoustic_feature_dim)
    if without_features.shape != expected_shape or with_features.shape != expected_shape:
        raise RuntimeError("artifact smoke produced an unexpected feature shape.")
    for name, waveform in (
        ("without-reference", without_waveform),
        ("with-reference", with_waveform),
    ):
        if waveform.dim() != 3 or not bool(torch.isfinite(waveform).all()):
            raise RuntimeError(f"artifact {name} smoke produced an invalid waveform.")
    with tempfile.TemporaryDirectory(prefix="semantic-acoustic-codec-") as tmp:
        save_artifact(tmp, support, config)
        loaded = load_artifact(tmp)
        loaded_without = loaded.sample_features(
            semantic,
            mask=mask,
            reference_features=None,
            reference_mask=None,
            generator=_generator(0),
        )
        loaded_with = loaded.sample_features(
            semantic,
            mask=mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            generator=_generator(0),
        )
    if not torch.allclose(without_features, loaded_without):
        raise RuntimeError("artifact smoke changed seeded null-reference FM features.")
    if not torch.allclose(with_features, loaded_with):
        raise RuntimeError("artifact smoke changed seeded reference-conditioned FM features.")
    print(
        "artifact smoke ok: "
        f"without_shape={tuple(without_waveform.shape)} "
        f"with_shape={tuple(with_waveform.shape)}"
    )


def _data_smoke(
    root: Path | None,
    *,
    codec: str,
    source: str,
    split: str,
    index: int,
    device: str,
) -> None:
    backend = load_semantic_acoustic(codec, device=device)
    data = DataConfig(
        source=source,
        root=None if root is None else str(root),
        split=split,
        sample_index=index,
        lba=LBAConfig(enabled=False),
    )
    batch = load_batch(
        data,
        codec=codec,
        frame_rate=backend.frame_rate,
        acoustic_layout=backend.acoustic_layout,
        semantic_pad_id=int(backend.semantic_codebook.size(0)),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    if source == "qwen_cross_text" and not batch.has_reference:
        raise RuntimeError("qwen_cross_text data smoke requires a target/reference pair.")
    _validate_batch(batch)
    target = masked_acoustic_features(
        backend,
        batch.acoustic_codes.to(device),
        batch.target_acoustic_mask.to(device),
    )
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("real target acoustic features must be finite.")
    reference_shape: tuple[int, ...] | None = None
    if batch.has_reference:
        reference_acoustic = _reference_acoustic(batch).to(device)
        reference_mask = _reference_acoustic_mask(batch).to(device)
        reference = masked_acoustic_features(backend, reference_acoustic, reference_mask)
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError("real reference acoustic features must be finite.")
        reference_shape = tuple(reference.shape)
    print(
        "data smoke ok: "
        f"source={source} codec={codec} split={split} index={index} "
        f"semantic_shape={tuple(batch.semantic_codes.shape)} "
        f"acoustic_shape={tuple(batch.acoustic_codes.shape)} "
        f"reference_feature_shape={reference_shape} frames={int(batch.mask.sum())}"
    )


def _validate_batch(batch: SemanticCodecBatch) -> None:
    if batch.semantic_codes.size(0) != 1:
        raise RuntimeError("single-sample data smoke should produce batch size 1.")
    if batch.acoustic_codes.size(-1) < 1:
        raise RuntimeError("real codec batch must contain acoustic codebooks.")
    if not bool(batch.mask.any()):
        raise RuntimeError("real codec batch contains no valid semantic frames.")
    if not bool((batch.semantic_codes[batch.mask] >= 0).all()):
        raise RuntimeError("valid semantic codes must be non-negative.")
    if not bool((batch.acoustic_codes[batch.target_acoustic_mask] >= 0).all()):
        raise RuntimeError("valid acoustic codes must be non-negative.")
    if batch.has_reference:
        reference_semantic = _reference_semantic(batch)
        reference_acoustic = _reference_acoustic(batch)
        reference_mask = _reference_mask(batch)
        reference_acoustic_mask = _reference_acoustic_mask(batch)
        if not bool((reference_semantic[reference_mask] >= 0).all()):
            raise RuntimeError("valid reference semantic codes must be non-negative.")
        if not bool((reference_acoustic[reference_acoustic_mask] >= 0).all()):
            raise RuntimeError("valid reference acoustic codes must be non-negative.")


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _reference_semantic(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_semantic_codes
    if value is None:
        raise RuntimeError("reference_semantic_codes are required for paired smoke data.")
    return value


def _reference_acoustic(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_acoustic_codes
    if value is None:
        raise RuntimeError("reference_acoustic_codes are required for paired smoke data.")
    return value


def _reference_mask(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_mask
    if value is None:
        raise RuntimeError("reference_mask is required for paired smoke data.")
    return value


def _reference_acoustic_mask(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_acoustic_mask
    if value is None:
        raise RuntimeError("reference_acoustic_mask is required for paired smoke data.")
    return value


if __name__ == "__main__":
    main()
