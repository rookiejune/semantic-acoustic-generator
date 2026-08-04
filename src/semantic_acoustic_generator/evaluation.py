from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from anytrain.codec import SemanticAcousticCodes, masked_acoustic_features
from torch import Tensor

from semantic_acoustic_generator.backend.adapter import LongCatFirstCodebookAdapter
from semantic_acoustic_generator.types import GeneratorBatch

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec

    from semantic_acoustic_generator.runtime.semantic import GeneratorRuntime


@dataclass(frozen=True)
class PairedFeatureEvaluation:
    without_reference: Tensor
    with_reference: Tensor
    mse_without_reference: float
    mse_with_reference: float

    @property
    def reference_gain(self) -> float:
        return self.mse_without_reference - self.mse_with_reference


@dataclass(frozen=True)
class FirstCodebookOracleEvaluation:
    audio: dict[str, Tensor]
    metrics: dict[str, object]


@torch.no_grad()
def target_acoustic_features(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    validate: bool = True,
) -> Tensor:
    return masked_acoustic_features(
        backend,
        batch.acoustic_codes,
        batch.acoustic_mask,
        validate=validate,
    )


@torch.no_grad()
def reference_acoustic_condition(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    validate: bool = True,
) -> tuple[Tensor, Tensor]:
    if not batch.has_reference:
        raise ValueError("reference acoustic condition requires reference codec units.")
    reference = batch.reference
    mask = reference.acoustic_mask.to(device=batch.semantic_codes.device)
    features = masked_acoustic_features(
        backend,
        reference.acoustic_codes,
        mask,
        validate=validate,
    )
    return features, mask


def masked_feature_mse(
    generated: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    name: str,
) -> float:
    if generated.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError(
            f"{name} feature tensors must align: "
            f"generated={tuple(generated.shape)}, target={tuple(target.shape)}, "
            f"mask={tuple(mask.shape)}"
        )
    value = (generated.float() - target.float()).pow(2)[mask].mean()
    if not bool(torch.isfinite(value).detach().cpu()):
        raise ValueError(f"{name} feature MSE must be finite.")
    return float(value.detach().cpu())


@torch.no_grad()
def evaluate_feature_pair(
    runtime: GeneratorRuntime,
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    seed: int,
    cfg_scale: float | None = None,
    name: str,
) -> PairedFeatureEvaluation:
    if not batch.has_reference:
        raise ValueError(f"{name} evaluation requires reference codec units.")
    target = target_acoustic_features(backend, batch)
    reference, reference_mask = reference_acoustic_condition(backend, batch)
    device = batch.semantic_codes.device
    without = runtime.sample_features(
        batch.semantic_codes,
        mask=batch.mask,
        reference_features=None,
        reference_mask=None,
        generator=seeded_generator(device, seed),
    )
    with_reference = runtime.sample_features(
        batch.semantic_codes,
        mask=batch.mask,
        reference_features=reference,
        reference_mask=reference_mask,
        cfg_scale=cfg_scale,
        generator=seeded_generator(device, seed),
    )
    return PairedFeatureEvaluation(
        without_reference=without,
        with_reference=with_reference,
        mse_without_reference=masked_feature_mse(
            without,
            target,
            batch.acoustic_mask,
            name=name,
        ),
        mse_with_reference=masked_feature_mse(
            with_reference,
            target,
            batch.acoustic_mask,
            name=name,
        ),
    )


def seeded_generator(device: torch.device | str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def evaluate_first_codebook_oracle(
    backend: LongCatFirstCodebookAdapter,
    batch: GeneratorBatch,
    *,
    sigmas: tuple[float, ...],
    seed: int,
) -> FirstCodebookOracleEvaluation:
    if batch.semantic_codes.size(0) != 1:
        raise ValueError("first-codebook oracle requires one sample per batch.")
    if any(value <= 0 for value in sigmas):
        raise ValueError("first-codebook oracle sigmas must be positive.")
    mask = batch.acoustic_mask.to(device=batch.acoustic_codes.device)
    semantic = batch.semantic_codes.masked_fill(~batch.mask[..., None], 0)
    acoustic = batch.acoustic_codes.masked_fill(~mask[..., None], 0)
    target = backend.acoustic_codes_to_features(acoustic)
    native = backend.native_stage0_features(acoustic)
    projected = backend.project_features(target)
    target_factors = backend.factor_codes(acoustic)
    full_codes = SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)
    audio = {
        "full_reconstruction": backend.backend.detokenize(full_codes),
        "stage0_code_reconstruction": backend.backend.decode_features(semantic, native),
        "exact_16d_reconstruction": backend.decode_features(semantic, target),
    }
    metrics: dict[str, object] = {
        "native_projection_max_abs": float((native - projected).abs().max().cpu()),
        "native_projection_mse": float((native.float() - projected.float()).square().mean().cpu()),
        "exact_snap_max_abs": float((backend.snap_features(target) - target).abs().max().cpu()),
        "groups": {},
    }
    group_metrics = metrics["groups"]
    if not isinstance(group_metrics, dict):
        raise RuntimeError("oracle group metrics must be a dict.")
    scale = torch.cat(
        tuple(codebook.float().std(dim=0, correction=0) for codebook in backend.factor_codebooks)
    ).view(1, 1, -1)
    generator = seeded_generator(target.device, seed)
    for sigma in sigmas:
        noise = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        raw = target + noise * scale.to(device=target.device, dtype=target.dtype) * sigma
        raw = raw.masked_fill(~mask.to(device=raw.device)[..., None], 0)
        snapped = backend.snap_features(raw)
        key = _sigma_key(sigma)
        raw_name = f"raw_sigma_{key}"
        snap_name = f"snap_sigma_{key}"
        audio[raw_name] = backend.decode_features(semantic, raw)
        audio[snap_name] = backend.decode_features(semantic, snapped)
        predicted_factors = backend.features_to_factor_codes(raw)
        valid = mask.to(device=predicted_factors.device)
        factor_accuracy = predicted_factors[valid].eq(
            target_factors.to(device=predicted_factors.device)[valid]
        ).float().mean(dim=0)
        group_metrics[raw_name] = {
            "feature_mse": float((raw.float() - target.float())[valid].square().mean().cpu()),
            "factor_a_accuracy": float(factor_accuracy[0].cpu()),
            "factor_b_accuracy": float(factor_accuracy[1].cpu()),
        }
        group_metrics[snap_name] = {
            "feature_mse": float(
                (snapped.float() - target.float())[valid].square().mean().cpu()
            ),
            "factor_a_accuracy": float(factor_accuracy[0].cpu()),
            "factor_b_accuracy": float(factor_accuracy[1].cpu()),
        }
    return FirstCodebookOracleEvaluation(audio=audio, metrics=metrics)


def _sigma_key(value: float) -> str:
    return format(value, ".6g").replace("-", "m").replace(".", "p")


__all__ = [
    "PairedFeatureEvaluation",
    "FirstCodebookOracleEvaluation",
    "evaluate_first_codebook_oracle",
    "evaluate_feature_pair",
    "masked_feature_mse",
    "reference_acoustic_condition",
    "seeded_generator",
    "target_acoustic_features",
]
