from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from anytrain.codec import masked_acoustic_features
from torch import Tensor

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


__all__ = [
    "PairedFeatureEvaluation",
    "evaluate_feature_pair",
    "masked_feature_mse",
    "reference_acoustic_condition",
    "seeded_generator",
    "target_acoustic_features",
]
