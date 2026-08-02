from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from anytrain.codec import masked_acoustic_features
from torch import Tensor

from semantic_acoustic_codec.types import SemanticCodecBatch

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec


@torch.no_grad()
def target_acoustic_features(
    backend: SemanticAcousticCodec,
    batch: SemanticCodecBatch,
    *,
    validate: bool = True,
) -> Tensor:
    return masked_acoustic_features(
        backend,
        batch.acoustic_codes,
        batch.target_acoustic_mask,
        validate=validate,
    )


@torch.no_grad()
def reference_acoustic_condition(
    backend: SemanticAcousticCodec,
    batch: SemanticCodecBatch,
    *,
    validate: bool = True,
) -> tuple[Tensor | None, Tensor | None]:
    if not batch.has_reference:
        return None, None
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


def seeded_generator(device: torch.device | str, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        generator = torch.Generator()
    return generator.manual_seed(seed)


__all__ = [
    "masked_feature_mse",
    "reference_acoustic_condition",
    "seeded_generator",
    "target_acoustic_features",
]
