"""Shared contracts and tensor invariants for acoustic-unit generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import nn

from semantic_acoustic_generator.config import Route

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class DecoderLoss:
    """Generator step loss with named scalar component losses."""

    loss: Tensor
    losses: dict[str, Tensor]
    primary: str


@runtime_checkable
class FeatureSampler(Protocol):
    def sample_features(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        flow_steps: int,
        unconditional_condition: Tensor | None = None,
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


@runtime_checkable
class AcousticCodeSampler(Protocol):
    def sample_acoustic_codes(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


class AcousticHead(nn.Module):
    route: Route


AcousticUnitGenerator = AcousticHead


def aligned_condition(
    condition: Tensor,
    mask: Tensor,
    *,
    target_mask: Tensor | None = None,
    validate: bool = True,
) -> tuple[Tensor, Tensor]:
    if validate and (
        condition.dim() != 3 or mask.shape != condition.shape[:2] or mask.dtype != torch.bool
    ):
        raise ValueError("condition and mask must have shapes [B, semantic_unit, C] and [B, unit].")
    if validate and not bool(mask.any(dim=1).all()):
        raise ValueError("each condition row must contain at least one valid semantic unit.")
    if target_mask is not None:
        if target_mask.shape != condition.shape[:2] or target_mask.dtype != torch.bool:
            raise ValueError("acoustic target mask must align with semantic frames.")
        if not torch.equal(mask, target_mask):
            raise ValueError("semantic and acoustic masks must match frame by frame.")
    return condition, mask


def normalized_features(
    features: Tensor,
    mask: Tensor,
    feature_mean: Tensor | None,
    feature_std: Tensor | None,
) -> Tensor:
    if features.dim() != 3 or mask.shape != features.shape[:2]:
        raise ValueError("acoustic target features and mask must align on [B, acoustic_unit].")
    if (feature_mean is None) != (feature_std is None):
        raise ValueError("feature_mean and feature_std must be set together.")
    if feature_mean is None or feature_std is None:
        return features
    features = features.to(device=feature_mean.device, dtype=feature_mean.dtype)
    return (features - feature_mean) / feature_std


__all__ = [
    "AcousticCodeSampler",
    "AcousticHead",
    "AcousticUnitGenerator",
    "DecoderLoss",
    "FeatureSampler",
]
