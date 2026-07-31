from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class SACCandidate:
    sample_id: int
    group_id: int
    candidate_id: int
    semantic_codes: Tensor
    semantic_mask: Tensor
    acoustic_features: Tensor | None = None
    acoustic_codes: Tensor | None = None
    acoustic_mask: Tensor | None = None
    waveform: Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_negative(self.sample_id, name="sample_id")
        _non_negative(self.group_id, name="group_id")
        _non_negative(self.candidate_id, name="candidate_id")
        if self.semantic_codes.dim() != 3:
            raise ValueError("semantic_codes must have shape [batch, time, codebook].")
        if self.semantic_mask.shape != self.semantic_codes.shape[:2]:
            raise ValueError("semantic_mask must align with semantic_codes.")
        if self.semantic_mask.dtype != torch.bool:
            raise TypeError("semantic_mask must be boolean.")
        if self.acoustic_features is not None and self.acoustic_features.dim() != 3:
            raise ValueError("acoustic_features must have shape [batch, unit, dim].")
        if self.acoustic_codes is not None and self.acoustic_codes.dim() != 3:
            raise ValueError("acoustic_codes must have shape [batch, unit, codebook].")
        if self.acoustic_mask is not None and self.acoustic_mask.dtype != torch.bool:
            raise TypeError("acoustic_mask must be boolean.")
        if self.waveform is not None and self.waveform.dim() != 3:
            raise ValueError("waveform must have shape [batch, channels, time].")


@dataclass(frozen=True)
class SACRollout:
    candidates: tuple[SACCandidate, ...]
    batch_size: int
    group_size: int

    def __post_init__(self) -> None:
        _positive(self.batch_size, name="batch_size")
        _positive(self.group_size, name="group_size")
        if len(self.candidates) != self.batch_size * self.group_size:
            raise ValueError("candidate count must equal batch_size * group_size.")
        for expected, candidate in enumerate(self.candidates):
            sample_id = expected // self.group_size
            candidate_id = expected % self.group_size
            if candidate.sample_id != sample_id or candidate.group_id != sample_id:
                raise ValueError("candidate sample_id/group_id must follow rollout order.")
            if candidate.candidate_id != candidate_id:
                raise ValueError("candidate_id must follow rollout order within each group.")


@dataclass(frozen=True)
class SACRewardBatch:
    rewards: Tensor
    group_mask: Tensor | None = None
    components: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rewards.dim() != 2:
            raise ValueError("rewards must have shape [batch, group].")
        if self.rewards.dtype == torch.bool or self.rewards.is_complex():
            raise TypeError("rewards must be real-valued.")
        if self.group_mask is not None:
            if self.group_mask.shape != self.rewards.shape:
                raise ValueError("group_mask must align with rewards.")
            if self.group_mask.dtype != torch.bool:
                raise TypeError("group_mask must be boolean.")
        for name, value in self.components.items():
            if value.shape != self.rewards.shape:
                raise ValueError(f"reward component {name!r} must align with rewards.")


def _positive(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _non_negative(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


__all__ = ["SACCandidate", "SACRewardBatch", "SACRollout"]
