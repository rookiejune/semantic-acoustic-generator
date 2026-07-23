from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from semantic_acoustic_codec.loss.types import LossItem


class Teacher(Protocol):
    @property
    def feature_dim(self) -> int: ...

    def __call__(
        self,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> Tensor: ...


class RepaLoss(nn.Module):
    """Align a selected DiT layer to detached teacher frame features."""

    def forward(
        self,
        representation: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> LossItem:
        if representation.shape != target.shape:
            raise ValueError("REPA representation and teacher shapes must match")
        if mask.shape != target.shape[:2]:
            raise ValueError("REPA mask must align with teacher frames")
        if mask.dtype != torch.bool:
            raise TypeError("REPA mask must be boolean")
        frame_mask = mask[..., None]
        prediction = F.normalize(
            representation.masked_fill(~frame_mask, 0).float(),
            dim=-1,
        )
        teacher = F.normalize(
            target.detach()
            .to(device=representation.device)
            .masked_fill(~frame_mask, 0)
            .float(),
            dim=-1,
        )
        frame_loss = (1 - (prediction * teacher).sum(dim=-1)).masked_fill(~mask, 0)
        weights = mask.to(dtype=frame_loss.dtype)
        frame_count = weights.sum(dim=1)
        loss = frame_loss.sum(dim=1) / frame_count.clamp_min(1)
        return LossItem(loss=loss, details={"cosine": 1 - loss})
