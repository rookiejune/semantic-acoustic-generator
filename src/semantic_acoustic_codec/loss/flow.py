from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from semantic_acoustic_codec.loss.types import LossItem


class TrainingSample(Protocol):
    x_t: Tensor
    t: Tensor
    velocity: Tensor


class FlowRuntime(Protocol):
    def training_sample(self, x_1: Tensor, *, x_0: Tensor | None = None) -> TrainingSample: ...


@dataclass(eq=False)
class FlowSample:
    x_t: Tensor
    t: Tensor
    velocity: Tensor


class RectifiedFlowRuntime:
    """Small local flow runtime for the package's per-sample loss contract."""

    def training_sample(self, x_1: Tensor, *, x_0: Tensor | None = None) -> FlowSample:
        if x_1.dim() != 3:
            raise ValueError("flow target must have shape [B, F, D].")
        noise = torch.randn_like(x_1) if x_0 is None else x_0
        if noise.shape != x_1.shape:
            raise ValueError("flow noise and target must have the same shape.")
        t = torch.rand(x_1.size(0), device=x_1.device, dtype=x_1.dtype)
        view_t = t[:, None, None]
        x_t = (1 - view_t) * noise + view_t * x_1
        return FlowSample(x_t=x_t, t=t, velocity=x_1 - noise)


class FeatureDecoder(Protocol):
    def __call__(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor,
    ) -> Tensor: ...

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


class FlowLoss(nn.Module):
    """Frame-masked velocity objective for acoustic feature decoders."""

    def forward(
        self,
        decoder: FeatureDecoder,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
    ) -> LossItem:
        self._validate_inputs(condition, target, mask)
        sample = runtime.training_sample(target)
        prediction = decoder(sample.x_t, sample.t, condition=condition, mask=mask)
        return self._loss(prediction, sample, target, mask)

    def forward_with_features(
        self,
        decoder: FeatureDecoder,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
    ) -> tuple[LossItem, Tensor]:
        self._validate_inputs(condition, target, mask)
        sample = runtime.training_sample(target)
        prediction, representation = decoder.forward_with_features(
            sample.x_t,
            sample.t,
            condition=condition,
            mask=mask,
        )
        return self._loss(prediction, sample, target, mask), representation

    def _loss(
        self,
        prediction: Tensor,
        sample: TrainingSample,
        target: Tensor,
        mask: Tensor,
    ) -> LossItem:
        if prediction.shape != sample.velocity.shape:
            raise ValueError("flow decoder output must match target latent shape.")
        frame_mask = mask[..., None]
        frame_loss = F.mse_loss(
            prediction.masked_fill(~frame_mask, 0),
            sample.velocity.masked_fill(~frame_mask, 0),
            reduction="none",
        ).mean(dim=-1)
        weights = mask.to(dtype=frame_loss.dtype)
        frames = weights.sum(dim=1)
        return LossItem(
            loss=frame_loss.sum(dim=1) / frames.clamp_min(1),
            details={
                "frames": frames.to(dtype=target.dtype),
                "t": sample.t.to(dtype=target.dtype),
            },
        )

    def _validate_inputs(self, condition: Tensor, target: Tensor, mask: Tensor) -> None:
        if condition.dim() != 3 or target.dim() != 3 or mask.dim() != 2:
            raise ValueError(
                "condition, target, and mask must have shapes [B, F, H], [B, F, D], and [B, F]."
            )
        if condition.shape[:2] != target.shape[:2] or mask.shape != target.shape[:2]:
            raise ValueError("flow condition, target, and mask must align on [batch, frame].")
        if mask.dtype != torch.bool:
            raise TypeError("flow mask must be boolean.")
