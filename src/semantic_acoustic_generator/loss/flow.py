from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from anytrain.loss import LossItem, MaskedFrameMSELoss
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class TrainingSample(Protocol):
    x_t: Tensor
    t: Tensor
    velocity: Tensor


class FlowRuntime(Protocol):
    def training_sample(self, x_1: Tensor, *, x_0: Tensor | None = None) -> TrainingSample: ...


class FeatureDecoder(Protocol):
    def __call__(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor,
        validate: bool = True,
    ) -> Tensor: ...

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]: ...


class FlowLoss(nn.Module):
    """Frame-masked velocity objective for acoustic feature decoders."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_loss = MaskedFrameMSELoss()

    def forward(
        self,
        decoder: FeatureDecoder,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
        *,
        validate: bool = True,
        include_details: bool = True,
    ) -> LossItem:
        if validate:
            self._validate_inputs(condition, target, mask)
        sample = runtime.training_sample(target)
        if validate:
            prediction = decoder(sample.x_t, sample.t, condition=condition, mask=mask)
        else:
            prediction = decoder(
                sample.x_t,
                sample.t,
                condition=condition,
                mask=mask,
                validate=False,
            )
        return self._loss(
            prediction,
            sample,
            target,
            mask,
            include_details=include_details,
        )

    def forward_with_features(
        self,
        decoder: FeatureDecoder,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
        *,
        validate: bool = True,
        include_details: bool = True,
    ) -> tuple[LossItem, Tensor]:
        if validate:
            self._validate_inputs(condition, target, mask)
        sample = runtime.training_sample(target)
        if validate:
            prediction, representation = decoder.forward_with_features(
                sample.x_t,
                sample.t,
                condition=condition,
                mask=mask,
            )
        else:
            prediction, representation = decoder.forward_with_features(
                sample.x_t,
                sample.t,
                condition=condition,
                mask=mask,
                validate=False,
            )
        return (
            self._loss(
                prediction,
                sample,
                target,
                mask,
                include_details=include_details,
            ),
            representation,
        )

    def _loss(
        self,
        prediction: Tensor,
        sample: TrainingSample,
        target: Tensor,
        mask: Tensor,
        *,
        include_details: bool,
    ) -> LossItem:
        if prediction.shape != sample.velocity.shape:
            raise ValueError("flow decoder output must match target latent shape.")
        item = self.frame_loss(
            prediction,
            sample.velocity,
            mask,
            details={"t": sample.t},
            detail_dtype=target.dtype,
        )
        if include_details:
            return item
        return LossItem(loss=item.loss, details=None)

    def _validate_inputs(self, condition: Tensor, target: Tensor, mask: Tensor) -> None:
        if condition.dim() != 3 or target.dim() != 3 or mask.dim() != 2:
            raise ValueError(
                "condition, target, and mask must have shapes [B, F, H], [B, F, D], and [B, F]."
            )
        if condition.shape[:2] != target.shape[:2] or mask.shape != target.shape[:2]:
            raise ValueError("flow condition, target, and mask must align on [batch, frame].")
        if mask.dtype != torch.bool:
            raise TypeError("flow mask must be boolean.")


__all__ = ["FlowLoss", "FlowRuntime"]
