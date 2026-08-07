from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from anytrain import observation
from anytrain.loss import MaskedFrameMSELoss
from torch import Tensor, nn


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
    """Frame-masked velocity objective with automatic timestep observations."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_loss = MaskedFrameMSELoss()
        observation.registry.recommend(self)

    def forward(
        self,
        decoder: FeatureDecoder,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
        *,
        validate: bool = True,
    ) -> Tensor:
        if validate:
            self._validate_inputs(condition, target, mask)
        sample = runtime.training_sample(target)
        prediction = decoder(
            sample.x_t,
            sample.t,
            condition=condition,
            mask=mask,
            **({} if validate else {"validate": False}),
        )
        return self._loss(prediction, sample, mask)

    def forward_with_features(
        self,
        decoder: FeatureDecoder,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
        *,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if validate:
            self._validate_inputs(condition, target, mask)
        sample = runtime.training_sample(target)
        prediction, representation = decoder.forward_with_features(
            sample.x_t,
            sample.t,
            condition=condition,
            mask=mask,
            **({} if validate else {"validate": False}),
        )
        return self._loss(prediction, sample, mask), representation

    def _loss(
        self,
        prediction: Tensor,
        sample: TrainingSample,
        mask: Tensor,
    ) -> Tensor:
        if prediction.shape != sample.velocity.shape:
            raise ValueError("flow decoder output must match target latent shape.")
        loss = self.frame_loss(prediction, sample.velocity, mask)
        observation.emit(
            self,
            "diagnostics",
            {
                "valid_frames": observation.Curve(
                    mask.sum().to(device=loss.device, dtype=loss.dtype)
                ),
            },
        )
        observation.emit(
            self,
            "time",
            _time_profile(prediction, sample.velocity, sample.t, mask),
        )
        return loss

    def _validate_inputs(self, condition: Tensor, target: Tensor, mask: Tensor) -> None:
        if condition.dim() != 3 or target.dim() != 3 or mask.dim() != 2:
            raise ValueError(
                "condition, target, and mask must have shapes [B, F, H], [B, F, D], and [B, F]."
            )
        if condition.shape[:2] != target.shape[:2] or mask.shape != target.shape[:2]:
            raise ValueError("flow condition, target, and mask must align on [batch, frame].")
        if mask.dtype != torch.bool:
            raise TypeError("flow mask must be boolean.")


def _time_profile(
    prediction: Tensor,
    target: Tensor,
    time: Tensor,
    mask: Tensor,
) -> observation.Profile:
    frame_loss = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
    if time.numel() == mask.size(0):
        counts = mask.sum(dim=1).clamp_min(1)
        values = frame_loss.masked_fill(~mask, 0).sum(dim=1) / counts
        coordinates = time.reshape(mask.size(0), -1)[:, 0]
    elif time.shape == mask.shape:
        values = frame_loss.masked_select(mask)
        coordinates = time.masked_select(mask)
    else:
        raise ValueError("flow timestep must provide one value per row or valid frame.")
    return observation.Profile(
        values.detach().float(),
        coordinates.detach().float(),
        _TIME_EDGES,
    )


def _collect_loss(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> observation.Curve:
    del module, inputs, kwargs
    if not isinstance(output, Tensor) or output.ndim != 0:
        raise TypeError("flow loss observation requires a scalar Tensor output.")
    return observation.Curve(output)


_TIME_EDGES = tuple(index / 10 for index in range(11))

observation.registry.register(
    FlowLoss,
    (
        observation.OutputObservation(
            "loss",
            _collect_loss,
            reduction=observation.Reduction.Mean,
            recommended=True,
            covers_descendants=True,
        ),
        observation.ForwardEvent(
            "diagnostics",
            reduction=observation.Reduction.Mean,
            recommended=True,
        ),
        observation.ForwardEvent(
            "time",
            reduction=observation.Reduction.Merge,
            history=observation.EMAHistory(),
            recommended=True,
        ),
    ),
)


__all__ = ["FlowLoss", "FlowRuntime"]
