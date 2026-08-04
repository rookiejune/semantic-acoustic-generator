from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from anytrain.module.dit import DiT, DiTConditionType
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class DiTDecoder(nn.Module):
    """DiT acoustic feature decoder trained with FM and optional REPA objectives."""

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        *,
        hidden_dim: int | None = None,
        layers: int = 8,
        heads: int = 8,
        ffn_ratio: int = 4,
        repa_feature_dim: int | None = None,
        repa_student_layer: int | None = None,
    ) -> None:
        super().__init__()
        self.decoder = DiT(
            input_dim=feature_dim,
            output_dim=feature_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            ffn_ratio=ffn_ratio,
            condition_dim=condition_dim,
            condition_type=DiTConditionType.FRAME_FILM,
            feature_dim=repa_feature_dim,
            feature_layer=repa_student_layer,
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        return self.decoder(x_t, t, condition=condition, mask=mask, validate=validate)

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        return self.decoder.forward_with_features(
            x_t,
            t,
            condition=condition,
            mask=mask,
            validate=validate,
        )

    @torch.no_grad()
    def sample(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        steps: int = 16,
        unconditional_condition: Tensor | None = None,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if steps < 1:
            raise ValueError("flow sample steps must be positive.")
        if condition.dim() != 3:
            raise ValueError("condition must have shape [B, F, C].")
        if condition.size(0) < 1 or condition.size(1) < 1:
            raise ValueError("condition must contain at least one frame per row.")
        if mask is not None and (mask.shape != condition.shape[:2] or mask.dtype != torch.bool):
            raise ValueError("flow sample mask must be boolean with shape [B, F].")
        if mask is not None and mask.device != condition.device:
            raise ValueError("flow sample mask and condition must use the same device.")
        if mask is not None and not bool(mask.any(dim=1).all()):
            raise ValueError("each flow sample mask row must contain at least one valid frame.")
        if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, (int, float)):
            raise TypeError("guidance_scale must be a number.")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative.")
        if unconditional_condition is not None:
            if unconditional_condition.shape != condition.shape:
                raise ValueError("unconditional_condition must match condition shape.")
            if unconditional_condition.device != condition.device:
                raise ValueError("unconditional_condition must use the same device as condition.")
            if unconditional_condition.dtype != condition.dtype:
                raise TypeError("unconditional_condition must use the same dtype as condition.")
        latent = torch.randn(
            (*condition.shape[:2], self.decoder.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        condition_state = self.decoder.prepare_condition(condition)
        use_guidance = unconditional_condition is not None and guidance_scale != 1.0
        unconditional_state = (
            self.decoder.prepare_condition(unconditional_condition) if use_guidance else None
        )
        dt = 1.0 / steps
        for index in range(steps):
            t = condition.new_full((condition.size(0),), (index + 0.5) * dt)
            velocity = self.decoder(
                latent,
                t,
                condition_state=condition_state,
                mask=mask,
                validate=False,
            )
            if unconditional_state is not None:
                unconditional = self.decoder(
                    latent,
                    t,
                    condition_state=unconditional_state,
                    mask=mask,
                    validate=False,
                )
                velocity = unconditional + guidance_scale * (velocity - unconditional)
            latent = latent + dt * velocity
            if mask is not None:
                latent = latent.masked_fill(~mask[..., None], 0)
        return latent


__all__ = ["DiTDecoder"]
