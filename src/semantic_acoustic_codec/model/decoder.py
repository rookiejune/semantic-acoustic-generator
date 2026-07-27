from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from anytrain.codec import AcousticLayout
from anytrain.loss import MaskedCodebookCrossEntropyLoss, MaskedCosineAlignmentLoss
from anytrain.module.qwen import QwenMTPCodebookPredictor
from torch import nn

from semantic_acoustic_codec.config import DecoderConfig, Route, RVQPredictor
from semantic_acoustic_codec.loss.flow import FlowLoss
from semantic_acoustic_codec.model.condition import FixedLengthConditioner
from semantic_acoustic_codec.model.dit import DiTDecoder
from semantic_acoustic_codec.model.rvq import AcousticRVQDecoder

if TYPE_CHECKING:
    from typing import Any

    from anytrain.codec import SemanticAcousticCodec
    from anytrain.loss import LossItem
    from torch import Tensor

    from semantic_acoustic_codec.loss.flow import FlowRuntime
    from semantic_acoustic_codec.loss.repa import Teacher
    from semantic_acoustic_codec.types import SemanticCodecBatch


@dataclass(frozen=True)
class DecoderLoss:
    loss: Tensor
    item: LossItem
    log_name: str
    logs: dict[str, Tensor | float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class CodecUnitGenerator(ABC, nn.Module):
    route: Route

    def __init__(self, condition_dim: int, *, fixed_length: int | None = None) -> None:
        super().__init__()
        self.fixed_conditioner = (
            None
            if fixed_length is None
            else FixedLengthConditioner(condition_dim, slots=fixed_length)
        )

    def _target_condition(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        acoustic_layout: AcousticLayout,
        output_length: int | None,
    ) -> tuple[Tensor, Tensor]:
        return _target_condition(
            condition,
            mask,
            acoustic_layout=acoustic_layout,
            output_length=output_length,
            fixed_conditioner=self.fixed_conditioner,
        )

    @abstractmethod
    def sample_features(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        flow_steps: int,
        temperature: float,
        top_p: float,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...

    def sample_acoustic_codes(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        top_p: float,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        raise RuntimeError("sample_acoustic_codes is only available for acoustic-code decoders.")

    @abstractmethod
    def loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_features: Tensor | None = None,
        target_codes: Tensor | None = None,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        repa_features: Tensor | None = None,
        flow_runtime: FlowRuntime | None = None,
        include_top1: bool = False,
    ) -> DecoderLoss: ...

    @abstractmethod
    def loss(
        self,
        batch: SemanticCodecBatch,
        condition: Tensor,
        target_features: Tensor | None = None,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
    ) -> DecoderLoss: ...


class FMFeatureGenerator(CodecUnitGenerator):
    route = Route.FM

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        config: DecoderConfig,
        *,
        fixed_length: int | None = None,
    ) -> None:
        super().__init__(condition_dim, fixed_length=fixed_length)
        self.core = DiTDecoder(
            condition_dim,
            feature_dim,
            hidden_dim=config.hidden_dim,
            layers=config.layers,
            heads=config.heads,
            ffn_ratio=config.ffn_ratio,
            repa_feature_dim=config.repa_feature_dim,
            repa_student_layer=config.repa_student_layer,
        )
        self.repa_loss_weight = config.repa_loss_weight
        self.flow_loss = FlowLoss()
        self.repa_loss = MaskedCosineAlignmentLoss()
        self.flow_runtime: FlowRuntime | None = None

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.core(x_t, t, condition=condition, mask=mask)

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.core.forward_with_features(x_t, t, condition=condition, mask=mask)

    @torch.no_grad()
    def sample_features(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        flow_steps: int,
        temperature: float,
        top_p: float,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del temperature, top_p
        target_condition, target_mask = self._target_condition(
            condition,
            mask,
            acoustic_layout=acoustic_layout,
            output_length=output_length,
        )
        features = self.core.sample(
            target_condition,
            mask=target_mask,
            steps=flow_steps,
            generator=generator,
        )
        return features * feature_std + feature_mean

    def loss(
        self,
        batch: SemanticCodecBatch,
        condition: Tensor,
        target_features: Tensor | None = None,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
    ) -> DecoderLoss:
        target_mask = _acoustic_mask(batch)
        target_condition, _ = self._target_condition(
            condition,
            batch.mask,
            acoustic_layout=batch.acoustic_layout,
            output_length=target_mask.size(1),
        )
        repa_features = None
        if self.repa_loss_weight > 0:
            if repa_teacher is None:
                raise RuntimeError("REPA requires a teacher.")
            if batch.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
                raise ValueError("REPA currently requires frame-aligned acoustic units.")
            repa_features = repa_teacher(
                batch.semantic_codes,
                batch.acoustic_codes,
                batch.mask,
            )
        return self.loss_from_condition(
            target_condition,
            target_mask,
            target_features=target_features,
            feature_mean=feature_mean,
            feature_std=feature_std,
            repa_features=repa_features,
        )

    def loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_features: Tensor | None = None,
        target_codes: Tensor | None = None,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        repa_features: Tensor | None = None,
        flow_runtime: FlowRuntime | None = None,
        include_top1: bool = False,
    ) -> DecoderLoss:
        del target_codes, include_top1
        if target_features is None:
            raise ValueError("FM loss requires acoustic target features.")
        if repa_features is not None and self.repa_loss_weight <= 0:
            raise ValueError("REPA features require a positive repa_loss_weight.")
        target = _normalized_features(
            target_features,
            target_mask,
            feature_mean,
            feature_std,
        )
        runtime = self._flow_runtime() if flow_runtime is None else flow_runtime
        if self.repa_loss_weight <= 0:
            item = self.flow_loss(
                self.core,
                condition,
                target,
                target_mask,
                runtime,
            )
            return DecoderLoss(
                loss=item.loss.mean(),
                item=item,
                log_name="train/flow_loss",
            )
        if repa_features is None:
            raise RuntimeError("REPA requires precomputed teacher features.")
        item, representation = self.flow_loss.forward_with_features(
            self.core,
            condition,
            target,
            target_mask,
            runtime,
        )
        repa = self.repa_loss(representation, repa_features, target_mask)
        item_loss = item.loss.mean()
        repa_loss = repa.loss.mean()
        return DecoderLoss(
            loss=item_loss + self.repa_loss_weight * repa_loss,
            item=item,
            log_name="train/flow_loss",
            logs={
                "train/flow_loss": item_loss,
                "train/repa_loss": repa_loss,
                "train/repa_weight": self.repa_loss_weight,
            },
            extras={"repa": repa},
        )

    def _flow_runtime(self) -> FlowRuntime:
        if self.flow_runtime is None:
            self.flow_runtime = _continuous_flow_runtime()
        return self.flow_runtime


class RVQCodeGenerator(CodecUnitGenerator):
    route = Route.RVQ

    def __init__(
        self,
        condition_dim: int,
        codebook_sizes: tuple[int, ...],
        config: DecoderConfig,
        *,
        fixed_length: int | None = None,
    ) -> None:
        super().__init__(condition_dim, fixed_length=fixed_length)
        if not codebook_sizes:
            raise ValueError("RVQ route requires acoustic codebooks.")
        self.predictor = config.rvq_predictor
        if fixed_length is not None and self.predictor is RVQPredictor.CODEBOOK_AR:
            raise ValueError("fixed-length RVQ requires the MTP predictor along the slot axis.")
        if config.rvq_predictor is RVQPredictor.CODEBOOK_AR:
            self.core: AcousticRVQDecoder | QwenMTPCodebookPredictor = AcousticRVQDecoder(
                condition_dim,
                len(codebook_sizes),
                codebook_sizes,
                hidden_dim=config.hidden_dim,
                layers=config.layers,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
            )
        elif config.rvq_predictor is RVQPredictor.MTP:
            self.core = QwenMTPCodebookPredictor(
                condition_dim,
                len(codebook_sizes),
                codebook_sizes,
                hidden_dim=config.hidden_dim,
                layers=config.layers,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
                mtp_layers=config.mtp_layers,
                mtp_heads=config.mtp_heads,
            )
        else:
            raise AssertionError(f"unsupported RVQ predictor: {config.rvq_predictor}")
        self.rvq_loss = MaskedCodebookCrossEntropyLoss()

    @torch.no_grad()
    def sample_features(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        flow_steps: int,
        temperature: float,
        top_p: float,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del feature_mean, feature_std, flow_steps, temperature, top_p, generator
        raise RuntimeError("RVQ feature conversion requires a codec runtime.")

    @torch.no_grad()
    def sample_acoustic_codes(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        top_p: float,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        target_condition, target_mask = self._target_condition(
            condition,
            mask,
            acoustic_layout=acoustic_layout,
            output_length=output_length,
        )
        return self.core.generate(
            target_condition,
            mask=target_mask,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )

    def loss(
        self,
        batch: SemanticCodecBatch,
        condition: Tensor,
        target_features: Tensor | None = None,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
    ) -> DecoderLoss:
        del target_features, feature_mean, feature_std, repa_teacher
        labels = batch.acoustic_codes
        target_mask = _acoustic_mask(batch)
        target_condition, _ = self._target_condition(
            condition,
            batch.mask,
            acoustic_layout=batch.acoustic_layout,
            output_length=labels.size(1),
        )
        return self.loss_from_condition(
            target_condition,
            target_mask,
            target_codes=labels,
        )

    def loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_features: Tensor | None = None,
        target_codes: Tensor | None = None,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        repa_features: Tensor | None = None,
        flow_runtime: FlowRuntime | None = None,
        include_top1: bool = False,
    ) -> DecoderLoss:
        del target_features, feature_mean, feature_std, repa_features, flow_runtime
        if target_codes is None:
            raise ValueError("RVQ loss requires acoustic target codes.")
        item = self.rvq_loss(
            self.core(condition, target_codes, mask=target_mask),
            target_codes,
            target_mask,
            include_top1=include_top1,
        )
        return DecoderLoss(
            loss=item.loss.mean(),
            item=item,
            log_name="train/rvq_loss",
        )

@torch.no_grad()
def backend_features(
    backend: SemanticAcousticCodec,
    acoustic_codes: Tensor,
    mask: Tensor,
) -> Tensor:
    if acoustic_codes.dim() != 3 or mask.shape != acoustic_codes.shape[:2]:
        raise ValueError(
            "acoustic_codes and mask must have shapes [B, F, K] and [B, F]."
    )
    safe_codes = acoustic_codes.masked_fill(~mask[..., None], 0)
    features = backend.acoustic_codes_to_features(safe_codes)
    feature_mask = mask.to(device=features.device)
    return features.masked_fill(~feature_mask[..., None], 0)


def _acoustic_mask(batch: SemanticCodecBatch) -> Tensor:
    mask = batch.acoustic_mask
    if mask is None:
        raise RuntimeError("SemanticCodecBatch must expose acoustic_mask after validation.")
    return mask


def _target_condition(
    condition: Tensor,
    mask: Tensor,
    *,
    acoustic_layout: AcousticLayout,
    output_length: int | None,
    fixed_conditioner: FixedLengthConditioner | None,
) -> tuple[Tensor, Tensor]:
    if condition.dim() != 3 or mask.shape != condition.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(
            "condition and mask must have shapes [B, semantic_unit, C] and [B, unit]."
        )
    if not bool(mask.any(dim=1).all()):
        raise ValueError("each condition row must contain at least one valid semantic unit.")
    if acoustic_layout is AcousticLayout.FRAME_ALIGNED:
        if output_length is not None and output_length != condition.size(1):
            raise ValueError("frame-aligned output_length must match the semantic unit length.")
        return condition, mask
    if acoustic_layout is not AcousticLayout.FIXED_LENGTH:
        raise ValueError(f"unsupported acoustic layout: {acoustic_layout!r}")
    if output_length is None or output_length < 1:
        raise ValueError("fixed-length acoustic generation requires a positive output_length.")
    if fixed_conditioner is None:
        raise RuntimeError("fixed-length generation requires a configured slot conditioner.")
    target_condition = fixed_conditioner(condition, mask, output_length=output_length)
    target_mask = torch.ones(
        condition.size(0),
        output_length,
        dtype=torch.bool,
        device=condition.device,
    )
    return target_condition, target_mask


def _normalized_features(
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


def _continuous_flow_runtime() -> FlowRuntime:
    from anytrain.framework.flow_matching import ContinuousFlowRuntime

    return ContinuousFlowRuntime()
