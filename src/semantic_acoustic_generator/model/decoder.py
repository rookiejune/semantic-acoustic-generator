from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from anytrain.codec import AcousticLayout
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from anytrain.loss import (
    LossItem,
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    MaskedFrameMSELoss,
)
from anytrain.module.qwen import QwenMTPCodebookPredictor
from torch import nn

from semantic_acoustic_generator.config import DecoderConfig, FMMode, Route, RVQPredictor
from semantic_acoustic_generator.loss.flow import FlowLoss, FlowRuntime
from semantic_acoustic_generator.model.condition import AlignedAnchor, FixedLengthConditioner
from semantic_acoustic_generator.model.dit import DiTDecoder
from semantic_acoustic_generator.model.rvq import AcousticRVQDecoder

if TYPE_CHECKING:
    from torch import Tensor

    from semantic_acoustic_generator.loss.repa import Teacher
    from semantic_acoustic_generator.types import GeneratorBatch


@dataclass(frozen=True)
class DecoderLoss:
    """Generator step loss with named anytrain ``LossItem`` outputs."""

    loss: Tensor
    items: dict[str, LossItem]
    primary: str
    scalars: dict[str, Tensor | float] = field(default_factory=dict)


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
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
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
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


class AcousticUnitGenerator(nn.Module):
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
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        return _target_condition(
            condition,
            mask,
            acoustic_layout=acoustic_layout,
            output_length=output_length,
            fixed_conditioner=self.fixed_conditioner,
            validate=validate,
        )


class FMFeatureGenerator(AcousticUnitGenerator):
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
        self.mode = config.fm_mode
        self.feature_dim = feature_dim
        self.core = (
            None
            if self.mode is FMMode.ANCHOR
            else DiTDecoder(
                condition_dim,
                feature_dim,
                hidden_dim=config.hidden_dim,
                layers=config.layers,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
                repa_feature_dim=config.repa_feature_dim,
                repa_student_layer=config.repa_student_layer,
            )
        )
        self.anchor = (
            None
            if self.mode is FMMode.FLOW
            else AlignedAnchor(
                condition_dim,
                feature_dim,
                hidden_dim=config.anchor_hidden_dim,
                layers=config.anchor_layers,
                kernel_size=config.anchor_kernel_size,
            )
        )
        self.repa_loss_weight = config.repa_loss_weight
        self.anchor_cosine_weight = config.anchor_cosine_weight
        self.anchor_factor_weight = config.anchor_factor_weight
        self.anchor_factor_temperature = config.anchor_factor_temperature
        self.flow_loss = FlowLoss()
        self.repa_loss = MaskedCosineAlignmentLoss()
        self.anchor_mse_loss = MaskedFrameMSELoss()
        self.anchor_cosine_loss = MaskedCosineAlignmentLoss()
        self.anchor_factor_loss = MaskedCodebookCrossEntropyLoss()
        self.flow_runtime: ContinuousFlowRuntime | None = None

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if self.core is None:
            raise RuntimeError("fm_mode=anchor does not expose a flow velocity decoder.")
        return self.core(x_t, t, condition=condition, mask=mask)

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.core is None:
            raise RuntimeError("fm_mode=anchor does not expose flow representations.")
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
        unconditional_condition: Tensor | None = None,
        cfg_scale: float = 1.0,
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
        target_unconditional = None
        if unconditional_condition is not None:
            target_unconditional, unconditional_mask = self._target_condition(
                unconditional_condition,
                mask,
                acoustic_layout=acoustic_layout,
                output_length=output_length,
            )
            if not torch.equal(target_mask, unconditional_mask):
                raise ValueError("conditional and unconditional FM masks must match.")
        anchor = self._anchor(target_condition, target_mask)
        if self.mode is FMMode.ANCHOR:
            normalized = anchor
        else:
            if self.core is None:
                raise RuntimeError("flow sampling requires a DiT decoder.")
            residual = self.core.sample(
                target_condition,
                mask=target_mask,
                steps=flow_steps,
                unconditional_condition=target_unconditional,
                guidance_scale=cfg_scale,
                generator=generator,
            )
            normalized = residual if self.mode is FMMode.FLOW else anchor + residual
        return normalized * feature_std + feature_mean

    def loss(
        self,
        batch: GeneratorBatch,
        condition: Tensor,
        target_features: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
        factor_targets: Tensor | None = None,
        factor_codebooks: tuple[Tensor, Tensor] | None = None,
    ) -> DecoderLoss:
        target_mask = batch.acoustic_mask
        target_condition, _ = self._target_condition(
            condition,
            batch.mask,
            acoustic_layout=batch.acoustic_layout,
            output_length=target_mask.size(1),
            validate=False,
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
        return self.feature_loss_from_condition(
            target_condition,
            target_mask,
            target_features=target_features,
            feature_mean=feature_mean,
            feature_std=feature_std,
            repa_features=repa_features,
            factor_targets=factor_targets,
            factor_codebooks=factor_codebooks,
            validate=False,
        )

    def feature_loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_features: Tensor,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        repa_features: Tensor | None = None,
        flow_runtime: FlowRuntime | None = None,
        factor_targets: Tensor | None = None,
        factor_codebooks: tuple[Tensor, Tensor] | None = None,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss:
        if repa_features is not None and self.repa_loss_weight <= 0:
            raise ValueError("REPA features require a positive repa_loss_weight.")
        target = _normalized_features(
            target_features,
            target_mask,
            feature_mean,
            feature_std,
        )
        if self.mode is not FMMode.FLOW:
            return self._anchor_loss(
                condition,
                target,
                target_features,
                target_mask,
                feature_mean=feature_mean,
                feature_std=feature_std,
                factor_targets=factor_targets,
                factor_codebooks=factor_codebooks,
                flow_runtime=flow_runtime,
                validate=validate,
                include_details=include_details,
            )
        runtime = self._flow_runtime() if flow_runtime is None else flow_runtime
        if self.core is None:
            raise RuntimeError("fm_mode=flow requires a DiT decoder.")
        if self.repa_loss_weight <= 0:
            item = self.flow_loss(
                self.core,
                condition,
                target,
                target_mask,
                runtime,
                validate=validate,
                include_details=include_details,
            )
            return DecoderLoss(loss=item.loss.mean(), items={"flow": item}, primary="flow")
        if repa_features is None:
            raise RuntimeError("REPA requires precomputed teacher features.")
        item, representation = self.flow_loss.forward_with_features(
            self.core,
            condition,
            target,
            target_mask,
            runtime,
            validate=validate,
            include_details=include_details,
        )
        repa = self.repa_loss(representation, repa_features, target_mask)
        flow_loss = item.loss.mean()
        repa_loss = repa.loss.mean()
        return DecoderLoss(
            loss=flow_loss + self.repa_loss_weight * repa_loss,
            items={"flow": item, "repa": repa},
            primary="flow",
            scalars={"repa_weight": self.repa_loss_weight},
        )

    def _anchor_loss(
        self,
        condition: Tensor,
        target: Tensor,
        target_features: Tensor,
        target_mask: Tensor,
        *,
        feature_mean: Tensor | None,
        feature_std: Tensor | None,
        factor_targets: Tensor | None,
        factor_codebooks: tuple[Tensor, Tensor] | None,
        flow_runtime: FlowRuntime | None,
        validate: bool,
        include_details: bool,
    ) -> DecoderLoss:
        anchor = self._anchor(condition, target_mask)
        mean = target_features.new_zeros(1, 1, target_features.size(-1)) if feature_mean is None else feature_mean
        std = target_features.new_ones(1, 1, target_features.size(-1)) if feature_std is None else feature_std
        raw_anchor = anchor * std + mean
        mse = self.anchor_mse_loss(anchor, target, target_mask)
        cosine = self.anchor_cosine_loss(raw_anchor, target_features, target_mask)
        if factor_targets is None or factor_codebooks is None:
            raise ValueError("anchor modes require factor targets and codebooks.")
        logits = self._factor_logits(raw_anchor, factor_codebooks)
        factor = self.anchor_factor_loss(
            logits,
            factor_targets,
            target_mask,
            validate=validate,
            include_top1=include_details,
            include_details=include_details,
        )
        loss = (
            mse.loss.mean()
            + self.anchor_cosine_weight * cosine.loss.mean()
            + self.anchor_factor_weight * factor.loss.mean()
        )
        items = {"anchor_mse": mse, "anchor_cosine": cosine, "anchor_factor": factor}
        scalars: dict[str, Tensor | float] = {
            "anchor_cosine_weight": self.anchor_cosine_weight,
            "anchor_factor_weight": self.anchor_factor_weight,
        }
        primary = "anchor_mse"
        if self.mode is FMMode.RESIDUAL:
            if self.core is None:
                raise RuntimeError("fm_mode=residual requires a DiT decoder.")
            runtime = self._flow_runtime() if flow_runtime is None else flow_runtime
            flow = self.flow_loss(
                self.core,
                condition,
                target - anchor.detach(),
                target_mask,
                runtime,
                validate=validate,
                include_details=include_details,
            )
            items["flow"] = flow
            loss = loss + flow.loss.mean()
            primary = "flow"
        scalars["total_loss"] = loss.detach()
        return DecoderLoss(loss=loss, items=items, primary=primary, scalars=scalars)

    def _anchor(self, condition: Tensor, mask: Tensor) -> Tensor:
        if self.anchor is None:
            return condition.new_zeros(*condition.shape[:2], self.feature_dim)
        return self.anchor(condition, mask)

    def _factor_logits(
        self,
        features: Tensor,
        codebooks: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        dims = (codebooks[0].size(-1), codebooks[1].size(-1))
        if sum(dims) != features.size(-1):
            raise ValueError("factor codebook dimensions must match anchor features.")
        split = features.split(dims, dim=-1)
        return tuple(
            torch.matmul(
                torch.nn.functional.normalize(value.float(), dim=-1),
                torch.nn.functional.normalize(codebook.float(), dim=-1).transpose(0, 1),
            ).to(dtype=features.dtype)
            / self.anchor_factor_temperature
            for value, codebook in zip(split, codebooks, strict=True)
        )  # type: ignore[return-value]

    def _flow_runtime(self) -> ContinuousFlowRuntime:
        if self.flow_runtime is None:
            self.flow_runtime = ContinuousFlowRuntime()
        return self.flow_runtime


class RVQCodeGenerator(AcousticUnitGenerator):
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
        batch: GeneratorBatch,
        condition: Tensor,
    ) -> DecoderLoss:
        labels = batch.acoustic_codes
        target_mask = batch.acoustic_mask
        target_condition, _ = self._target_condition(
            condition,
            batch.mask,
            acoustic_layout=batch.acoustic_layout,
            output_length=labels.size(1),
            validate=False,
        )
        return self.code_loss_from_condition(
            target_condition,
            target_mask,
            target_codes=labels,
            validate=False,
            include_details=False,
        )

    def code_loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_codes: Tensor,
        include_top1: bool = False,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss:
        if isinstance(self.core, AcousticRVQDecoder):
            packed = self.core.forward_packed(
                condition,
                target_codes,
                mask=target_mask,
                validate=validate,
            )
            item = self.rvq_loss.forward_packed(
                packed,
                include_top1=include_top1,
                validate=False,
                include_details=include_details,
            )
        else:
            item = self.rvq_loss(
                self.core(condition, target_codes, mask=target_mask, validate=validate),
                target_codes,
                target_mask,
                include_top1=include_top1,
                validate=validate,
                include_details=include_details,
            )
        return DecoderLoss(loss=item.loss.mean(), items={"rvq": item}, primary="rvq")


def _target_condition(
    condition: Tensor,
    mask: Tensor,
    *,
    acoustic_layout: AcousticLayout,
    output_length: int | None,
    fixed_conditioner: FixedLengthConditioner | None,
    validate: bool = True,
) -> tuple[Tensor, Tensor]:
    if validate and (
        condition.dim() != 3 or mask.shape != condition.shape[:2] or mask.dtype != torch.bool
    ):
        raise ValueError("condition and mask must have shapes [B, semantic_unit, C] and [B, unit].")
    if validate and not bool(mask.any(dim=1).all()):
        raise ValueError("each condition row must contain at least one valid semantic unit.")
    if acoustic_layout is AcousticLayout.FRAME_ALIGNED:
        if validate and output_length is not None and output_length != condition.size(1):
            raise ValueError("frame-aligned output_length must match the semantic unit length.")
        return condition, mask
    if acoustic_layout is not AcousticLayout.FIXED_LENGTH:
        raise ValueError(f"unsupported acoustic layout: {acoustic_layout!r}")
    if output_length is None or output_length < 1:
        raise ValueError("fixed-length acoustic generation requires a positive output_length.")
    if fixed_conditioner is None:
        raise RuntimeError("fixed-length generation requires a configured slot conditioner.")
    target_condition = fixed_conditioner(
        condition,
        mask,
        output_length=output_length,
        validate=validate,
    )
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
