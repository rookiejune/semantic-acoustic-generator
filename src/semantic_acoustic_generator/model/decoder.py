from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from anytrain.loss import (
    LossItem,
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    MaskedFrameMSELoss,
)
from anytrain.module.qwen import QwenMTPCodebookPredictor
from torch import nn

from semantic_acoustic_generator.config import (
    AnchorTarget,
    DecoderConfig,
    FactorPredictor,
    FMMode,
    Route,
    RVQPredictor,
)
from semantic_acoustic_generator.loss.flow import FlowLoss, FlowRuntime
from semantic_acoustic_generator.model.condition import AlignedAnchor
from semantic_acoustic_generator.model.dit import DiTDecoder
from semantic_acoustic_generator.model.rvq import AcousticRVQDecoder, FactorDepthPredictor

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


class AcousticUnitGenerator(nn.Module):
    route: Route


class FMFeatureGenerator(AcousticUnitGenerator):
    route = Route.FM

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        config: DecoderConfig,
        *,
        factor_codebooks: tuple[Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.mode = config.fm_mode
        self.anchor_target = config.anchor_target
        self.factor_predictor = config.factor_predictor
        self.feature_dim = feature_dim
        self.factor_depth: FactorDepthPredictor | None
        self.factor_codebook_a: Tensor | None
        self.factor_codebook_b: Tensor | None
        self._factor_codebook_names: tuple[str, ...]
        if self.anchor_target is AnchorTarget.FACTOR:
            if factor_codebooks is None:
                raise ValueError("anchor_target=factor requires factor codebook pairs.")
            _validate_factor_codebooks(factor_codebooks, feature_dim=feature_dim)
            self.factor_codebook_a = nn.Buffer(factor_codebooks[0].detach().clone())
            self.factor_codebook_b = nn.Buffer(factor_codebooks[1].detach().clone())
            names = factor_codebook_names(len(factor_codebooks) // 2)
            for name, value in zip(
                names[2:],
                factor_codebooks[2:],
                strict=True,
            ):
                buffer = nn.Buffer(value.detach().clone())
                setattr(self, name, buffer)
            self._factor_codebook_names = names
            if self.factor_predictor is FactorPredictor.DEPTH_AR:
                anchor_output_dim = config.anchor_hidden_dim
                self.factor_depth = FactorDepthPredictor(
                    config.anchor_hidden_dim,
                    factor_codebooks,
                    hidden_dim=config.hidden_dim,
                    layers=config.layers,
                    heads=config.heads,
                    ffn_ratio=config.ffn_ratio,
                )
            else:
                anchor_output_dim = sum(value.size(0) for value in factor_codebooks)
                self.factor_depth = None
        else:
            if factor_codebooks is not None:
                raise ValueError("factor codebooks require anchor_target=factor.")
            self.factor_codebook_a = None
            self.factor_codebook_b = None
            self._factor_codebook_names = ()
            self.factor_depth = None
            anchor_output_dim = feature_dim
        self.anchor_output_dim = anchor_output_dim
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
                anchor_output_dim,
                hidden_dim=config.anchor_hidden_dim,
                layers=config.anchor_layers,
                kernel_size=config.anchor_kernel_size,
                context=config.anchor_context,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
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
    def sample_factor_codes(
        self,
        condition: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if self.anchor_target is not AnchorTarget.FACTOR:
            raise RuntimeError("factor-code generation requires anchor_target=factor.")
        target_condition, target_mask = _aligned_condition(condition, mask)
        return self._factor_codes_from_anchor(
            self._anchor(target_condition, target_mask),
            target_mask,
        )

    @torch.no_grad()
    def factor_logits(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        factor_targets: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Return factor logits, teacher-forcing previous stages for depth-AR."""
        if self.anchor_target is not AnchorTarget.FACTOR:
            raise RuntimeError("factor logits require anchor_target=factor.")
        target_condition, target_mask = _aligned_condition(condition, mask)
        anchor = self._anchor(target_condition, target_mask)
        if self.factor_depth is None:
            return self._factor_output(anchor)
        if factor_targets is None:
            raise ValueError("depth-AR factor logits require factor targets for teacher forcing.")
        return self.factor_depth(
            anchor,
            factor_targets,
            mask=target_mask,
        )

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
        generator: torch.Generator | None = None,
    ) -> Tensor:
        target_condition, target_mask = _aligned_condition(condition, mask)
        target_unconditional = None
        if unconditional_condition is not None:
            target_unconditional, unconditional_mask = _aligned_condition(
                unconditional_condition,
                mask,
            )
            if not torch.equal(target_mask, unconditional_mask):
                raise ValueError("conditional and unconditional FM masks must match.")
        anchor = self._anchor(target_condition, target_mask)
        if self.anchor_target is AnchorTarget.FACTOR:
            return self._factor_features(anchor, target_mask)
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
        target_features: Tensor | None,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
        factor_targets: Tensor | None = None,
        factor_codebooks: tuple[Tensor, ...] | None = None,
    ) -> DecoderLoss:
        target_mask = batch.acoustic_mask
        target_condition, _ = _aligned_condition(
            condition,
            batch.mask,
            target_mask=target_mask,
            validate=False,
        )
        repa_features = None
        if self.repa_loss_weight > 0:
            if repa_teacher is None:
                raise RuntimeError("REPA requires a teacher.")
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
        target_features: Tensor | None,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        repa_features: Tensor | None = None,
        flow_runtime: FlowRuntime | None = None,
        factor_targets: Tensor | None = None,
        factor_codebooks: tuple[Tensor, ...] | None = None,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss:
        if repa_features is not None and self.repa_loss_weight <= 0:
            raise ValueError("REPA features require a positive repa_loss_weight.")
        if self.anchor_target is AnchorTarget.FACTOR:
            return self._factor_loss(
                condition,
                target_mask,
                factor_targets=factor_targets,
                validate=validate,
                include_details=include_details,
            )
        if target_features is None:
            raise ValueError("anchor_target=feature requires target_features.")
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
        factor_codebooks: tuple[Tensor, ...] | None,
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

    def _factor_loss(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        factor_targets: Tensor | None,
        validate: bool,
        include_details: bool,
    ) -> DecoderLoss:
        if factor_targets is None:
            raise ValueError("anchor_target=factor requires factor targets.")
        anchor = self._anchor(condition, target_mask)
        logits = (
            self._factor_output(anchor)
            if self.factor_depth is None
            else self.factor_depth(
                anchor,
                factor_targets,
                mask=target_mask,
                validate=validate,
            )
        )
        factor = self.anchor_factor_loss(
            logits,
            factor_targets,
            target_mask,
            validate=validate,
            include_top1=include_details,
            include_details=include_details,
        )
        loss = factor.loss.mean()
        return DecoderLoss(
            loss=loss,
            items={"anchor_factor": factor},
            primary="anchor_factor",
            scalars={"total_loss": loss.detach()},
        )

    def _anchor(self, condition: Tensor, mask: Tensor) -> Tensor:
        if self.anchor is None:
            return condition.new_zeros(*condition.shape[:2], self.anchor_output_dim)
        return self.anchor(condition, mask)

    def _factor_output(self, logits: Tensor) -> tuple[Tensor, ...]:
        codebooks = self._stored_factor_codebooks()
        sizes = tuple(value.size(0) for value in codebooks)
        if logits.size(-1) != sum(sizes):
            raise ValueError("factor logits do not match stored codebook sizes.")
        return logits.split(sizes, dim=-1)  # type: ignore[return-value]

    def _factor_features(self, anchor: Tensor, mask: Tensor) -> Tensor:
        codebooks = self._stored_factor_codebooks()
        codes = self._factor_codes_from_anchor(anchor, mask)
        features = torch.cat(
            tuple(
                torch.nn.functional.embedding(value, codebook)
                for value, codebook in zip(codes.unbind(dim=-1), codebooks, strict=True)
            ),
            dim=-1,
        )
        return features.masked_fill(~mask[..., None], 0)

    def _factor_codes_from_anchor(self, anchor: Tensor, mask: Tensor) -> Tensor:
        if self.factor_depth is not None:
            return self.factor_depth.generate(anchor, mask=mask)
        return self._factor_codes(anchor, mask)

    def _factor_codes(self, logits: Tensor, mask: Tensor) -> Tensor:
        codes = torch.stack(
            tuple(value.argmax(dim=-1) for value in self._factor_output(logits)),
            dim=-1,
        )
        return codes.masked_fill(~mask[..., None], 0)

    def _stored_factor_codebooks(self) -> tuple[Tensor, ...]:
        if not self._factor_codebook_names:
            raise RuntimeError("factor target requires stored factor codebooks.")
        result: list[Tensor] = []
        for name in self._factor_codebook_names:
            value = getattr(self, name, None)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"stored factor codebook {name!r} is missing.")
            result.append(value)
        return tuple(result)

    def _factor_logits(
        self,
        features: Tensor,
        codebooks: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        dims = tuple(value.size(-1) for value in codebooks)
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


def _validate_factor_codebooks(
    codebooks: tuple[Tensor, ...],
    *,
    feature_dim: int,
) -> None:
    if len(codebooks) < 2 or len(codebooks) % 2 != 0:
        raise ValueError("factor target requires one codebook pair per acoustic stage.")
    if any(value.dim() != 2 or value.size(0) < 1 or value.size(1) < 1 for value in codebooks):
        raise ValueError("factor codebooks must contain non-empty rank-2 tensors.")
    if any(not value.is_floating_point() or value.is_complex() for value in codebooks):
        raise TypeError("factor codebooks must use a real floating point dtype.")
    if any(
        value.device != codebooks[0].device or value.dtype != codebooks[0].dtype
        for value in codebooks[1:]
    ):
        raise ValueError("factor codebooks must share a device and dtype.")
    if sum(value.size(1) for value in codebooks) != feature_dim:
        raise ValueError("factor codebook dimensions must sum to feature_dim.")


def factor_codebook_names(codebooks: int) -> tuple[str, ...]:
    if isinstance(codebooks, bool) or not isinstance(codebooks, int):
        raise TypeError("factor codebook count must be an integer.")
    if codebooks <= 0:
        raise ValueError("factor codebook count must be positive.")
    names: list[str] = []
    for stage in range(codebooks):
        prefix = "factor_codebook" if stage == 0 else f"factor_codebook_{stage}"
        names.extend((f"{prefix}_a", f"{prefix}_b"))
    return tuple(names)


class RVQCodeGenerator(AcousticUnitGenerator):
    route = Route.RVQ

    def __init__(
        self,
        condition_dim: int,
        codebook_sizes: tuple[int, ...],
        config: DecoderConfig,
    ) -> None:
        super().__init__()
        if not codebook_sizes:
            raise ValueError("RVQ route requires acoustic codebooks.")
        self.predictor = config.rvq_predictor
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
        generator: torch.Generator | None = None,
    ) -> Tensor:
        target_condition, target_mask = _aligned_condition(condition, mask)
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
        target_condition, _ = _aligned_condition(
            condition,
            batch.mask,
            target_mask=target_mask,
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


def _aligned_condition(
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
