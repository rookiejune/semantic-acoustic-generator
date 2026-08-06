"""Continuous acoustic-feature generator implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from anytrain.loss import (
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    MaskedFrameMSELoss,
)
from torch import nn

from semantic_acoustic_generator.config import (
    AnchorTarget,
    DecoderConfig,
    FactorPredictor,
    FMMode,
    Route,
)
from semantic_acoustic_generator.loss.flow import FlowLoss, FlowRuntime
from semantic_acoustic_generator.model.condition import AlignedAnchor
from semantic_acoustic_generator.model.dit import DiTDecoder
from semantic_acoustic_generator.model.generator import (
    AcousticUnitGenerator,
    DecoderLoss,
    aligned_condition,
    normalized_features,
)
from semantic_acoustic_generator.model.rvq import FactorDepthPredictor

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from semantic_acoustic_generator.loss.repa import Teacher
    from semantic_acoustic_generator.types import GeneratorBatch


class FMFeatureGenerator(AcousticUnitGenerator):
    """Common FM generator contract with construction dispatched by behavior."""

    route = Route.FM
    mode: FMMode
    anchor_target: AnchorTarget
    factor_predictor: FactorPredictor
    feature_dim: int
    core: DiTDecoder | None
    anchor: AlignedAnchor | None
    factor_depth: FactorDepthPredictor | None
    factor_codebook_a: Tensor | None
    factor_codebook_b: Tensor | None
    _factor_codebook_names: tuple[str, ...]
    repa_loss_weight: float
    flow_runtime: ContinuousFlowRuntime | None

    def __new__(
        cls,
        condition_dim: int,
        feature_dim: int,
        config: DecoderConfig,
        *,
        factor_codebooks: tuple[Tensor, ...] | None = None,
    ) -> FMFeatureGenerator:
        del condition_dim, feature_dim, factor_codebooks
        if cls is FMFeatureGenerator:
            if config.anchor_target is AnchorTarget.FACTOR:
                implementation = _FactorAnchorFeatureGenerator
            elif config.fm_mode is FMMode.FLOW:
                implementation = _FlowFeatureGenerator
            else:
                implementation = _AlignedFeatureGenerator
            return cast(FMFeatureGenerator, object.__new__(implementation))
        return super().__new__(cls)

    def _init_common(self, feature_dim: int, config: DecoderConfig) -> None:
        super().__init__()
        self.mode = config.fm_mode
        self.anchor_target = config.anchor_target
        self.factor_predictor = config.factor_predictor
        self.feature_dim = feature_dim
        self.repa_loss_weight = config.repa_loss_weight
        self.flow_runtime = None

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
        del condition, mask
        raise RuntimeError("factor-code generation requires anchor_target=factor.")

    @torch.no_grad()
    def factor_logits(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        factor_targets: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        del condition, mask, factor_targets
        raise RuntimeError("factor logits require anchor_target=factor.")

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
        del (
            condition,
            mask,
            feature_mean,
            feature_std,
            flow_steps,
            unconditional_condition,
            cfg_scale,
            generator,
        )
        raise NotImplementedError

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
        factor_targeter: Callable[[int, Tensor], Tensor] | None = None,
        include_details: bool = True,
    ) -> DecoderLoss:
        target_mask = batch.acoustic_mask
        target_condition, _ = aligned_condition(
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
            factor_targeter=factor_targeter,
            validate=False,
            include_details=include_details,
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
        factor_targeter: Callable[[int, Tensor], Tensor] | None = None,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss:
        del (
            condition,
            target_mask,
            target_features,
            feature_mean,
            feature_std,
            repa_features,
            flow_runtime,
            factor_targets,
            factor_codebooks,
            factor_targeter,
            validate,
            include_details,
        )
        raise NotImplementedError

    def _flow_runtime(self) -> ContinuousFlowRuntime:
        if self.flow_runtime is None:
            self.flow_runtime = ContinuousFlowRuntime()
        return self.flow_runtime


class _FlowFeatureGenerator(FMFeatureGenerator):
    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        config: DecoderConfig,
        *,
        factor_codebooks: tuple[Tensor, ...] | None = None,
    ) -> None:
        if factor_codebooks is not None:
            raise ValueError("factor codebooks require anchor_target=factor.")
        self._init_common(feature_dim, config)
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
        self.anchor = None
        self.factor_depth = None
        self.factor_codebook_a = None
        self.factor_codebook_b = None
        self._factor_codebook_names = ()
        self.flow_loss = FlowLoss()
        self.repa_loss = MaskedCosineAlignmentLoss()

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
        target_condition, target_mask, target_unconditional = _fm_sample_inputs(
            condition,
            mask,
            unconditional_condition,
        )
        core = self.core
        if core is None:
            raise RuntimeError("flow generator is missing its DiT decoder.")
        normalized = core.sample(
            target_condition,
            mask=target_mask,
            steps=flow_steps,
            unconditional_condition=target_unconditional,
            guidance_scale=cfg_scale,
            generator=generator,
        )
        return normalized * feature_std + feature_mean

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
        factor_targeter: Callable[[int, Tensor], Tensor] | None = None,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss:
        del factor_targets, factor_codebooks, factor_targeter
        if repa_features is not None and self.repa_loss_weight <= 0:
            raise ValueError("REPA features require a positive repa_loss_weight.")
        if target_features is None:
            raise ValueError("anchor_target=feature requires target_features.")
        target = normalized_features(target_features, target_mask, feature_mean, feature_std)
        runtime = self._flow_runtime() if flow_runtime is None else flow_runtime
        core = self.core
        if core is None:
            raise RuntimeError("flow generator is missing its DiT decoder.")
        if self.repa_loss_weight <= 0:
            item = self.flow_loss(
                core,
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
            core,
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


class _AlignedFeatureGenerator(FMFeatureGenerator):
    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        config: DecoderConfig,
        *,
        factor_codebooks: tuple[Tensor, ...] | None = None,
    ) -> None:
        if factor_codebooks is not None:
            raise ValueError("factor codebooks require anchor_target=factor.")
        self._init_common(feature_dim, config)
        self.core = (
            None
            if config.fm_mode is FMMode.ANCHOR
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
        self.anchor = AlignedAnchor(
            condition_dim,
            feature_dim,
            hidden_dim=config.anchor_hidden_dim,
            layers=config.anchor_layers,
            kernel_size=config.anchor_kernel_size,
            context=config.anchor_context,
            heads=config.heads,
            ffn_ratio=config.ffn_ratio,
        )
        self.factor_depth = None
        self.factor_codebook_a = None
        self.factor_codebook_b = None
        self._factor_codebook_names = ()
        self.anchor_cosine_weight = config.anchor_cosine_weight
        self.anchor_factor_weight = config.anchor_factor_weight
        self.anchor_factor_temperature = config.anchor_factor_temperature
        self.flow_loss = FlowLoss()
        self.anchor_mse_loss = MaskedFrameMSELoss()
        self.anchor_cosine_loss = MaskedCosineAlignmentLoss()
        self.anchor_factor_loss = MaskedCodebookCrossEntropyLoss()

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
        target_condition, target_mask, target_unconditional = _fm_sample_inputs(
            condition,
            mask,
            unconditional_condition,
        )
        anchor_module = self.anchor
        if anchor_module is None:
            raise RuntimeError("aligned generator is missing its anchor.")
        anchor = anchor_module(target_condition, target_mask)
        normalized = anchor
        if self.core is not None:
            residual = self.core.sample(
                target_condition,
                mask=target_mask,
                steps=flow_steps,
                unconditional_condition=target_unconditional,
                guidance_scale=cfg_scale,
                generator=generator,
            )
            normalized = anchor + residual
        return normalized * feature_std + feature_mean

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
        factor_targeter: Callable[[int, Tensor], Tensor] | None = None,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss:
        del factor_targeter
        if repa_features is not None:
            raise ValueError("REPA features require fm_mode=flow.")
        if target_features is None:
            raise ValueError("anchor_target=feature requires target_features.")
        target = normalized_features(target_features, target_mask, feature_mean, feature_std)
        anchor_module = self.anchor
        if anchor_module is None:
            raise RuntimeError("aligned generator is missing its anchor.")
        anchor = anchor_module(condition, target_mask)
        mean = (
            target_features.new_zeros(1, 1, target_features.size(-1))
            if feature_mean is None
            else feature_mean
        )
        std = (
            target_features.new_ones(1, 1, target_features.size(-1))
            if feature_std is None
            else feature_std
        )
        raw_anchor = anchor * std + mean
        mse = self.anchor_mse_loss(anchor, target, target_mask)
        cosine = self.anchor_cosine_loss(raw_anchor, target_features, target_mask)
        if factor_targets is None or factor_codebooks is None:
            raise ValueError("anchor modes require factor targets and codebooks.")
        logits = _factor_logits(
            raw_anchor,
            factor_codebooks,
            temperature=self.anchor_factor_temperature,
        )
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
        if self.core is not None:
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


class _FactorAnchorFeatureGenerator(FMFeatureGenerator):
    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        config: DecoderConfig,
        *,
        factor_codebooks: tuple[Tensor, ...] | None = None,
    ) -> None:
        if factor_codebooks is None:
            raise ValueError("anchor_target=factor requires factor codebook pairs.")
        _validate_factor_codebooks(factor_codebooks, feature_dim=feature_dim)
        self._init_common(feature_dim, config)
        self.factor_codebook_a = nn.Buffer(factor_codebooks[0].detach().clone())
        self.factor_codebook_b = nn.Buffer(factor_codebooks[1].detach().clone())
        names = factor_codebook_names(len(factor_codebooks) // 2)
        for name, value in zip(names[2:], factor_codebooks[2:], strict=True):
            setattr(self, name, nn.Buffer(value.detach().clone()))
        self._factor_codebook_names = names
        if config.factor_predictor is FactorPredictor.PARALLEL:
            anchor_output_dim = sum(value.size(0) for value in factor_codebooks)
            self.factor_depth = None
        else:
            anchor_output_dim = config.anchor_hidden_dim
            self.factor_depth = FactorDepthPredictor(
                config.anchor_hidden_dim,
                factor_codebooks,
                hidden_dim=config.hidden_dim,
                layers=config.layers,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
                recurrent=config.factor_predictor is FactorPredictor.DEPTH_RECURRENT,
            )
        self.core = None
        self.anchor = AlignedAnchor(
            condition_dim,
            anchor_output_dim,
            hidden_dim=config.anchor_hidden_dim,
            layers=config.anchor_layers,
            kernel_size=config.anchor_kernel_size,
            context=config.anchor_context,
            heads=config.heads,
            ffn_ratio=config.ffn_ratio,
        )
        self.anchor_factor_loss = MaskedCodebookCrossEntropyLoss()

    @torch.no_grad()
    def sample_factor_codes(self, condition: Tensor, mask: Tensor) -> Tensor:
        target_condition, target_mask = aligned_condition(condition, mask)
        anchor_module = self.anchor
        if anchor_module is None:
            raise RuntimeError("factor generator is missing its anchor.")
        anchor = anchor_module(target_condition, target_mask)
        return self._factor_codes_from_anchor(anchor, target_mask)

    @torch.no_grad()
    def factor_logits(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        factor_targets: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Return factor logits, teacher-forcing previous stages for depth-AR."""
        target_condition, target_mask = aligned_condition(condition, mask)
        anchor_module = self.anchor
        if anchor_module is None:
            raise RuntimeError("factor generator is missing its anchor.")
        anchor = anchor_module(target_condition, target_mask)
        if self.factor_depth is None:
            return self._factor_output(anchor)
        if factor_targets is None:
            raise ValueError("depth-AR factor logits require factor targets for teacher forcing.")
        return self.factor_depth(anchor, factor_targets, mask=target_mask)

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
        del feature_mean, feature_std, flow_steps, cfg_scale, generator
        target_condition, target_mask, _ = _fm_sample_inputs(
            condition,
            mask,
            unconditional_condition,
        )
        anchor_module = self.anchor
        if anchor_module is None:
            raise RuntimeError("factor generator is missing its anchor.")
        anchor = anchor_module(target_condition, target_mask)
        return self._factor_features(anchor, target_mask)

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
        factor_targeter: Callable[[int, Tensor], Tensor] | None = None,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss:
        del target_features, feature_mean, feature_std, flow_runtime, factor_codebooks
        if repa_features is not None:
            raise ValueError("REPA features require anchor_target=feature and fm_mode=flow.")
        if factor_targets is None:
            raise ValueError("anchor_target=factor requires factor targets.")
        anchor_module = self.anchor
        if anchor_module is None:
            raise RuntimeError("factor generator is missing its anchor.")
        anchor = anchor_module(condition, target_mask)
        if self.factor_depth is None:
            factor = self.anchor_factor_loss(
                self._factor_output(anchor),
                factor_targets,
                target_mask,
                validate=validate,
                include_top1=include_details,
                include_details=include_details,
            )
        else:
            packed = (
                self.factor_depth.forward_packed(
                    anchor,
                    factor_targets,
                    mask=target_mask,
                    validate=validate,
                )
                if factor_targeter is None
                else self.factor_depth.forward_packed_retargeted(
                    anchor,
                    factor_targets,
                    factor_targeter,
                    mask=target_mask,
                    validate=validate,
                )
            )
            factor = self.anchor_factor_loss.forward_packed(
                packed,
                validate=False,
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
        codes = torch.stack(
            tuple(value.argmax(dim=-1) for value in self._factor_output(anchor)),
            dim=-1,
        )
        return codes.masked_fill(~mask[..., None], 0)

    def _stored_factor_codebooks(self) -> tuple[Tensor, ...]:
        result: list[Tensor] = []
        for name in self._factor_codebook_names:
            value = getattr(self, name, None)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"stored factor codebook {name!r} is missing.")
            result.append(value)
        return tuple(result)


def _fm_sample_inputs(
    condition: Tensor,
    mask: Tensor,
    unconditional_condition: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    target_condition, target_mask = aligned_condition(condition, mask)
    if unconditional_condition is None:
        return target_condition, target_mask, None
    target_unconditional, unconditional_mask = aligned_condition(unconditional_condition, mask)
    if not torch.equal(target_mask, unconditional_mask):
        raise ValueError("conditional and unconditional FM masks must match.")
    return target_condition, target_mask, target_unconditional


def _factor_logits(
    features: Tensor,
    codebooks: tuple[Tensor, ...],
    *,
    temperature: float,
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
        / temperature
        for value, codebook in zip(split, codebooks, strict=True)
    )  # type: ignore[return-value]


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


__all__ = ["FMFeatureGenerator", "factor_codebook_names"]
