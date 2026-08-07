"""Route-specific training targets and validation errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from anytrain.codec import SemanticAcousticCodec, masked_acoustic_features

from semantic_acoustic_generator.backend import LongCatCodebookAdapter
from semantic_acoustic_generator.config import AnchorTarget, Route
from semantic_acoustic_generator.model.code import RVQCodeGenerator
from semantic_acoustic_generator.model.feature import FMFeatureGenerator
from semantic_acoustic_generator.model.generator import DecoderLoss
from semantic_acoustic_generator.runtime.semantic import GeneratorSupport
from semantic_acoustic_generator.types import GeneratorBatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from semantic_acoustic_generator.loss.repa import Teacher


def training_loss(
    support: GeneratorSupport,
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    condition: Tensor,
    *,
    repa_teacher: Teacher | None,
    residual_retarget: bool,
) -> DecoderLoss:
    generator = support.head
    if support.route is Route.FM:
        if not isinstance(generator, FMFeatureGenerator):
            raise TypeError("FM support requires an FMFeatureGenerator.")
        target = (
            target_features(backend, batch)
            if generator.anchor_target is AnchorTarget.FEATURE
            else None
        )
        return generator.loss(
            batch,
            condition,
            target,
            feature_mean=support.feature_mean,
            feature_std=support.feature_std,
            repa_teacher=repa_teacher,
            factor_targets=factor_targets(backend, batch),
            factor_codebooks=(
                None
                if generator.anchor_target is AnchorTarget.FACTOR
                else factor_codebooks(backend)
            ),
            factor_targeter=factor_targeter(
                backend,
                batch,
                residual_retarget=residual_retarget,
            ),
        )
    if not isinstance(generator, RVQCodeGenerator):
        raise TypeError("RVQ support requires an RVQCodeGenerator.")
    return generator.loss(batch, condition)


@torch.no_grad()
def validation_error(
    support: GeneratorSupport,
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    reference_features: Tensor | None,
    reference_mask: Tensor | None,
    generator: torch.Generator,
) -> Tensor:
    if support.route is Route.FM:
        feature_generator = support.head
        if (
            isinstance(feature_generator, FMFeatureGenerator)
            and feature_generator.anchor_target is AnchorTarget.FACTOR
        ):
            condition = support.condition(
                batch.semantic_codes,
                mask=batch.mask,
                reference_features=reference_features,
                reference_mask=reference_mask,
            )
            predicted = feature_generator.sample_factor_codes(condition, batch.mask)
            target = factor_targets(backend, batch)
            if target is None:
                raise RuntimeError("factor validation requires factor targets.")
            error = predicted.ne(target.to(device=predicted.device)).float().mean(-1)
            return _masked_mean(error, batch.acoustic_mask)
        prediction = support.sample_features(
            batch.semantic_codes,
            mask=batch.mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            generator=generator,
        )
        target = target_features(backend, batch).to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        return _masked_mean((prediction - target).square().mean(dim=-1), batch.acoustic_mask)
    prediction = support.sample_acoustic_codes(
        batch.semantic_codes,
        mask=batch.mask,
        reference_features=reference_features,
        reference_mask=reference_mask,
        generator=generator,
    )
    target_codes = batch.acoustic_codes.to(device=prediction.device)
    error = (prediction != target_codes).float().mean(dim=-1)
    return _masked_mean(error, batch.acoustic_mask)


def validation_metric(support: GeneratorSupport) -> str:
    if support.route is Route.RVQ:
        return "code_error"
    generator = support.head
    if isinstance(generator, FMFeatureGenerator) and generator.anchor_target is AnchorTarget.FACTOR:
        return "factor_code_error"
    return "feature_mse"


@torch.no_grad()
def target_features(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
) -> Tensor:
    return masked_acoustic_features(
        backend,
        batch.acoustic_codes,
        batch.acoustic_mask,
        validate=False,
    )


@torch.no_grad()
def reference_features(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    indices: Tensor | None = None,
) -> Tensor | None:
    if not batch.has_reference:
        return None
    reference = batch.reference
    codes = reference.acoustic_codes
    mask = reference.acoustic_mask
    if indices is not None:
        selected = indices.to(device=codes.device)
        codes = codes.index_select(0, selected)
        mask = mask.index_select(0, selected)
    mask = mask.to(device=batch.semantic_codes.device)
    return masked_acoustic_features(backend, codes, mask, validate=False)


@torch.no_grad()
def factor_targets(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
) -> Tensor | None:
    if not isinstance(backend, LongCatCodebookAdapter):
        return None
    codes = batch.acoustic_codes.masked_fill(~batch.acoustic_mask[..., None], 0)
    return backend.factor_codes(codes, validate_values=False)


def factor_codebooks(backend: SemanticAcousticCodec) -> tuple[Tensor, ...] | None:
    if not isinstance(backend, LongCatCodebookAdapter):
        return None
    return backend.factor_codebooks


def factor_targeter(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    residual_retarget: bool,
) -> Callable[[int, Tensor], Tensor] | None:
    if not residual_retarget:
        return None
    if not isinstance(backend, LongCatCodebookAdapter):
        raise RuntimeError("residual retargeting requires a LongCat codebook adapter.")
    return backend.residual_factor_targeter(
        batch.acoustic_codes,
        batch.acoustic_mask,
    )


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    aligned = mask.to(device=value.device)
    return value[aligned].mean()
