"""Generator module construction and feature-statistics services."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from anytrain.codec import SemanticAcousticCodec, masked_acoustic_features, semantic_acoustic_spec

from semantic_acoustic_generator.backend import LongCatCodebookAdapter, adapt_backend
from semantic_acoustic_generator.config import AnchorTarget, Route
from semantic_acoustic_generator.pl_module.module import GeneratorModule
from semantic_acoustic_generator.runtime.semantic import (
    GeneratorConfig,
    build_support,
)
from semantic_acoustic_generator.types import GeneratorBatch

if TYPE_CHECKING:
    from torch import Tensor

    from semantic_acoustic_generator.loss.repa import Teacher


@torch.no_grad()
def build_module(
    backend: SemanticAcousticCodec,
    config: GeneratorConfig,
    sample: GeneratorBatch | None = None,
    *,
    normalize_features: bool = True,
    feature_mean: tuple[float, ...] | None = None,
    feature_std: tuple[float, ...] | None = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    reference_dropout: float = 0.5,
    validation_seed: int = 0,
    residual_retarget: bool = False,
    repa_teacher: Teacher | None = None,
) -> GeneratorModule:
    rvq_factor_codebooks = (
        LongCatCodebookAdapter(
            backend,
            codebooks=len(backend.acoustic_codebook_sizes),
        ).factor_codebooks
        if config.route is Route.RVQ and backend.name == "longcat"
        else None
    )
    backend = adapt_backend(
        backend,
        config.feature_adapter,
        codebooks=config.feature_codebooks,
    )
    normalize_features = (
        normalize_features
        and config.route is not Route.RVQ
        and config.head.anchor_target is AnchorTarget.FEATURE
    )
    if normalize_features:
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("feature_mean and feature_std must be provided together.")
        if feature_mean is None or feature_std is None:
            if sample is None:
                raise ValueError("feature normalization requires dataset feature statistics.")
            feature_mean, feature_std = feature_stats(backend, sample)
        config = replace(config, feature_mean=feature_mean, feature_std=feature_std)
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
        factor_codebooks=(
            rvq_factor_codebooks
            if config.route is Route.RVQ
            else (
                backend.factor_codebooks
                if isinstance(backend, LongCatCodebookAdapter)
                and config.head.anchor_target is AnchorTarget.FACTOR
                else None
            )
        ),
    )
    return GeneratorModule(
        support,
        config,
        backend=backend,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        reference_dropout=reference_dropout,
        validation_seed=validation_seed,
        residual_retarget=residual_retarget,
        repa_teacher=repa_teacher,
    )


@torch.no_grad()
def feature_stats(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    acoustic_mask = batch.acoustic_mask
    target = masked_acoustic_features(backend, batch.acoustic_codes, acoustic_mask).float()
    acoustic_mask = acoustic_mask.to(device=target.device)
    valid = target[acoustic_mask]
    if valid.numel() == 0:
        raise ValueError("feature stats require at least one valid frame.")
    mean = valid.mean(dim=0)
    std = valid.std(dim=0, correction=0).clamp_min(1e-5)
    return _tuple(mean), _tuple(std)


@torch.no_grad()
def dataset_feature_stats(
    backend: SemanticAcousticCodec,
    batches: Iterable[GeneratorBatch],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mean: Tensor | None = None
    m2: Tensor | None = None
    count = 0
    for batch in batches:
        mask = batch.acoustic_mask
        target = masked_acoustic_features(backend, batch.acoustic_codes, mask).float()
        valid = target[mask.to(device=target.device)]
        if valid.numel() == 0:
            continue
        batch_variance, batch_mean = torch.var_mean(valid, dim=0, correction=0)
        batch_count = valid.size(0)
        batch_mean = batch_mean.double()
        batch_m2 = batch_variance.double() * batch_count
        if mean is None or m2 is None:
            mean = batch_mean
            m2 = batch_m2
            count = batch_count
            continue
        combined_count = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * (batch_count / combined_count)
        m2 = m2 + batch_m2 + delta.square() * (count * batch_count / combined_count)
        count = combined_count
    if count == 0 or mean is None or m2 is None:
        raise ValueError("feature stats require at least one valid acoustic unit.")
    std = (m2 / count).clamp_min(0).sqrt().clamp_min(1e-5)
    stats = torch.stack((mean, std)).cpu()
    return (
        tuple(float(item) for item in stats[0]),
        tuple(float(item) for item in stats[1]),
    )


def _tuple(value: Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu())


__all__ = ["build_module", "dataset_feature_stats", "feature_stats"]
