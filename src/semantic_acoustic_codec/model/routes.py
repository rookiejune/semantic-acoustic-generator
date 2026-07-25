from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from semantic_acoustic_codec.config import DecoderConfig, Initialization, Route
from semantic_acoustic_codec.model.condition import ReferenceConditioner, SemanticConditioner
from semantic_acoustic_codec.model.decoder import (
    CodecUnitGenerator,
    FMFeatureGenerator,
    RVQCodeGenerator,
)

if TYPE_CHECKING:
    from torch import Tensor



@dataclass(frozen=True)
class RouteModules:
    conditioner: SemanticConditioner
    reference_conditioner: ReferenceConditioner
    generator: CodecUnitGenerator
    route: Route
    acoustic_codebook_sizes: tuple[int, ...]


def build_route(
    route: Route,
    semantic_codebook: Tensor,
    acoustic_feature_dim: int,
    acoustic_codebook_sizes: tuple[int, ...],
    *,
    condition_dim: int,
    decoder: DecoderConfig | None = None,
    initialization: Initialization = Initialization.CODEC,
    seed: int = 0,
) -> RouteModules:
    options = DecoderConfig() if decoder is None else decoder
    conditioner = SemanticConditioner(
        semantic_codebook,
        condition_dim=condition_dim,
        initialization=initialization,
        seed=seed,
    )
    reference_conditioner = ReferenceConditioner(
        acoustic_feature_dim,
        condition_dim,
    )
    if route is Route.FM:
        module = FMFeatureGenerator(
            condition_dim,
            acoustic_feature_dim,
            options,
        )
    elif route is Route.RVQ:
        module = RVQCodeGenerator(condition_dim, acoustic_codebook_sizes, options)
    else:
        raise AssertionError(f"unsupported route: {route}")
    return RouteModules(
        conditioner=conditioner,
        reference_conditioner=reference_conditioner,
        generator=module,
        route=route,
        acoustic_codebook_sizes=acoustic_codebook_sizes,
    )
