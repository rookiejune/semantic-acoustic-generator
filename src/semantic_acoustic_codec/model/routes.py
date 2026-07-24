from __future__ import annotations

from dataclasses import dataclass

from semantic_acoustic_codec.config import (
    AdapterType,
    DecoderConfig,
    Initialization,
    Route,
)
from semantic_acoustic_codec.model.condition import ReferenceConditioner, SemanticConditioner
from semantic_acoustic_codec.model.decoder import (
    CodecUnitGenerator,
    FMFeatureGenerator,
    RVQCodeGenerator,
)
from semantic_acoustic_codec.runtime.protocol import CodecBackend


@dataclass(frozen=True)
class RouteModules:
    conditioner: SemanticConditioner
    reference_conditioner: ReferenceConditioner
    generator: CodecUnitGenerator
    route: Route


def build_route(
    route: Route,
    backend: CodecBackend,
    *,
    condition_dim: int,
    decoder: DecoderConfig | None = None,
    adapter: AdapterType | None = AdapterType.LINEAR,
    initialization: Initialization = Initialization.CODEC,
    seed: int = 0,
) -> RouteModules:
    options = DecoderConfig() if decoder is None else decoder
    conditioner = SemanticConditioner(
        backend.semantic_codebook,
        condition_dim=condition_dim,
        adapter=adapter,
        initialization=initialization,
        seed=seed,
    )
    reference_conditioner = ReferenceConditioner(
        backend.acoustic_feature_dim,
        condition_dim,
    )
    if route is Route.FM:
        module = FMFeatureGenerator(
            condition_dim,
            backend.acoustic_feature_dim,
            options,
        )
    elif route is Route.RVQ:
        module = RVQCodeGenerator(condition_dim, backend.acoustic_codebook_sizes, options)
    else:
        raise AssertionError(f"unsupported route: {route}")
    return RouteModules(
        conditioner=conditioner,
        reference_conditioner=reference_conditioner,
        generator=module,
        route=route,
    )
