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
    AcousticDecoder,
    FlowAcousticDecoder,
    RVQAcousticDecoder,
)
from semantic_acoustic_codec.runtime.protocol import TeacherCodec


@dataclass(frozen=True)
class RouteModules:
    conditioner: SemanticConditioner
    reference_conditioner: ReferenceConditioner
    decoder: AcousticDecoder
    route: Route


def build_route(
    route: Route,
    teacher: TeacherCodec,
    *,
    condition_dim: int,
    decoder: DecoderConfig | None = None,
    adapter: AdapterType | None = AdapterType.LINEAR,
    initialization: Initialization = Initialization.CODEC,
    seed: int = 0,
) -> RouteModules:
    options = DecoderConfig() if decoder is None else decoder
    conditioner = SemanticConditioner(
        teacher.semantic_codebook,
        condition_dim=condition_dim,
        adapter=adapter,
        initialization=initialization,
        seed=seed,
    )
    reference_conditioner = ReferenceConditioner(
        teacher.acoustic_feature_dim,
        condition_dim,
    )
    if route is Route.FM:
        module = FlowAcousticDecoder(
            condition_dim,
            teacher.acoustic_feature_dim,
            options,
        )
    elif route is Route.RVQ:
        module = RVQAcousticDecoder(condition_dim, teacher.acoustic_codebook_sizes, options)
    else:
        raise AssertionError(f"unsupported route: {route}")
    return RouteModules(
        conditioner=conditioner,
        reference_conditioner=reference_conditioner,
        decoder=module,
        route=route,
    )
