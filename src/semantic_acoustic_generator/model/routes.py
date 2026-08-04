from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anytrain.codec import AcousticLayout

from semantic_acoustic_generator.config import DecoderConfig, FMMode, Initialization, Route
from semantic_acoustic_generator.model.condition import ReferenceConditioner, SemanticConditioner
from semantic_acoustic_generator.model.decoder import (
    AcousticUnitGenerator,
    FMFeatureGenerator,
    RVQCodeGenerator,
)

if TYPE_CHECKING:
    from torch import Tensor



@dataclass(frozen=True)
class RouteModules:
    conditioner: SemanticConditioner
    reference_conditioner: ReferenceConditioner
    generator: AcousticUnitGenerator
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
    acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
    acoustic_unit_length: int | None = None,
) -> RouteModules:
    if not isinstance(acoustic_layout, AcousticLayout):
        raise TypeError("acoustic_layout must be an AcousticLayout.")
    if acoustic_layout is AcousticLayout.FIXED_LENGTH:
        if acoustic_unit_length is None or acoustic_unit_length <= 0:
            raise ValueError("fixed-length routes require a positive acoustic_unit_length.")
        fixed_length = acoustic_unit_length
    else:
        if acoustic_unit_length is not None:
            raise ValueError("frame-aligned routes must not set acoustic_unit_length.")
        fixed_length = None
    options = DecoderConfig() if decoder is None else decoder
    if route is not Route.FM and options.fm_mode is not FMMode.FLOW:
        raise ValueError("fm_mode is only supported by the FM route.")
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
            fixed_length=fixed_length,
        )
    elif route is Route.RVQ:
        module = RVQCodeGenerator(
            condition_dim,
            acoustic_codebook_sizes,
            options,
            fixed_length=fixed_length,
        )
    else:
        raise AssertionError(f"unsupported route: {route}")
    return RouteModules(
        conditioner=conditioner,
        reference_conditioner=reference_conditioner,
        generator=module,
        route=route,
        acoustic_codebook_sizes=acoustic_codebook_sizes,
    )
