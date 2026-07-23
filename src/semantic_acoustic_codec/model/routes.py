from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from semantic_acoustic_codec.config import AdapterType, DecoderConfig, Initialization, Route
from semantic_acoustic_codec.model.condition import ReferenceConditioner, SemanticConditioner
from semantic_acoustic_codec.model.dit import DiTDecoder
from semantic_acoustic_codec.model.rvq import AcousticRVQDecoder
from semantic_acoustic_codec.runtime.protocol import TeacherCodec


@dataclass(frozen=True)
class RouteModules:
    conditioner: SemanticConditioner
    reference_conditioner: ReferenceConditioner
    decoder: nn.Module
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
        module: nn.Module = DiTDecoder(
            condition_dim,
            teacher.acoustic_feature_dim,
            hidden_dim=options.hidden_dim,
            layers=options.layers,
            heads=options.heads,
            ffn_ratio=options.ffn_ratio,
            repa_feature_dim=options.repa_feature_dim,
            repa_student_layer=options.repa_student_layer,
        )
    elif route is Route.RVQ:
        sizes = teacher.acoustic_codebook_sizes
        if not sizes:
            raise ValueError("RVQ route requires acoustic codebooks.")
        module = AcousticRVQDecoder(
            condition_dim,
            len(sizes),
            sizes,
            hidden_dim=options.hidden_dim,
            layers=options.layers,
            heads=options.heads,
            ffn_ratio=options.ffn_ratio,
        )
    else:
        raise AssertionError(f"unsupported route: {route}")
    return RouteModules(
        conditioner=conditioner,
        reference_conditioner=reference_conditioner,
        decoder=module,
        route=route,
    )


@torch.no_grad()
def teacher_features(teacher: TeacherCodec, acoustic_codes: Tensor, mask: Tensor) -> Tensor:
    if acoustic_codes.dim() != 3 or mask.shape != acoustic_codes.shape[:2]:
        raise ValueError("acoustic_codes and mask must have shapes [B, F, K] and [B, F].")
    features = teacher.acoustic_codes_to_features(acoustic_codes.masked_fill(~mask[..., None], 0))
    return features.masked_fill(~mask[..., None], 0)
