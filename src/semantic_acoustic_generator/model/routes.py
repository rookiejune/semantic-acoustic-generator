from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anytrain.codec import AcousticLayout

from semantic_acoustic_generator.config import (
    AnchorTarget,
    BackboneConfig,
    FactorPredictor,
    FMMode,
    HeadConfig,
    Route,
)
from semantic_acoustic_generator.model.backbone import QwenBackbone
from semantic_acoustic_generator.model.code import RVQCodeGenerator
from semantic_acoustic_generator.model.condition import ReferenceConditioner
from semantic_acoustic_generator.model.feature import FMFeatureGenerator
from semantic_acoustic_generator.model.generator import AcousticHead
from semantic_acoustic_generator.model.model import AcousticGeneratorModel

if TYPE_CHECKING:
    from torch import Tensor

@dataclass(frozen=True)
class RouteModules:
    model: AcousticGeneratorModel
    reference_conditioner: ReferenceConditioner
    route: Route
    acoustic_codebook_sizes: tuple[int, ...]

    @property
    def backbone(self) -> QwenBackbone:
        return self.model.backbone

    @property
    def head(self) -> AcousticHead:
        return self.model.head

    @property
    def conditioner(self):
        return self.backbone.embedding

    @property
    def generator(self) -> AcousticHead:
        return self.head


def build_route(
    route: Route,
    semantic_codebook: Tensor,
    acoustic_feature_dim: int,
    acoustic_codebook_sizes: tuple[int, ...],
    *,
    backbone: BackboneConfig | None = None,
    head: HeadConfig | None = None,
    acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
    acoustic_unit_length: int | None = None,
    factor_codebooks: tuple[Tensor, ...] | None = None,
) -> RouteModules:
    if not isinstance(acoustic_layout, AcousticLayout):
        raise TypeError("acoustic_layout must be an AcousticLayout.")
    if acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
        raise ValueError("generator routes require frame-aligned acoustic units.")
    if acoustic_unit_length is not None:
        raise ValueError("frame-aligned routes must not set acoustic_unit_length.")
    backbone_options = BackboneConfig() if backbone is None else backbone
    head_options = HeadConfig() if head is None else head
    if route is not Route.FM and head_options.fm_mode is not FMMode.FLOW:
        raise ValueError("fm_mode is only supported by the FM route.")
    if (
        route is Route.FM
        and head_options.factor_predictor is not FactorPredictor.PARALLEL
        and head_options.anchor_target is not AnchorTarget.FACTOR
    ):
        raise ValueError("FM depth factor predictors require anchor_target=factor.")
    semantic_backbone = QwenBackbone(semantic_codebook, backbone_options)
    reference_conditioner = ReferenceConditioner(
        acoustic_feature_dim,
        backbone_options.hidden_dim,
    )
    if route is Route.FM:
        output_head = FMFeatureGenerator(
            backbone_options.hidden_dim,
            acoustic_feature_dim,
            head_options,
            factor_codebooks=factor_codebooks,
        )
    elif route is Route.RVQ:
        if factor_codebooks is None:
            raise ValueError("RVQ route requires AGRVQ factor codebook pairs.")
        output_head = RVQCodeGenerator(
            backbone_options.hidden_dim,
            acoustic_codebook_sizes,
            head_options,
            factor_codebooks=factor_codebooks,
        )
    else:
        raise AssertionError(f"unsupported route: {route}")
    return RouteModules(
        model=AcousticGeneratorModel(semantic_backbone, output_head),
        reference_conditioner=reference_conditioner,
        route=route,
        acoustic_codebook_sizes=acoustic_codebook_sizes,
    )
