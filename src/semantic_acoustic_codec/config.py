from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from semantic_acoustic_codec._compat import StrEnum, auto


class Route(StrEnum):
    FM = auto()
    RVQ = auto()


class AdapterType(StrEnum):
    LINEAR = auto()
    MLP = auto()


class Initialization(StrEnum):
    CODEC = auto()
    RANDOM = auto()


class RVQPredictor(StrEnum):
    CODEBOOK_AR = auto()
    MTP = auto()


@dataclass(frozen=True)
class DecoderConfig:
    hidden_dim: int | None = None
    layers: int = 8
    heads: int = 8
    ffn_ratio: int = 4
    rvq_predictor: RVQPredictor = RVQPredictor.CODEBOOK_AR
    mtp_layers: int = 2
    mtp_heads: int = 4
    repa_feature_dim: int | None = None
    repa_student_layer: int | None = None
    repa_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.layers <= 0 or self.heads <= 0 or self.ffn_ratio <= 0:
            raise ValueError("decoder depth, heads, and FFN ratio must be positive.")
        if self.mtp_layers <= 0 or self.mtp_heads <= 0:
            raise ValueError("MTP depth and heads must be positive.")
        if self.repa_loss_weight < 0:
            raise ValueError("repa_loss_weight must be non-negative.")
        if self.repa_loss_weight > 0 and self.repa_feature_dim is None:
            raise ValueError("repa_feature_dim is required when repa_loss_weight is positive.")


def decoder_options(
    config: DecoderConfig | Mapping[str, object] | None,
) -> DecoderConfig:
    if config is None:
        return DecoderConfig()
    if isinstance(config, DecoderConfig):
        return config
    predictor = cast(str, config.get("rvq_predictor", "codebook_ar"))
    return DecoderConfig(
        hidden_dim=cast(int | None, config["hidden_dim"]),
        layers=cast(int, config["layers"]),
        heads=cast(int, config["heads"]),
        ffn_ratio=cast(int, config["ffn_ratio"]),
        rvq_predictor=RVQPredictor(predictor),
        mtp_layers=int(cast(int, config.get("mtp_layers", 2))),
        mtp_heads=int(cast(int, config.get("mtp_heads", 4))),
        repa_feature_dim=cast(int | None, config.get("repa_feature_dim")),
        repa_student_layer=cast(int | None, config.get("repa_student_layer")),
        repa_loss_weight=float(cast(float, config.get("repa_loss_weight", 0.0))),
    )


__all__ = [
    "AdapterType",
    "DecoderConfig",
    "Initialization",
    "RVQPredictor",
    "Route",
    "decoder_options",
]
