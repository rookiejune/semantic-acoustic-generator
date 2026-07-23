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


@dataclass(frozen=True)
class DecoderConfig:
    hidden_dim: int | None = None
    layers: int = 8
    heads: int = 8
    ffn_ratio: int = 4
    repa_feature_dim: int | None = None
    repa_student_layer: int | None = None
    repa_loss_weight: float = 0.0

    def __post_init__(self) -> None:
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
    return DecoderConfig(
        hidden_dim=cast(int | None, config["hidden_dim"]),
        layers=cast(int, config["layers"]),
        heads=cast(int, config["heads"]),
        ffn_ratio=cast(int, config["ffn_ratio"]),
        repa_feature_dim=cast(int | None, config.get("repa_feature_dim")),
        repa_student_layer=cast(int | None, config.get("repa_student_layer")),
        repa_loss_weight=float(config.get("repa_loss_weight", 0.0)),
    )


__all__ = [
    "AdapterType",
    "DecoderConfig",
    "Initialization",
    "Route",
    "decoder_options",
]
