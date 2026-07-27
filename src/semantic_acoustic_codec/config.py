from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from semantic_acoustic_codec._compat import StrEnum, auto

if TYPE_CHECKING:
    from collections.abc import Mapping


class Route(StrEnum):
    FM = auto()
    RVQ = auto()


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
    rvq_predictor: RVQPredictor = RVQPredictor.MTP
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
    predictor = config.get("rvq_predictor", RVQPredictor.MTP.value)
    if not isinstance(predictor, str):
        raise TypeError("rvq_predictor must be a string.")
    return DecoderConfig(
        hidden_dim=_optional_int(config.get("hidden_dim"), name="hidden_dim"),
        layers=_int(config["layers"], name="layers"),
        heads=_int(config["heads"], name="heads"),
        ffn_ratio=_int(config["ffn_ratio"], name="ffn_ratio"),
        rvq_predictor=RVQPredictor(predictor),
        mtp_layers=_int(config.get("mtp_layers", 2), name="mtp_layers"),
        mtp_heads=_int(config.get("mtp_heads", 4), name="mtp_heads"),
        repa_feature_dim=_optional_int(config.get("repa_feature_dim"), name="repa_feature_dim"),
        repa_student_layer=_optional_int(
            config.get("repa_student_layer"),
            name="repa_student_layer",
        ),
        repa_loss_weight=_float(config.get("repa_loss_weight", 0.0), name="repa_loss_weight"),
    )


def _optional_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None.")
    return value


def _int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    return float(value)


__all__ = [
    "DecoderConfig",
    "Initialization",
    "RVQPredictor",
    "Route",
    "decoder_options",
]
