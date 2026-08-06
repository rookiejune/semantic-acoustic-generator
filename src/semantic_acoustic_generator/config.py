from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from semantic_acoustic_generator._compat import StrEnum, auto

if TYPE_CHECKING:
    from collections.abc import Mapping


class Route(StrEnum):
    FM = auto()
    RVQ = auto()


class FeatureAdapter(StrEnum):
    NONE = auto()
    LONGCAT_FIRST_CODEBOOK = auto()
    LONGCAT_CODEBOOKS = auto()


class FMMode(StrEnum):
    FLOW = auto()
    ANCHOR = auto()
    RESIDUAL = auto()


class AnchorContext(StrEnum):
    LOCAL = auto()
    TRANSFORMER = auto()
    QWEN_FILM = auto()


class AnchorTarget(StrEnum):
    FEATURE = auto()
    FACTOR = auto()


class FactorPredictor(StrEnum):
    PARALLEL = auto()
    DEPTH_AR = auto()
    DEPTH_RECURRENT = auto()


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
    fm_mode: FMMode = FMMode.FLOW
    anchor_context: AnchorContext = AnchorContext.LOCAL
    anchor_target: AnchorTarget = AnchorTarget.FEATURE
    factor_predictor: FactorPredictor = FactorPredictor.PARALLEL
    anchor_hidden_dim: int = 512
    anchor_layers: int = 4
    anchor_kernel_size: int = 3
    anchor_cosine_weight: float = 0.1
    anchor_factor_weight: float = 0.1
    anchor_factor_temperature: float = 0.1

    def __post_init__(self) -> None:
        _optional_int(self.hidden_dim, name="hidden_dim")
        _int(self.layers, name="layers")
        _int(self.heads, name="heads")
        _int(self.ffn_ratio, name="ffn_ratio")
        if not isinstance(self.rvq_predictor, RVQPredictor):
            raise TypeError("rvq_predictor must be an RVQPredictor.")
        _int(self.mtp_layers, name="mtp_layers")
        _int(self.mtp_heads, name="mtp_heads")
        _optional_int(self.repa_feature_dim, name="repa_feature_dim")
        _optional_int(self.repa_student_layer, name="repa_student_layer")
        _float(self.repa_loss_weight, name="repa_loss_weight")
        if not isinstance(self.fm_mode, FMMode):
            raise TypeError("fm_mode must be an FMMode.")
        if not isinstance(self.anchor_context, AnchorContext):
            raise TypeError("anchor_context must be an AnchorContext.")
        if not isinstance(self.anchor_target, AnchorTarget):
            raise TypeError("anchor_target must be an AnchorTarget.")
        if not isinstance(self.factor_predictor, FactorPredictor):
            raise TypeError("factor_predictor must be a FactorPredictor.")
        _int(self.anchor_hidden_dim, name="anchor_hidden_dim")
        _int(self.anchor_layers, name="anchor_layers")
        _int(self.anchor_kernel_size, name="anchor_kernel_size")
        _float(self.anchor_cosine_weight, name="anchor_cosine_weight")
        _float(self.anchor_factor_weight, name="anchor_factor_weight")
        _float(self.anchor_factor_temperature, name="anchor_factor_temperature")
        if self.layers <= 0 or self.heads <= 0 or self.ffn_ratio <= 0:
            raise ValueError("decoder depth, heads, and FFN ratio must be positive.")
        if self.mtp_layers <= 0 or self.mtp_heads <= 0:
            raise ValueError("MTP depth and heads must be positive.")
        if self.repa_loss_weight < 0:
            raise ValueError("repa_loss_weight must be non-negative.")
        if self.repa_loss_weight > 0 and self.repa_feature_dim is None:
            raise ValueError("repa_feature_dim is required when repa_loss_weight is positive.")
        if self.anchor_hidden_dim <= 0 or self.anchor_layers <= 0:
            raise ValueError("anchor hidden_dim and layers must be positive.")
        if (
            self.anchor_context in {AnchorContext.TRANSFORMER, AnchorContext.QWEN_FILM}
            and self.anchor_hidden_dim % self.heads != 0
        ):
            raise ValueError("attention anchor hidden_dim must be divisible by heads.")
        if self.anchor_kernel_size <= 0 or self.anchor_kernel_size % 2 == 0:
            raise ValueError("anchor_kernel_size must be a positive odd integer.")
        if self.anchor_cosine_weight < 0 or self.anchor_factor_weight < 0:
            raise ValueError("anchor loss weights must be non-negative.")
        if self.anchor_factor_temperature <= 0:
            raise ValueError("anchor_factor_temperature must be positive.")
        if self.anchor_target is AnchorTarget.FACTOR and self.fm_mode is not FMMode.ANCHOR:
            raise ValueError("anchor_target=factor requires fm_mode=anchor.")
        if self.factor_predictor is not FactorPredictor.PARALLEL and (
            self.anchor_target is not AnchorTarget.FACTOR
        ):
            raise ValueError("depth factor predictors require anchor_target=factor.")
        if self.fm_mode is not FMMode.FLOW and self.repa_loss_weight > 0:
            raise ValueError("REPA is only supported by fm_mode=flow.")


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
    fm_mode = config.get("fm_mode", FMMode.FLOW.value)
    if not isinstance(fm_mode, str):
        raise TypeError("fm_mode must be a string.")
    anchor_context = config.get("anchor_context", AnchorContext.LOCAL.value)
    if not isinstance(anchor_context, str):
        raise TypeError("anchor_context must be a string.")
    anchor_target = config.get("anchor_target", AnchorTarget.FEATURE.value)
    if not isinstance(anchor_target, str):
        raise TypeError("anchor_target must be a string.")
    factor_predictor = config.get("factor_predictor", FactorPredictor.PARALLEL.value)
    if not isinstance(factor_predictor, str):
        raise TypeError("factor_predictor must be a string.")
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
        fm_mode=FMMode(fm_mode),
        anchor_context=AnchorContext(anchor_context),
        anchor_target=AnchorTarget(anchor_target),
        factor_predictor=FactorPredictor(factor_predictor),
        anchor_hidden_dim=_int(config.get("anchor_hidden_dim", 512), name="anchor_hidden_dim"),
        anchor_layers=_int(config.get("anchor_layers", 4), name="anchor_layers"),
        anchor_kernel_size=_int(
            config.get("anchor_kernel_size", 3),
            name="anchor_kernel_size",
        ),
        anchor_cosine_weight=_float(
            config.get("anchor_cosine_weight", 0.1),
            name="anchor_cosine_weight",
        ),
        anchor_factor_weight=_float(
            config.get("anchor_factor_weight", 0.1),
            name="anchor_factor_weight",
        ),
        anchor_factor_temperature=_float(
            config.get("anchor_factor_temperature", 0.1),
            name="anchor_factor_temperature",
        ),
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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


__all__ = [
    "AnchorContext",
    "AnchorTarget",
    "DecoderConfig",
    "FactorPredictor",
    "FeatureAdapter",
    "FMMode",
    "Initialization",
    "RVQPredictor",
    "Route",
    "decoder_options",
]
