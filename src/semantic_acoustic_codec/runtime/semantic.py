from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodec
from torch import Tensor, nn

from semantic_acoustic_codec._tensor import is_signed_integer_dtype
from semantic_acoustic_codec.config import (
    DecoderConfig,
    Initialization,
    Route,
)
from semantic_acoustic_codec.model.routes import RouteModules, build_route
from semantic_acoustic_codec.runtime.metadata import (
    support_metadata,
    validate_backend_metadata,
    validate_support_metadata,
)

__all__ = [
    "SamplingConfig",
    "SemanticCodecRuntime",
    "SemanticCodecSupport",
    "SemanticSupportConfig",
    "build_support",
]


@dataclass(frozen=True)
class SamplingConfig:
    flow_steps: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    cfg_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.flow_steps < 1:
            raise ValueError("flow_steps must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        _cfg_scale(self.cfg_scale)


@dataclass(frozen=True)
class SemanticSupportConfig:
    route: Route
    condition_dim: int
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    initialization: Initialization = Initialization.CODEC
    seed: int = 0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    feature_mean: tuple[float, ...] | None = None
    feature_std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")
        if (self.feature_mean is None) != (self.feature_std is None):
            raise ValueError("feature_mean and feature_std must be set together.")
        if self.feature_mean is None or self.feature_std is None:
            return
        if len(self.feature_mean) != len(self.feature_std):
            raise ValueError("feature_mean and feature_std must have the same length.")
        if not self.feature_mean:
            raise ValueError("feature normalization must not be empty.")
        if any(value <= 0 for value in self.feature_std):
            raise ValueError("feature_std values must be positive.")


class SemanticCodecSupport(nn.Module):
    """Trainable semantic-to-acoustic unit generator with no codec ownership."""

    def __init__(
        self,
        modules: RouteModules,
        acoustic_feature_dim: int,
        *,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        acoustic_unit_length: int | None = None,
        sampling: SamplingConfig | None = None,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        artifact_backend_metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if acoustic_feature_dim <= 0:
            raise ValueError("acoustic_feature_dim must be positive.")
        if not isinstance(acoustic_layout, AcousticLayout):
            raise TypeError("acoustic_layout must be an AcousticLayout.")
        if acoustic_layout is AcousticLayout.FIXED_LENGTH:
            if acoustic_unit_length is None or acoustic_unit_length <= 0:
                raise ValueError("fixed-length acoustic layout requires a positive unit length.")
        elif acoustic_unit_length is not None:
            raise ValueError("frame-aligned acoustic layout must not set acoustic_unit_length.")
        self.conditioner = modules.conditioner
        self.reference_conditioner = modules.reference_conditioner
        self.generator = modules.generator
        self.route = modules.route
        self.acoustic_feature_dim = acoustic_feature_dim
        self.acoustic_codebook_sizes = modules.acoustic_codebook_sizes
        self.acoustic_layout = acoustic_layout
        self.acoustic_unit_length = acoustic_unit_length
        self.sampling = SamplingConfig() if sampling is None else sampling
        self.artifact_backend_metadata = (
            None if artifact_backend_metadata is None else dict(artifact_backend_metadata)
        )
        self.feature_mean = nn.Buffer(_feature_stat(acoustic_feature_dim, feature_mean, fill=0.0))
        self.feature_std = nn.Buffer(_feature_stat(acoustic_feature_dim, feature_std, fill=1.0))

    @torch.no_grad()
    def sample_features(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        output_length: int | None = None,
        cfg_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        condition = self.condition(
            prepared,
            mask=frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
        )
        if self.route is Route.RVQ:
            raise RuntimeError("RVQ feature conversion requires a codec runtime.")
        guidance_scale = _cfg_scale(self.sampling.cfg_scale if cfg_scale is None else cfg_scale)
        unconditional_condition = None
        if reference_features is not None and guidance_scale != 1.0:
            unconditional_condition = self.condition(
                prepared,
                mask=frame_mask,
                reference_features=None,
                reference_mask=None,
            )
        target_length = self._output_length(output_length)
        target_mask = self._output_mask(frame_mask, target_length)
        features = self.generator.sample_features(
            condition,
            frame_mask,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            flow_steps=self.sampling.flow_steps,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            unconditional_condition=unconditional_condition,
            cfg_scale=guidance_scale,
            acoustic_layout=self.acoustic_layout,
            output_length=target_length,
            generator=generator,
        )
        return features.masked_fill(~target_mask[..., None], 0)

    @torch.no_grad()
    def sample_acoustic_codes(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        condition = self.condition(
            prepared,
            mask=frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
        )
        target_length = self._output_length(output_length)
        return self.generator.sample_acoustic_codes(
            condition,
            frame_mask,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            acoustic_layout=self.acoustic_layout,
            output_length=target_length,
            generator=generator,
        )

    def _output_length(self, value: int | None) -> int | None:
        if self.acoustic_layout is AcousticLayout.FRAME_ALIGNED:
            if value is not None:
                raise ValueError("frame-aligned generation does not accept output_length.")
            return None
        length = self.acoustic_unit_length if value is None else value
        if length is None or length <= 0:
            raise ValueError("fixed-length generation requires a positive output_length.")
        if self.acoustic_unit_length is not None and length != self.acoustic_unit_length:
            raise ValueError(
                "output_length must match the codec acoustic_unit_length "
                f"{self.acoustic_unit_length}, got {length}."
            )
        return length

    def _output_mask(self, frame_mask: Tensor, length: int | None) -> Tensor:
        if length is None:
            return frame_mask
        return torch.ones(
            frame_mask.size(0),
            length,
            device=frame_mask.device,
            dtype=torch.bool,
        )

    def condition(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        use_reference: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask, validate=validate)
        semantic = self.conditioner(prepared, validate=validate)
        reference = self._reference_condition(
            batch_size=prepared.size(0),
            reference_features=reference_features,
            reference_mask=reference_mask,
            use_reference=use_reference,
            validate=validate,
        )
        return (semantic + reference).masked_fill(~frame_mask[..., None], 0)

    def _reference_condition(
        self,
        *,
        batch_size: int,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
        use_reference: Tensor | None,
        validate: bool,
    ) -> Tensor:
        features = reference_features
        if validate and features is not None and reference_mask is not None:
            if reference_mask.shape != features.shape[:2]:
                raise ValueError("reference_mask must align with reference_features.")
            if reference_mask.dtype != torch.bool:
                raise TypeError("reference_mask must be boolean.")

        parameter = self.reference_conditioner.null_condition
        if features is not None:
            features = features.to(device=parameter.device, dtype=parameter.dtype)
        if reference_mask is not None:
            reference_mask = reference_mask.to(device=parameter.device)
        return self.reference_conditioner(
            features,
            mask=reference_mask,
            batch_size=batch_size,
            use_reference=use_reference,
            validate=validate,
        )

    def _semantic_input(
        self,
        value: Tensor,
        mask: Tensor | None,
        *,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if value.dim() == 2:
            semantic = value[:, :, None]
        elif value.dim() == 3 and value.size(-1) == 1:
            semantic = value
        else:
            raise ValueError("semantic_codes must have shape [B, F] or [B, F, 1].")
        if validate and (semantic.size(0) < 1 or semantic.size(1) < 1):
            raise ValueError("semantic_codes must not be empty.")
        if validate and not is_signed_integer_dtype(semantic.dtype):
            raise TypeError("semantic_codes must use a signed integer dtype.")

        reference = self.conditioner.embedding.weight
        prepared = semantic.to(device=reference.device, dtype=torch.long).contiguous()
        if mask is None:
            frame_mask = torch.ones(prepared.shape[:2], device=prepared.device, dtype=torch.bool)
        else:
            if validate and mask.shape != prepared.shape[:2]:
                raise ValueError("mask must align with semantic_codes on [B, F].")
            if validate and mask.dtype != torch.bool:
                raise TypeError("mask must be boolean.")
            frame_mask = mask.to(device=prepared.device)
        if validate and not bool(frame_mask.any(dim=1).all()):
            raise ValueError("each semantic sequence must contain at least one valid frame.")

        if validate:
            valid = prepared[..., 0][frame_mask]
            if bool((valid < 0).any()):
                raise ValueError("valid semantic_codes must not contain negative IDs.")
            if bool((valid >= self.conditioner.semantic_codebook_size).any()):
                raise ValueError("semantic_codes contain an ID outside the semantic codebook.")
        return prepared.masked_fill(~frame_mask[..., None], self.conditioner.semantic_pad_id), frame_mask


class SemanticCodecRuntime:
    """Compose a semantic unit generator with a codec for waveform I/O."""

    def __init__(
        self,
        support: SemanticCodecSupport,
        backend: SemanticAcousticCodec,
    ) -> None:
        self.support = support
        self.backend = backend
        metadata = support.artifact_backend_metadata
        if metadata is None:
            validate_support_metadata(support_metadata(support), backend)
        else:
            validate_backend_metadata(metadata, backend)

    @property
    def sample_rate(self) -> int:
        return self.backend.sample_rate

    @property
    def frame_rate(self) -> float:
        value = self.backend.frame_rate
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("semantic-acoustic backend frame_rate must be positive.")
        return float(value)

    @torch.no_grad()
    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        codes = self.backend.tokenize(audio, sample_rate)
        semantic = codes.semantic
        if semantic.dim() != 3 or semantic.size(-1) < 1:
            raise ValueError("backend semantic codes must have shape [B, F, K].")
        if not is_signed_integer_dtype(semantic.dtype):
            raise TypeError("backend semantic codes must use signed integer codes.")
        return semantic[..., :1].to(dtype=torch.long).contiguous()

    @torch.no_grad()
    def sample_features(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        cfg_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        output_length = self.backend.acoustic_unit_length
        if self.support.route is Route.FM:
            return self.support.sample_features(
                semantic_codes,
                mask=mask,
                reference_features=reference_features,
                reference_mask=reference_mask,
                output_length=output_length,
                cfg_scale=cfg_scale,
                generator=generator,
            )
        codes = self.support.sample_acoustic_codes(
            semantic_codes,
            mask=mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            output_length=output_length,
            generator=generator,
        )
        _, frame_mask = self.support._semantic_input(semantic_codes, mask)
        target_mask = self.support._output_mask(frame_mask, output_length)
        features = self.backend.acoustic_codes_to_features(codes)
        return features.masked_fill(~target_mask.to(device=features.device)[..., None], 0)

    @torch.no_grad()
    def decode(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        cfg_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self.support._semantic_input(semantic_codes, mask)
        prepared, frame_mask = _trim_decode_input(prepared, frame_mask)
        features = self.sample_features(
            prepared,
            mask=frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            cfg_scale=cfg_scale,
            generator=generator,
        )
        return self.backend.decode_features(prepared, features)

    @torch.no_grad()
    def decode_features(
        self,
        semantic_codes: Tensor,
        features: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        prepared, frame_mask = self.support._semantic_input(semantic_codes, mask)
        if features.dim() != 3 or features.size(0) != prepared.size(0):
            raise ValueError("features must have shape [B, acoustic_unit, D].")
        if features.size(-1) != self.support.acoustic_feature_dim:
            raise ValueError("features must match support acoustic_feature_dim.")
        original_length = prepared.size(1)
        prepared, frame_mask = _trim_decode_input(prepared, frame_mask)
        if self.support.acoustic_layout is AcousticLayout.FRAME_ALIGNED:
            if features.size(1) != original_length:
                raise ValueError("frame-aligned features must match the padded semantic length.")
            features = features[:, : prepared.size(1)]
        expected_length = (
            prepared.size(1)
            if self.support.acoustic_layout is AcousticLayout.FRAME_ALIGNED
            else self.support.acoustic_unit_length
        )
        if expected_length is None or features.size(1) != expected_length:
            raise ValueError(
                "features must align with the backend acoustic unit length "
                f"{expected_length}, got {features.size(1)}."
            )
        return self.backend.decode_features(prepared, features)


def build_support(
    config: SemanticSupportConfig,
    *,
    semantic_codebook: Tensor,
    acoustic_feature_dim: int,
    acoustic_codebook_sizes: tuple[int, ...],
    acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
    acoustic_unit_length: int | None = None,
    artifact_backend_metadata: Mapping[str, object] | None = None,
) -> SemanticCodecSupport:
    modules = build_route(
        config.route,
        semantic_codebook,
        acoustic_feature_dim,
        acoustic_codebook_sizes,
        condition_dim=config.condition_dim,
        decoder=config.decoder,
        initialization=config.initialization,
        seed=config.seed,
        acoustic_layout=acoustic_layout,
        acoustic_unit_length=acoustic_unit_length,
    )
    mean = None if config.feature_mean is None else torch.tensor(config.feature_mean, dtype=torch.float32)
    std = None if config.feature_std is None else torch.tensor(config.feature_std, dtype=torch.float32)
    return SemanticCodecSupport(
        modules,
        acoustic_feature_dim,
        acoustic_layout=acoustic_layout,
        acoustic_unit_length=acoustic_unit_length,
        sampling=config.sampling,
        feature_mean=mean,
        feature_std=std,
        artifact_backend_metadata=artifact_backend_metadata,
    )


def _feature_stat(acoustic_feature_dim: int, value: Tensor | None, *, fill: float) -> Tensor:
    if value is None:
        return torch.full((1, 1, acoustic_feature_dim), fill)
    if value.dim() == 1:
        value = value.view(1, 1, -1)
    if value.shape != (1, 1, acoustic_feature_dim):
        raise ValueError("feature normalization must match backend acoustic_feature_dim.")
    if not torch.is_floating_point(value):
        raise TypeError("feature normalization tensors must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("feature normalization tensors must be finite.")
    return value.detach().clone()


def _trim_decode_input(semantic: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    lengths = mask.sum(dim=1)
    if not bool((lengths == lengths[0]).all()):
        raise ValueError(
            "waveform decode requires equal valid semantic lengths; group or decode rows separately."
        )
    length = int(lengths[0])
    expected = torch.arange(mask.size(1), device=mask.device)[None] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("waveform decode mask must describe contiguous right padding.")
    return semantic[:, :length].contiguous(), mask[:, :length].contiguous()


def _cfg_scale(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("cfg_scale must be a number.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("cfg_scale must be finite and non-negative.")
    return result
