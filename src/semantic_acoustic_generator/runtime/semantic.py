from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from anytrain.codec import (
    AcousticLayout,
    SemanticAcousticCodec,
    SemanticAcousticCodecSpec,
    semantic_acoustic_spec,
)
from torch import Tensor, nn

from semantic_acoustic_generator._tensor import is_signed_integer_dtype
from semantic_acoustic_generator.backend import adapt_backend
from semantic_acoustic_generator.config import (
    DecoderConfig,
    FeatureAdapter,
    Initialization,
    Route,
)
from semantic_acoustic_generator.model.decoder import AcousticCodeSampler, FeatureSampler
from semantic_acoustic_generator.model.routes import RouteModules, build_route
from semantic_acoustic_generator.runtime.metadata import (
    support_metadata,
    validate_backend_metadata,
    validate_support_metadata,
)

__all__ = [
    "SamplingConfig",
    "GeneratorRuntime",
    "GeneratorSupport",
    "GeneratorConfig",
    "build_support",
]


@dataclass(frozen=True)
class SamplingConfig:
    flow_steps: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    cfg_scale: float = 1.0

    def __post_init__(self) -> None:
        _integer(self.flow_steps, name="flow_steps")
        temperature = _finite_number(self.temperature, name="temperature")
        top_p = _finite_number(self.top_p, name="top_p")
        if self.flow_steps < 1:
            raise ValueError("flow_steps must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        _cfg_scale(self.cfg_scale)


@dataclass(frozen=True)
class GeneratorConfig:
    route: Route
    condition_dim: int
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    initialization: Initialization = Initialization.CODEC
    seed: int = 0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    feature_adapter: FeatureAdapter = FeatureAdapter.NONE
    feature_mean: tuple[float, ...] | None = None
    feature_std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route, Route):
            raise TypeError("route must be a Route.")
        _integer(self.condition_dim, name="condition_dim")
        if not isinstance(self.decoder, DecoderConfig):
            raise TypeError("decoder must be a DecoderConfig.")
        if not isinstance(self.initialization, Initialization):
            raise TypeError("initialization must be an Initialization.")
        _integer(self.seed, name="seed")
        if not isinstance(self.sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig.")
        if not isinstance(self.feature_adapter, FeatureAdapter):
            raise TypeError("feature_adapter must be a FeatureAdapter.")
        if self.condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")
        if self.feature_adapter is not FeatureAdapter.NONE and self.route is not Route.FM:
            raise ValueError("feature_adapter requires the FM route.")
        if (self.feature_mean is None) != (self.feature_std is None):
            raise ValueError("feature_mean and feature_std must be set together.")
        if self.feature_mean is None or self.feature_std is None:
            return
        _feature_values(self.feature_mean, name="feature_mean")
        _feature_values(self.feature_std, name="feature_std")
        if len(self.feature_mean) != len(self.feature_std):
            raise ValueError("feature_mean and feature_std must have the same length.")
        if not self.feature_mean:
            raise ValueError("feature normalization must not be empty.")
        if any(value <= 0 for value in self.feature_std):
            raise ValueError("feature_std values must be positive.")


class GeneratorSupport(nn.Module):
    """Trainable semantic-to-acoustic unit generator with no codec ownership."""

    def __init__(
        self,
        modules: RouteModules,
        codec_spec: SemanticAcousticCodecSpec,
        *,
        config: GeneratorConfig | None = None,
        sampling: SamplingConfig | None = None,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        artifact_backend_metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(codec_spec, SemanticAcousticCodecSpec):
            raise TypeError("codec_spec must be a SemanticAcousticCodecSpec.")
        _validate_frame_aligned_spec(codec_spec)
        if modules.acoustic_codebook_sizes != codec_spec.acoustic_codebook_sizes:
            raise ValueError("route acoustic codebooks must match codec_spec.")
        self.conditioner = modules.conditioner
        self.reference_conditioner = modules.reference_conditioner
        self.generator = modules.generator
        self.route = modules.route
        self.codec_spec = codec_spec
        self.acoustic_feature_dim = codec_spec.acoustic_feature_dim
        self.acoustic_codebook_sizes = modules.acoustic_codebook_sizes
        self.acoustic_layout = codec_spec.acoustic_layout
        self.acoustic_unit_length = codec_spec.acoustic_unit_length
        self.config = config
        self.sampling = SamplingConfig() if sampling is None else sampling
        self.artifact_backend_metadata = (
            None if artifact_backend_metadata is None else dict(artifact_backend_metadata)
        )
        self.feature_mean = nn.Buffer(
            _feature_stat(codec_spec.acoustic_feature_dim, feature_mean, fill=0.0)
        )
        self.feature_std = nn.Buffer(
            _feature_stat(codec_spec.acoustic_feature_dim, feature_std, fill=1.0)
        )

    @property
    def feature_sampler(self) -> FeatureSampler:
        if not isinstance(self.generator, FeatureSampler):
            raise RuntimeError("feature generation requires the FM route.")
        return self.generator

    @property
    def code_sampler(self) -> AcousticCodeSampler:
        if not isinstance(self.generator, AcousticCodeSampler):
            raise RuntimeError("acoustic-code generation requires the RVQ route.")
        return self.generator

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
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        return self._sample_features(
            prepared,
            frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            cfg_scale=cfg_scale,
            generator=generator,
        )

    def _sample_features(
        self,
        prepared: Tensor,
        frame_mask: Tensor,
        *,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
        cfg_scale: float | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        semantic = self.conditioner(prepared, validate=False)
        condition = self._condition(
            semantic,
            frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            use_reference=None,
            reference_indices=None,
            validate=True,
        )
        if self.route is Route.RVQ:
            raise RuntimeError("RVQ feature conversion requires a codec runtime.")
        guidance_scale = _cfg_scale(self.sampling.cfg_scale if cfg_scale is None else cfg_scale)
        unconditional_condition = None
        if reference_features is not None and guidance_scale != 1.0:
            unconditional_condition = self._condition(
                semantic,
                frame_mask,
                reference_features=None,
                reference_mask=None,
                use_reference=None,
                reference_indices=None,
                validate=False,
            )
        features = self.feature_sampler.sample_features(
            condition,
            frame_mask,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            flow_steps=self.sampling.flow_steps,
            unconditional_condition=unconditional_condition,
            cfg_scale=guidance_scale,
            generator=generator,
        )
        return features.masked_fill(~frame_mask[..., None], 0)

    @torch.no_grad()
    def sample_acoustic_codes(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        return self._sample_acoustic_codes(
            prepared,
            frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            generator=generator,
        )

    def _sample_acoustic_codes(
        self,
        prepared: Tensor,
        frame_mask: Tensor,
        *,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        semantic = self.conditioner(prepared, validate=False)
        condition = self._condition(
            semantic,
            frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            use_reference=None,
            reference_indices=None,
            validate=True,
        )
        return self.code_sampler.sample_acoustic_codes(
            condition,
            frame_mask,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            generator=generator,
        )

    def condition(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        use_reference: Tensor | None = None,
        reference_indices: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask, validate=validate)
        semantic = self.conditioner(prepared, validate=False)
        return self._condition(
            semantic,
            frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            use_reference=use_reference,
            reference_indices=reference_indices,
            validate=validate,
        )

    def _condition(
        self,
        semantic: Tensor,
        frame_mask: Tensor,
        *,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
        use_reference: Tensor | None,
        reference_indices: Tensor | None,
        validate: bool,
    ) -> Tensor:
        reference = self._reference_condition(
            batch_size=semantic.size(0),
            reference_features=reference_features,
            reference_mask=reference_mask,
            use_reference=use_reference,
            reference_indices=reference_indices,
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
        reference_indices: Tensor | None,
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
        if reference_indices is not None:
            reference_indices = reference_indices.to(device=parameter.device)
        return self.reference_conditioner(
            features,
            mask=reference_mask,
            batch_size=batch_size,
            use_reference=use_reference,
            row_indices=reference_indices,
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
        return prepared.masked_fill(
            ~frame_mask[..., None], self.conditioner.semantic_pad_id
        ), frame_mask


class GeneratorRuntime:
    """Compose a semantic unit generator with a codec for waveform I/O."""

    def __init__(
        self,
        support: GeneratorSupport,
        backend: SemanticAcousticCodec,
    ) -> None:
        self.support = support
        adapter = (
            FeatureAdapter.NONE
            if support.config is None
            else support.config.feature_adapter
        )
        self.backend = adapt_backend(backend, adapter)
        _validate_frame_aligned_spec(semantic_acoustic_spec(self.backend))
        metadata = support.artifact_backend_metadata
        if metadata is None:
            validate_support_metadata(support_metadata(support), self.backend)
        else:
            validate_backend_metadata(metadata, self.backend)

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
        prepared, frame_mask = self.support._semantic_input(semantic_codes, mask)
        return self._sample_features(
            prepared,
            frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            cfg_scale=cfg_scale,
            generator=generator,
        )

    def _sample_features(
        self,
        prepared: Tensor,
        frame_mask: Tensor,
        *,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
        cfg_scale: float | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        if self.support.route is Route.FM:
            return self.support._sample_features(
                prepared,
                frame_mask,
                reference_features=reference_features,
                reference_mask=reference_mask,
                cfg_scale=cfg_scale,
                generator=generator,
            )
        codes = self.support._sample_acoustic_codes(
            prepared,
            frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            generator=generator,
        )
        features = self.backend.acoustic_codes_to_features(codes)
        return features.masked_fill(~frame_mask.to(device=features.device)[..., None], 0)

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
        features = self._sample_features(
            prepared,
            frame_mask,
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
        if features.size(1) != original_length:
            raise ValueError("frame-aligned features must match the padded semantic length.")
        features = features[:, : prepared.size(1)]
        return self.backend.decode_features(prepared, features)


def build_support(
    config: GeneratorConfig,
    *,
    semantic_codebook: Tensor,
    codec_spec: SemanticAcousticCodecSpec,
    artifact_backend_metadata: Mapping[str, object] | None = None,
) -> GeneratorSupport:
    _validate_frame_aligned_spec(codec_spec)
    _validate_semantic_codebook(semantic_codebook, codec_spec)
    modules = build_route(
        config.route,
        semantic_codebook,
        codec_spec.acoustic_feature_dim,
        codec_spec.acoustic_codebook_sizes,
        condition_dim=config.condition_dim,
        decoder=config.decoder,
        initialization=config.initialization,
        seed=config.seed,
    )
    mean = (
        None
        if config.feature_mean is None
        else torch.tensor(config.feature_mean, dtype=torch.float32)
    )
    std = (
        None
        if config.feature_std is None
        else torch.tensor(config.feature_std, dtype=torch.float32)
    )
    return GeneratorSupport(
        modules,
        codec_spec,
        config=config,
        sampling=config.sampling,
        feature_mean=mean,
        feature_std=std,
        artifact_backend_metadata=artifact_backend_metadata,
    )


def _validate_semantic_codebook(
    semantic_codebook: Tensor,
    codec_spec: SemanticAcousticCodecSpec,
) -> None:
    if not isinstance(codec_spec, SemanticAcousticCodecSpec):
        raise TypeError("codec_spec must be a SemanticAcousticCodecSpec.")
    if len(codec_spec.semantic_codebook_sizes) != 1:
        raise ValueError("generator support requires exactly one semantic codebook.")
    expected = (
        codec_spec.semantic_codebook_sizes[0],
        codec_spec.semantic_embedding_dim,
    )
    if tuple(semantic_codebook.shape) != expected:
        raise ValueError(
            "semantic_codebook must match codec_spec [vocab, embedding]: "
            f"{tuple(semantic_codebook.shape)} != {expected}."
        )


def _validate_frame_aligned_spec(codec_spec: SemanticAcousticCodecSpec) -> None:
    if codec_spec.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
        raise ValueError(
            "semantic-acoustic-generator supports only frame-aligned acoustic units."
        )
    if codec_spec.acoustic_unit_length is not None:
        raise ValueError(
            "frame-aligned semantic-acoustic specs must not set acoustic_unit_length."
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
    result = _finite_number(value, name="cfg_scale")
    if result < 0:
        raise ValueError("cfg_scale must be non-negative.")
    return result


def _integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _feature_values(value: object, *, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple.")
    for item in value:
        _finite_number(item, name=f"{name} values")
