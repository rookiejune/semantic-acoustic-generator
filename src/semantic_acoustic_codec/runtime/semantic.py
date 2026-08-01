from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, cast, runtime_checkable

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodec
from torch import Tensor, nn

from semantic_acoustic_codec._tensor import is_signed_integer_dtype
from semantic_acoustic_codec.config import (
    DecoderConfig,
    Initialization,
    Route,
    RVQPredictor,
)
from semantic_acoustic_codec.model.decoder import (
    CodecUnitGenerator,
    FMFeatureGenerator,
    RVQCodeGenerator,
)
from semantic_acoustic_codec.model.routes import RouteModules, build_route

SCHEMA_VERSION = 7
CONFIG_NAME = "codec.json"
CHECKPOINT_NAME = "model.ckpt"


@dataclass(frozen=True)
class SamplingConfig:
    flow_steps: int = 16
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.flow_steps < 1:
            raise ValueError("flow_steps must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")


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


@runtime_checkable
class AcousticGeneratorBackend(Protocol):
    """Backend metadata required when reusing only an acoustic generator."""

    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_layout(self) -> AcousticLayout: ...

    @property
    def acoustic_unit_length(self) -> int | None: ...


@dataclass(frozen=True)
class AcousticGeneratorSpec:
    """Portable decoder contract consumed by external condition producers."""

    route: Route
    condition_dim: int
    decoder: DecoderConfig
    backend_name: str
    sample_rate: int
    frame_rate: float
    semantic_frame_rate: float
    semantic_vocab_size: int
    semantic_embedding_dim: int
    acoustic_feature_dim: int
    acoustic_codebook_sizes: tuple[int, ...]
    acoustic_layout: AcousticLayout
    acoustic_unit_length: int | None
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    sampling: SamplingConfig

    def validate_backend(self, backend: SemanticAcousticCodec) -> None:
        _validate_backend_metadata(self.backend_metadata(), backend)

    def validate_acoustic_backend(self, backend: AcousticGeneratorBackend) -> None:
        _validate_acoustic_backend_metadata(self.acoustic_backend_metadata(), backend)

    def backend_metadata(self) -> dict[str, object]:
        return {
            "name": self.backend_name,
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "semantic_frame_rate": self.semantic_frame_rate,
            "semantic_vocab_size": self.semantic_vocab_size,
            "semantic_embedding_dim": self.semantic_embedding_dim,
            **self.acoustic_backend_metadata(),
        }

    def acoustic_backend_metadata(self) -> dict[str, object]:
        return {
            "acoustic_feature_dim": self.acoustic_feature_dim,
            "acoustic_codebook_sizes": list(self.acoustic_codebook_sizes),
            "acoustic_layout": self.acoustic_layout.value,
            "acoustic_unit_length": self.acoustic_unit_length,
        }


@dataclass(frozen=True)
class AcousticGeneratorArtifact:
    """Loaded generator plus the condition and backend contract of its weights."""

    generator: CodecUnitGenerator
    spec: AcousticGeneratorSpec


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
            _validate_support_metadata(_support_metadata(support), backend)
        else:
            _validate_backend_metadata(metadata, backend)

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
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self.support._semantic_input(semantic_codes, mask)
        prepared, frame_mask = _trim_decode_input(prepared, frame_mask)
        features = self.sample_features(
            prepared,
            mask=frame_mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
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


def save_artifact(
    path: str | Path,
    support: SemanticCodecSupport,
    config: SemanticSupportConfig,
    *,
    backend: SemanticAcousticCodec,
) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if config.route is not support.route:
        raise ValueError("artifact config route must match support route.")
    metadata = _backend_metadata(support, backend)
    _validate_backend_metadata(metadata, backend)
    torch.save(support.state_dict(), root / CHECKPOINT_NAME)
    data = {
        "schema_version": SCHEMA_VERSION,
        "config": _config_dict(config),
        "backend": metadata,
        "checkpoint": CHECKPOINT_NAME,
    }
    (root / CONFIG_NAME).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_artifact(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> SemanticCodecSupport:
    support, _ = _load_artifact(path, device=device)
    return support


def load_generator_artifact(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> AcousticGeneratorArtifact:
    """Load the reusable acoustic generator and its strict input contract."""
    checkpoint, metadata, config = _artifact(path)
    generator = _generator(config, metadata)
    if device is not None:
        generator.to(device=device)
    state = _load_state(checkpoint, device=device)
    generator.load_state_dict(_generator_state(state))
    generator.eval()
    acoustic_feature_dim = _metadata_int(metadata, "acoustic_feature_dim")
    spec = AcousticGeneratorSpec(
        route=config.route,
        condition_dim=config.condition_dim,
        decoder=config.decoder,
        backend_name=_metadata_string(metadata, "name"),
        sample_rate=_metadata_int(metadata, "sample_rate"),
        frame_rate=_metadata_float(metadata, "frame_rate"),
        semantic_frame_rate=_metadata_float(metadata, "semantic_frame_rate"),
        semantic_vocab_size=_metadata_int(metadata, "semantic_vocab_size"),
        semantic_embedding_dim=_metadata_int(metadata, "semantic_embedding_dim"),
        acoustic_feature_dim=acoustic_feature_dim,
        acoustic_codebook_sizes=_metadata_sizes(metadata),
        acoustic_layout=AcousticLayout(str(metadata["acoustic_layout"])),
        acoustic_unit_length=_metadata_optional_int(metadata, "acoustic_unit_length"),
        feature_mean=_feature_values(config.feature_mean, acoustic_feature_dim, fill=0.0),
        feature_std=_feature_values(config.feature_std, acoustic_feature_dim, fill=1.0),
        sampling=config.sampling,
    )
    return AcousticGeneratorArtifact(generator=generator, spec=spec)


def _load_artifact(
    path: str | Path,
    *,
    device: str | torch.device | None,
) -> tuple[SemanticCodecSupport, SemanticSupportConfig]:
    checkpoint, backend_data, config = _artifact(path)
    support = build_support(
        config,
        semantic_codebook=torch.zeros(
            _metadata_int(backend_data, "semantic_vocab_size"),
            _metadata_int(backend_data, "semantic_embedding_dim"),
        ),
        acoustic_feature_dim=_metadata_int(backend_data, "acoustic_feature_dim"),
        acoustic_codebook_sizes=_metadata_sizes(backend_data),
        acoustic_layout=AcousticLayout(str(backend_data["acoustic_layout"])),
        acoustic_unit_length=_metadata_optional_int(backend_data, "acoustic_unit_length"),
        artifact_backend_metadata=backend_data,
    )
    state = _load_state(checkpoint, device=device)
    support.load_state_dict(state)
    if device is not None:
        support.to(device=device)
    support.eval()
    return support, config


def _artifact(
    path: str | Path,
) -> tuple[Path, Mapping[str, object], SemanticSupportConfig]:
    root = Path(path)
    data = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("semantic codec config must contain a mapping.")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported semantic codec schema: {data.get('schema_version')!r}")
    backend = data.get("backend")
    if not isinstance(backend, Mapping):
        raise TypeError("semantic codec backend metadata must be a mapping.")
    raw_config = data.get("config")
    if not isinstance(raw_config, Mapping):
        raise TypeError("semantic codec config metadata must be a mapping.")
    checkpoint = data.get("checkpoint", CHECKPOINT_NAME)
    if not isinstance(checkpoint, str) or not checkpoint:
        raise TypeError("semantic codec checkpoint must be a non-empty string.")
    return root / checkpoint, cast(Mapping[str, object], backend), _config(raw_config)


def _generator(
    config: SemanticSupportConfig,
    metadata: Mapping[str, object],
) -> CodecUnitGenerator:
    acoustic_feature_dim = _metadata_int(metadata, "acoustic_feature_dim")
    acoustic_codebook_sizes = _metadata_sizes(metadata)
    layout = AcousticLayout(str(metadata["acoustic_layout"]))
    fixed_length = (
        _metadata_optional_int(metadata, "acoustic_unit_length")
        if layout is AcousticLayout.FIXED_LENGTH
        else None
    )
    if config.route is Route.FM:
        return FMFeatureGenerator(
            config.condition_dim,
            acoustic_feature_dim,
            config.decoder,
            fixed_length=fixed_length,
        )
    if config.route is Route.RVQ:
        return RVQCodeGenerator(
            config.condition_dim,
            acoustic_codebook_sizes,
            config.decoder,
            fixed_length=fixed_length,
        )
    raise AssertionError(f"unsupported route: {config.route}")


def _generator_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    prefix = "generator."
    result = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not result:
        raise RuntimeError("semantic codec checkpoint is missing generator state.")
    return result


def _load_state(path: Path, *, device: str | torch.device | None) -> Mapping[str, Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, Mapping):
        raise TypeError("semantic codec checkpoint must contain a state dict mapping.")
    return cast(Mapping[str, Tensor], state)



def _config_dict(config: SemanticSupportConfig) -> dict[str, object]:
    data = asdict(config)
    data["route"] = config.route.value
    data["initialization"] = config.initialization.value
    decoder = cast(dict[str, object], data["decoder"])
    decoder["rvq_predictor"] = config.decoder.rvq_predictor.value
    return cast(dict[str, object], data)


def _config(data: Mapping[str, Any]) -> SemanticSupportConfig:
    decoder = cast(Mapping[str, Any], data["decoder"])
    sampling = cast(Mapping[str, Any], data["sampling"])
    return SemanticSupportConfig(
        route=Route(cast(str, data["route"])),
        condition_dim=int(data["condition_dim"]),
        decoder=DecoderConfig(
            hidden_dim=cast(Optional[int], decoder["hidden_dim"]),
            layers=int(decoder["layers"]),
            heads=int(decoder["heads"]),
            ffn_ratio=int(decoder["ffn_ratio"]),
            rvq_predictor=RVQPredictor(
                cast(str, decoder.get("rvq_predictor", RVQPredictor.MTP.value))
            ),
            mtp_layers=int(decoder.get("mtp_layers", 2)),
            mtp_heads=int(decoder.get("mtp_heads", 4)),
            repa_feature_dim=cast(Optional[int], decoder.get("repa_feature_dim")),
            repa_student_layer=cast(Optional[int], decoder.get("repa_student_layer")),
            repa_loss_weight=float(decoder.get("repa_loss_weight", 0.0)),
        ),
        initialization=Initialization(cast(str, data["initialization"])),
        seed=int(data["seed"]),
        sampling=SamplingConfig(
            flow_steps=int(sampling["flow_steps"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
        ),
        feature_mean=_float_tuple(data.get("feature_mean")),
        feature_std=_float_tuple(data.get("feature_std")),
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


def _feature_values(
    value: tuple[float, ...] | None,
    acoustic_feature_dim: int,
    *,
    fill: float,
) -> tuple[float, ...]:
    values = (fill,) * acoustic_feature_dim if value is None else value
    if len(values) != acoustic_feature_dim:
        raise ValueError("feature normalization must match backend acoustic_feature_dim.")
    if any(not math.isfinite(item) for item in values):
        raise ValueError("feature normalization values must be finite.")
    return values


def _float_tuple(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("feature normalization metadata must be a list.")
    return tuple(float(item) for item in value)


def _codebook_sizes(support: SemanticCodecSupport) -> tuple[int, ...]:
    return support.acoustic_codebook_sizes


def _support_metadata(support: SemanticCodecSupport) -> dict[str, object]:
    return {
        "semantic_vocab_size": support.conditioner.semantic_codebook_size,
        "semantic_embedding_dim": support.conditioner.embedding.embedding_dim,
        "acoustic_feature_dim": support.acoustic_feature_dim,
        "acoustic_codebook_sizes": list(_codebook_sizes(support)),
        "acoustic_layout": support.acoustic_layout.value,
        "acoustic_unit_length": support.acoustic_unit_length,
    }


def _backend_metadata(
    support: SemanticCodecSupport,
    backend: SemanticAcousticCodec,
) -> dict[str, object]:
    return {
        "name": backend.name,
        "sample_rate": backend.sample_rate,
        "frame_rate": float(backend.frame_rate),
        "semantic_frame_rate": float(backend.semantic_frame_rate),
        **_support_metadata(support),
    }


def _validate_backend_metadata(
    data: Mapping[str, object],
    backend: SemanticAcousticCodec,
) -> None:
    expected = {
        "name": backend.name,
        "sample_rate": backend.sample_rate,
        "frame_rate": float(backend.frame_rate),
        "semantic_frame_rate": float(backend.semantic_frame_rate),
        **_expected_support_metadata(backend),
    }
    _validate_metadata(data, expected)


def _validate_support_metadata(
    data: Mapping[str, object],
    backend: SemanticAcousticCodec,
) -> None:
    _validate_metadata(data, _expected_support_metadata(backend))


def _expected_support_metadata(backend: SemanticAcousticCodec) -> dict[str, object]:
    return {
        "semantic_vocab_size": int(backend.semantic_codebook.size(0)),
        "semantic_embedding_dim": int(backend.semantic_codebook.size(1)),
        "acoustic_feature_dim": backend.acoustic_feature_dim,
        "acoustic_codebook_sizes": list(backend.acoustic_codebook_sizes),
        "acoustic_layout": backend.acoustic_layout.value,
        "acoustic_unit_length": backend.acoustic_unit_length,
    }


def _validate_metadata(
    data: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"backend metadata mismatch for {key}: {data.get(key)!r} != {value!r}")


def _validate_acoustic_backend_metadata(
    data: Mapping[str, object],
    backend: AcousticGeneratorBackend,
) -> None:
    expected = {
        "acoustic_feature_dim": backend.acoustic_feature_dim,
        "acoustic_codebook_sizes": list(backend.acoustic_codebook_sizes),
        "acoustic_layout": backend.acoustic_layout.value,
        "acoustic_unit_length": backend.acoustic_unit_length,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"backend metadata mismatch for {key}: {data.get(key)!r} != {value!r}")


def _metadata_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"artifact backend metadata {key!r} must be an integer.")
    return value


def _metadata_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"artifact backend metadata {key!r} must be a number.")
    if value <= 0:
        raise ValueError(f"artifact backend metadata {key!r} must be positive.")
    return float(value)


def _metadata_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact backend metadata {key!r} must be a non-empty string.")
    return value


def _metadata_sizes(data: Mapping[str, object]) -> tuple[int, ...]:
    value = data.get("acoustic_codebook_sizes")
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise TypeError("artifact acoustic_codebook_sizes must be a list of integers.")
    sizes = tuple(value)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("artifact acoustic_codebook_sizes must contain positive integers.")
    return sizes


def _metadata_optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"artifact backend metadata {key!r} must be an integer or null.")
    if value <= 0:
        raise ValueError(f"artifact backend metadata {key!r} must be positive.")
    return value


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
