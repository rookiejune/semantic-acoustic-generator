from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodec
from torch import Tensor

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
from semantic_acoustic_codec.runtime.metadata import (
    support_metadata,
    validate_backend_metadata,
)

if TYPE_CHECKING:
    from semantic_acoustic_codec.runtime.semantic import (
        SamplingConfig,
        SemanticCodecSupport,
        SemanticSupportConfig,
    )

SCHEMA_VERSION = 7
CONFIG_NAME = "codec.json"
CHECKPOINT_NAME = "model.ckpt"

__all__ = [
    "AcousticGeneratorArtifact",
    "AcousticGeneratorBackend",
    "AcousticGeneratorSpec",
    "load_artifact",
    "load_generator_artifact",
    "save_artifact",
]


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
        validate_backend_metadata(self.backend_metadata(), backend)

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
    validate_backend_metadata(metadata, backend)
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


def _backend_metadata(
    support: SemanticCodecSupport,
    backend: SemanticAcousticCodec,
) -> dict[str, object]:
    return {
        "name": backend.name,
        "sample_rate": backend.sample_rate,
        "frame_rate": float(backend.frame_rate),
        "semantic_frame_rate": float(backend.semantic_frame_rate),
        **support_metadata(support),
    }

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
    _validate_metadata(data, expected)


def _load_artifact(
    path: str | Path,
    *,
    device: str | torch.device | None,
) -> tuple[SemanticCodecSupport, SemanticSupportConfig]:
    from semantic_acoustic_codec.runtime.semantic import build_support

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


def _artifact(path: str | Path) -> tuple[Path, Mapping[str, object], SemanticSupportConfig]:
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
    from semantic_acoustic_codec.runtime.semantic import SamplingConfig, SemanticSupportConfig

    decoder = cast(Mapping[str, Any], data["decoder"])
    sampling = cast(Mapping[str, Any], data["sampling"])
    return SemanticSupportConfig(
        route=Route(cast(str, data["route"])),
        condition_dim=int(data["condition_dim"]),
        decoder=DecoderConfig(
            hidden_dim=cast(int | None, decoder["hidden_dim"]),
            layers=int(decoder["layers"]),
            heads=int(decoder["heads"]),
            ffn_ratio=int(decoder["ffn_ratio"]),
            rvq_predictor=RVQPredictor(
                cast(str, _schema_field(decoder, "rvq_predictor", owner="decoder"))
            ),
            mtp_layers=int(_schema_field(decoder, "mtp_layers", owner="decoder")),
            mtp_heads=int(_schema_field(decoder, "mtp_heads", owner="decoder")),
            repa_feature_dim=cast(
                int | None,
                _schema_field(decoder, "repa_feature_dim", owner="decoder"),
            ),
            repa_student_layer=cast(
                int | None,
                _schema_field(decoder, "repa_student_layer", owner="decoder"),
            ),
            repa_loss_weight=float(
                _schema_field(decoder, "repa_loss_weight", owner="decoder")
            ),
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


def _schema_field(data: Mapping[str, Any], key: str, *, owner: str) -> Any:
    if key not in data:
        raise ValueError(
            f"semantic codec schema {SCHEMA_VERSION} {owner} is missing {key!r}."
        )
    return data[key]


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

def _validate_metadata(
    data: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
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
