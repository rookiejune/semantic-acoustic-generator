from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch
from anytrain.codec import (
    AcousticLayout,
    SemanticAcousticCodec,
    SemanticAcousticCodecSpec,
)
from torch import Tensor

from semantic_acoustic_generator.backend import adapt_backend
from semantic_acoustic_generator.config import (
    AnchorContext,
    AnchorTarget,
    BackboneConfig,
    FactorPredictor,
    FeatureAdapter,
    FMMode,
    HeadConfig,
    Initialization,
    Route,
)
from semantic_acoustic_generator.model.code import RVQCodeGenerator
from semantic_acoustic_generator.model.feature import FMFeatureGenerator, factor_codebook_names
from semantic_acoustic_generator.model.generator import AcousticHead
from semantic_acoustic_generator.model.backbone import QwenBackbone
from semantic_acoustic_generator.model.model import AcousticGeneratorModel
from semantic_acoustic_generator.runtime.metadata import (
    support_metadata,
    validate_backend_metadata,
)

if TYPE_CHECKING:
    from semantic_acoustic_generator.runtime.semantic import (
        GeneratorConfig,
        GeneratorSupport,
        SamplingConfig,
    )

SCHEMA_VERSION = 9
CONFIG_NAME = "generator.json"
LEGACY_SCHEMA_VERSION = 7
LEGACY_CONFIG_NAME = "codec.json"
CHECKPOINT_NAME = "model.ckpt"

__all__ = [
    "AcousticGeneratorArtifact",
    "AcousticGeneratorBackend",
    "AcousticGeneratorSpec",
    "AcousticModelArtifact",
    "load_artifact",
    "load_generator_artifact",
    "load_model_artifact",
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
    backbone: BackboneConfig
    feature_adapter: FeatureAdapter
    feature_codebooks: int
    head: HeadConfig
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

    @property
    def condition_dim(self) -> int:
        return self.backbone.hidden_dim

    @property
    def decoder(self) -> HeadConfig:
        return self.head

    def __post_init__(self) -> None:
        if self.feature_codebooks <= 0:
            raise ValueError("artifact feature_codebooks must be positive.")
        if self.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
            raise ValueError(
                "acoustic generator artifacts support only frame-aligned units."
            )
        if self.acoustic_unit_length is not None:
            raise ValueError(
                "frame-aligned acoustic generator artifacts must not set "
                "acoustic_unit_length."
            )

    def validate_backend(self, backend: SemanticAcousticCodec) -> None:
        validate_backend_metadata(
            self.backend_metadata(),
            adapt_backend(
                backend,
                self.feature_adapter,
                codebooks=self.feature_codebooks,
            ),
        )

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

    generator: AcousticHead
    spec: AcousticGeneratorSpec


@dataclass(frozen=True)
class AcousticModelArtifact:
    """Loaded semantic backbone and acoustic head with their codec contract."""

    model: AcousticGeneratorModel
    spec: AcousticGeneratorSpec

    @property
    def backbone(self) -> QwenBackbone:
        return self.model.backbone

    @property
    def head(self) -> AcousticHead:
        return self.model.head


def save_artifact(
    path: str | Path,
    support: GeneratorSupport,
    *,
    backend: SemanticAcousticCodec,
) -> None:
    artifact_config = support.config
    if artifact_config is None:
        raise ValueError("artifact support must expose its construction config.")
    backend = adapt_backend(
        backend,
        artifact_config.feature_adapter,
        codebooks=artifact_config.feature_codebooks,
    )
    _validate_support_runtime_config(support, artifact_config)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    metadata = _backend_metadata(support, backend)
    validate_backend_metadata(metadata, backend)
    torch.save(support.state_dict(), root / CHECKPOINT_NAME)
    data = {
        "schema_version": SCHEMA_VERSION,
        "config": _config_dict(artifact_config),
        "backend": metadata,
        "checkpoint": CHECKPOINT_NAME,
    }
    (root / CONFIG_NAME).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_artifact(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> GeneratorSupport:
    support, _ = _load_artifact(path, device=device)
    return support


def load_generator_artifact(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> AcousticGeneratorArtifact:
    """Load the reusable acoustic generator and its strict input contract."""
    checkpoint, metadata, config = _artifact(path)
    state = _load_state(checkpoint)
    generator_state = _generator_state(state)
    generator = _generator(
        config,
        metadata,
        factor_codebooks=_factor_codebooks(generator_state, config, prefix=""),
    )
    generator.load_state_dict(generator_state)
    if device is not None:
        generator.to(device=device)
    generator.eval()
    codec_spec = _codec_spec(metadata)
    return AcousticGeneratorArtifact(
        generator=generator,
        spec=_spec(config, metadata, codec_spec),
    )


def load_model_artifact(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> AcousticModelArtifact:
    """Load the schema-9 ``backbone + head`` model without a codec instance."""
    checkpoint, metadata, config = _artifact(path)
    state = _load_state(checkpoint)
    model_state = _model_state(state)
    codec_spec = _codec_spec(metadata)
    factor_codebooks = _factor_codebooks(model_state, config, prefix="head.")
    model = AcousticGeneratorModel(
        QwenBackbone(
            torch.zeros(
                codec_spec.semantic_codebook_sizes[0],
                codec_spec.semantic_embedding_dim,
            ),
            config.backbone,
        ),
        _generator(
            config,
            metadata,
            factor_codebooks=factor_codebooks,
        ),
    )
    model.load_state_dict(model_state)
    if device is not None:
        model.to(device=device)
    model.eval()
    return AcousticModelArtifact(
        model=model,
        spec=_spec(config, metadata, codec_spec),
    )


def _backend_metadata(
    support: GeneratorSupport,
    backend: SemanticAcousticCodec,
) -> dict[str, object]:
    return {
        "name": backend.name,
        "sample_rate": backend.sample_rate,
        "frame_rate": float(backend.frame_rate),
        "semantic_frame_rate": float(backend.semantic_frame_rate),
        **support_metadata(support),
    }


def _validate_support_runtime_config(
    support: GeneratorSupport,
    config: GeneratorConfig,
) -> None:
    if support.route is not config.route:
        raise ValueError("support route no longer matches its construction config.")
    if support.sampling != config.sampling:
        raise ValueError("support sampling no longer matches its construction config.")
    expected_mean = _feature_values(config.feature_mean, support.acoustic_feature_dim, fill=0.0)
    expected_std = _feature_values(config.feature_std, support.acoustic_feature_dim, fill=1.0)
    mean = support.feature_mean.detach()
    std = support.feature_std.detach()
    expected_mean_tensor = mean.new_tensor(expected_mean).view_as(mean)
    expected_std_tensor = std.new_tensor(expected_std).view_as(std)
    if not torch.equal(mean, expected_mean_tensor) or not torch.equal(std, expected_std_tensor):
        raise ValueError("support feature normalization no longer matches its construction config.")


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
) -> tuple[GeneratorSupport, GeneratorConfig]:
    from semantic_acoustic_generator.runtime.semantic import build_support

    checkpoint, backend_data, config = _artifact(path)
    codec_spec = _codec_spec(backend_data)
    state = _load_state(checkpoint)
    support = build_support(
        config,
        semantic_codebook=torch.zeros(
            codec_spec.semantic_codebook_sizes[0],
            codec_spec.semantic_embedding_dim,
        ),
        codec_spec=codec_spec,
        artifact_backend_metadata=backend_data,
        factor_codebooks=_factor_codebooks(
            state,
            config,
            prefix=(
                "model.head."
                if any(key.startswith("model.head.") for key in state)
                else "generator."
            ),
        ),
    )
    support.load_state_dict(state)
    if device is not None:
        support.to(device=device)
    support.eval()
    return support, config


def _artifact(path: str | Path) -> tuple[Path, Mapping[str, object], GeneratorConfig]:
    root = Path(path)
    config_path, expected_schema = _manifest(root)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("semantic-acoustic generator manifest must contain a mapping.")
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != expected_schema
    ):
        raise ValueError(
            f"unsupported semantic-acoustic generator schema in {config_path.name}: "
            f"{schema_version!r}"
        )
    backend = data.get("backend")
    if not isinstance(backend, Mapping):
        raise TypeError("generator backend metadata must be a mapping.")
    raw_config = data.get("config")
    if not isinstance(raw_config, Mapping):
        raise TypeError("generator config metadata must be a mapping.")
    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise TypeError("generator checkpoint must be a non-empty string.")
    return root / checkpoint, cast(Mapping[str, object], backend), _config(raw_config)


def _manifest(root: Path) -> tuple[Path, int]:
    current = root / CONFIG_NAME
    if current.is_file():
        return current, SCHEMA_VERSION
    legacy = root / LEGACY_CONFIG_NAME
    if legacy.is_file():
        return legacy, LEGACY_SCHEMA_VERSION
    raise FileNotFoundError(
        f"generator artifact manifest not found; expected {current.name!r} "
        f"or legacy {legacy.name!r} in {root}."
    )


def _generator(
    config: GeneratorConfig,
    metadata: Mapping[str, object],
    *,
    factor_codebooks: tuple[Tensor, ...] | None = None,
) -> AcousticHead:
    codec_spec = _codec_spec(metadata)
    if config.route is Route.FM:
        return FMFeatureGenerator(
            config.backbone.hidden_dim,
            codec_spec.acoustic_feature_dim,
            config.head,
            factor_codebooks=factor_codebooks,
        )
    if config.route is Route.RVQ:
        return RVQCodeGenerator(
            config.backbone.hidden_dim,
            codec_spec.acoustic_codebook_sizes,
            config.head,
            factor_codebooks=factor_codebooks,
        )
    raise AssertionError(f"unsupported route: {config.route}")


def _spec(
    config: GeneratorConfig,
    metadata: Mapping[str, object],
    codec_spec: SemanticAcousticCodecSpec,
) -> AcousticGeneratorSpec:
    acoustic_feature_dim = codec_spec.acoustic_feature_dim
    return AcousticGeneratorSpec(
        route=config.route,
        backbone=config.backbone,
        feature_adapter=config.feature_adapter,
        feature_codebooks=config.feature_codebooks,
        head=config.head,
        backend_name=_metadata_string(metadata, "name"),
        sample_rate=codec_spec.sample_rate,
        frame_rate=codec_spec.frame_rate,
        semantic_frame_rate=codec_spec.semantic_frame_rate,
        semantic_vocab_size=codec_spec.semantic_codebook_sizes[0],
        semantic_embedding_dim=codec_spec.semantic_embedding_dim,
        acoustic_feature_dim=acoustic_feature_dim,
        acoustic_codebook_sizes=codec_spec.acoustic_codebook_sizes,
        acoustic_layout=codec_spec.acoustic_layout,
        acoustic_unit_length=codec_spec.acoustic_unit_length,
        feature_mean=_feature_values(config.feature_mean, acoustic_feature_dim, fill=0.0),
        feature_std=_feature_values(config.feature_std, acoustic_feature_dim, fill=1.0),
        sampling=config.sampling,
    )


def _codec_spec(metadata: Mapping[str, object]) -> SemanticAcousticCodecSpec:
    spec = SemanticAcousticCodecSpec(
        sample_rate=_metadata_int(metadata, "sample_rate"),
        frame_rate=_metadata_float(metadata, "frame_rate"),
        semantic_frame_rate=_metadata_float(metadata, "semantic_frame_rate"),
        semantic_codebook_sizes=(_metadata_int(metadata, "semantic_vocab_size"),),
        semantic_embedding_dim=_metadata_int(metadata, "semantic_embedding_dim"),
        acoustic_codebook_sizes=_metadata_sizes(metadata),
        acoustic_feature_dim=_metadata_int(metadata, "acoustic_feature_dim"),
        acoustic_layout=AcousticLayout(_metadata_string(metadata, "acoustic_layout")),
        acoustic_unit_length=_metadata_optional_int(metadata, "acoustic_unit_length"),
    )
    if spec.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
        raise ValueError(
            "semantic-acoustic generator artifacts require frame-aligned acoustic units."
        )
    if spec.acoustic_unit_length is not None:
        raise ValueError(
            "frame-aligned generator artifacts must not set acoustic_unit_length."
        )
    return spec


def _generator_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    prefixes = ("model.head.", "generator.")
    result: dict[str, Tensor] = {}
    for prefix in prefixes:
        result = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        if result:
            break
    if not result:
        raise RuntimeError("generator checkpoint is missing generator state.")
    return result


def _model_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    prefix = "model."
    result = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not result:
        raise RuntimeError(
            "generator checkpoint is missing schema-9 model state; "
            "legacy artifacts expose only the acoustic head."
        )
    return result


def _factor_codebooks(
    state: Mapping[str, Tensor],
    config: GeneratorConfig,
    *,
    prefix: str,
) -> tuple[Tensor, ...] | None:
    if config.route is Route.RVQ:
        values: list[Tensor] = []
        index = 0
        while (key := f"{prefix}core.classifiers.{index}.codebook") in state:
            values.append(state[key])
            index += 1
        if values and len(values) % 2 != 0:
            raise RuntimeError("AGRVQ artifact contains an incomplete factor-codebook pair.")
        return tuple(values) if values else None
    if config.head.anchor_target is AnchorTarget.FEATURE:
        return None
    keys = tuple(
        f"{prefix}{name}" for name in factor_codebook_names(config.feature_codebooks)
    )
    if any(key not in state for key in keys):
        raise RuntimeError("factor-target artifact is missing stored factor codebooks.")
    return tuple(state[key] for key in keys)


def _load_state(path: Path) -> Mapping[str, Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("generator checkpoint must contain a state dict mapping.")
    return cast(Mapping[str, Tensor], state)


def _config_dict(config: GeneratorConfig) -> dict[str, object]:
    data = asdict(config)
    data["route"] = config.route.value
    data["feature_adapter"] = config.feature_adapter.value
    backbone = cast(dict[str, object], data["backbone"])
    backbone["embedding_initialization"] = config.backbone.embedding_initialization.value
    head = cast(dict[str, object], data["head"])
    head["codebook_initialization"] = config.head.codebook_initialization.value
    head["fm_mode"] = config.head.fm_mode.value
    head["anchor_context"] = config.head.anchor_context.value
    head["anchor_target"] = config.head.anchor_target.value
    head["factor_predictor"] = config.head.factor_predictor.value
    return cast(dict[str, object], data)


def _config(data: Mapping[str, object]) -> GeneratorConfig:
    from semantic_acoustic_generator.runtime.semantic import GeneratorConfig, SamplingConfig

    backbone = (
        _schema_mapping(data, "backbone", owner="config")
        if "backbone" in data
        else None
    )
    head = _schema_mapping(data, "head", owner="config") if "head" in data else None
    head_data = _schema_mapping(data, "decoder", owner="config") if head is None else head
    head_owner = "decoder" if head is None else "head"
    sampling = _schema_mapping(data, "sampling", owner="config")
    return GeneratorConfig(
        route=Route(_schema_string(data, "route", owner="config")),
        backbone=BackboneConfig(
            hidden_dim=(
                _schema_int(data, "condition_dim", owner="config")
                if backbone is None
                else _schema_int(backbone, "hidden_dim", owner="backbone")
            ),
            layers=(
                4
                if backbone is None
                else _schema_int(backbone, "layers", owner="backbone")
            ),
            heads=(
                8
                if backbone is None
                else _schema_int(backbone, "heads", owner="backbone")
            ),
            ffn_ratio=(
                4
                if backbone is None
                else _schema_int(backbone, "ffn_ratio", owner="backbone")
            ),
            embedding_initialization=Initialization(
                _schema_string(data, "initialization", owner="config")
                if backbone is None
                else _schema_string(
                    backbone,
                    "embedding_initialization",
                    owner="backbone",
                )
            ),
            seed=(
                _schema_int(data, "seed", owner="config")
                if backbone is None
                else _schema_int(backbone, "seed", owner="backbone")
            ),
        ),
        feature_adapter=FeatureAdapter(
            _schema_optional_string(
                data,
                "feature_adapter",
                owner="config",
                default=FeatureAdapter.NONE.value,
            )
        ),
        feature_codebooks=_schema_optional_int_default(
            data,
            "feature_codebooks",
            owner="config",
            default=1,
        ),
        head=HeadConfig(
            codebook_initialization=Initialization(
                _schema_optional_string(
                    head_data,
                    (
                        "codebook_initialization"
                        if "codebook_initialization" in head_data
                        else "initialization"
                    ),
                    owner=head_owner,
                    default=Initialization.CODEC.value,
                )
            ),
            seed=_schema_optional_int_default(
                head_data,
                "seed",
                owner=head_owner,
                default=0,
            ),
            hidden_dim=_schema_optional_int(head_data, "hidden_dim", owner=head_owner),
            layers=_schema_int(head_data, "layers", owner=head_owner),
            heads=_schema_int(head_data, "heads", owner=head_owner),
            ffn_ratio=_schema_int(head_data, "ffn_ratio", owner=head_owner),
            repa_feature_dim=_schema_optional_int(
                head_data,
                "repa_feature_dim",
                owner=head_owner,
            ),
            repa_student_layer=_schema_optional_int(
                head_data,
                "repa_student_layer",
                owner=head_owner,
            ),
            repa_loss_weight=_schema_float(
                head_data,
                "repa_loss_weight",
                owner=head_owner,
            ),
            fm_mode=FMMode(
                _schema_optional_string(
                    head_data,
                    "fm_mode",
                    owner=head_owner,
                    default=FMMode.FLOW.value,
                )
            ),
            anchor_context=AnchorContext(
                _schema_optional_string(
                    head_data,
                    "anchor_context",
                    owner=head_owner,
                    default=AnchorContext.LOCAL.value,
                )
            ),
            anchor_target=AnchorTarget(
                _schema_optional_string(
                    head_data,
                    "anchor_target",
                    owner=head_owner,
                    default=AnchorTarget.FEATURE.value,
                )
            ),
            factor_predictor=FactorPredictor(
                _schema_optional_string(
                    head_data,
                    "factor_predictor",
                    owner=head_owner,
                    default=FactorPredictor.PARALLEL.value,
                )
            ),
            anchor_hidden_dim=_schema_optional_int_default(
                head_data,
                "anchor_hidden_dim",
                owner=head_owner,
                default=512,
            ),
            anchor_layers=_schema_optional_int_default(
                head_data,
                "anchor_layers",
                owner=head_owner,
                default=4,
            ),
            anchor_kernel_size=_schema_optional_int_default(
                head_data,
                "anchor_kernel_size",
                owner=head_owner,
                default=3,
            ),
            anchor_cosine_weight=_schema_optional_float(
                head_data,
                "anchor_cosine_weight",
                owner=head_owner,
                default=0.1,
            ),
            anchor_factor_weight=_schema_optional_float(
                head_data,
                "anchor_factor_weight",
                owner=head_owner,
                default=0.1,
            ),
            anchor_factor_temperature=_schema_optional_float(
                head_data,
                "anchor_factor_temperature",
                owner=head_owner,
                default=0.1,
            ),
        ),
        sampling=SamplingConfig(
            flow_steps=_schema_int(sampling, "flow_steps", owner="sampling"),
            temperature=_schema_float(sampling, "temperature", owner="sampling"),
            top_p=_schema_float(sampling, "top_p", owner="sampling"),
            cfg_scale=_schema_float(sampling, "cfg_scale", owner="sampling"),
        ),
        feature_mean=_float_tuple(
            _schema_field(data, "feature_mean", owner="config"),
            name="feature_mean",
        ),
        feature_std=_float_tuple(
            _schema_field(data, "feature_std", owner="config"),
            name="feature_std",
        ),
    )


def _schema_field(data: Mapping[str, object], key: str, *, owner: str) -> object:
    if key not in data:
        raise ValueError(f"generator artifact {owner} is missing {key!r}.")
    return data[key]


def _schema_mapping(
    data: Mapping[str, object],
    key: str,
    *,
    owner: str,
) -> Mapping[str, object]:
    value = _schema_field(data, key, owner=owner)
    if not isinstance(value, Mapping):
        raise TypeError(f"generator artifact {owner}.{key} must be a mapping.")
    return cast(Mapping[str, object], value)


def _schema_string(data: Mapping[str, object], key: str, *, owner: str) -> str:
    value = _schema_field(data, key, owner=owner)
    if not isinstance(value, str):
        raise TypeError(f"generator artifact {owner}.{key} must be a string.")
    return value


def _schema_optional_string(
    data: Mapping[str, object],
    key: str,
    *,
    owner: str,
    default: str,
) -> str:
    if key not in data:
        return default
    return _schema_string(data, key, owner=owner)


def _schema_optional_int_default(
    data: Mapping[str, object],
    key: str,
    *,
    owner: str,
    default: int,
) -> int:
    if key not in data:
        return default
    return _schema_int(data, key, owner=owner)


def _schema_optional_float(
    data: Mapping[str, object],
    key: str,
    *,
    owner: str,
    default: float,
) -> float:
    if key not in data:
        return default
    return _schema_float(data, key, owner=owner)


def _schema_int(data: Mapping[str, object], key: str, *, owner: str) -> int:
    value = _schema_field(data, key, owner=owner)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"generator artifact {owner}.{key} must be an integer.")
    return value


def _schema_optional_int(
    data: Mapping[str, object],
    key: str,
    *,
    owner: str,
) -> int | None:
    value = _schema_field(data, key, owner=owner)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"generator artifact {owner}.{key} must be an integer or null.")
    return value


def _schema_float(data: Mapping[str, object], key: str, *, owner: str) -> float:
    value = _schema_field(data, key, owner=owner)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"generator artifact {owner}.{key} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"generator artifact {owner}.{key} must be finite.")
    return result


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


def _float_tuple(value: object, *, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{name} metadata must be a list or null.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{name} metadata must contain only numbers.")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} metadata values must be finite.")
    return result


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
    if value <= 0:
        raise ValueError(f"artifact backend metadata {key!r} must be positive.")
    return value


def _metadata_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"artifact backend metadata {key!r} must be a number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"artifact backend metadata {key!r} must be positive.")
    return result


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
