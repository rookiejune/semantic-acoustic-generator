from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from semantic_acoustic_codec._tensor import is_signed_integer_dtype
from semantic_acoustic_codec.backend import LongCatBackend
from semantic_acoustic_codec.config import (
    AdapterType,
    DecoderConfig,
    Initialization,
    Route,
    RVQPredictor,
)
from semantic_acoustic_codec.model.routes import RouteModules, build_route
from semantic_acoustic_codec.runtime.protocol import CodecBackend

SCHEMA_VERSION = 2
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
    adapter: AdapterType | None = AdapterType.LINEAR
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
    """Semantic-only support wrapper around a real codec backend and unit generator."""

    def __init__(
        self,
        backend: CodecBackend,
        modules: RouteModules,
        *,
        sampling: SamplingConfig | None = None,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.conditioner = modules.conditioner
        self.reference_conditioner = modules.reference_conditioner
        self.generator = modules.generator
        self.route = modules.route
        self.sampling = SamplingConfig() if sampling is None else sampling
        self.feature_mean = nn.Buffer(_feature_stat(backend, feature_mean, fill=0.0))
        self.feature_std = nn.Buffer(_feature_stat(backend, feature_std, fill=1.0))

    @property
    def sample_rate(self) -> int:
        return self.backend.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.backend.frame_rate

    @property
    def semantic_codebook(self) -> Tensor:
        return self.backend.semantic_codebook

    @torch.no_grad()
    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        codes = self.backend.encode(audio, sample_rate)
        if codes.dim() != 3 or codes.size(-1) < 1:
            raise ValueError("backend encode must return codes with shape [B, F, K].")
        if not is_signed_integer_dtype(codes.dtype):
            raise TypeError("backend encode must return signed integer codes.")
        return codes[..., :1].to(dtype=torch.long).contiguous()

    @torch.no_grad()
    def decode(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        features = self.sample_features(
            prepared,
            mask=frame_mask,
            generator=generator,
        )
        return self.backend.decode_features(prepared, features)

    @torch.no_grad()
    def sample_features(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        condition = self.condition(prepared, mask=frame_mask)
        features = self.generator.sample_features(
            self.backend,
            condition,
            frame_mask,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            flow_steps=self.sampling.flow_steps,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            generator=generator,
        )
        return features.masked_fill(~frame_mask[..., None], 0)

    @torch.no_grad()
    def sample_acoustic_codes(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        condition = self.condition(prepared, mask=frame_mask)
        return self.generator.sample_acoustic_codes(
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
        reference_acoustic_codes: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        semantic = self.conditioner(prepared)
        reference = self._reference_condition(
            batch_size=prepared.size(0),
            reference_acoustic_codes=reference_acoustic_codes,
            reference_features=reference_features,
            reference_mask=reference_mask,
        )
        return (semantic + reference).masked_fill(~frame_mask[..., None], 0)

    @torch.no_grad()
    def decode_features(self, semantic_codes: Tensor, features: Tensor) -> Tensor:
        prepared, _ = self._semantic_input(semantic_codes, None)
        if features.shape[:2] != prepared.shape[:2] or features.dim() != 3:
            raise ValueError("features must have shape [B, F, D] and align with semantic codes.")
        if features.size(-1) != self.backend.acoustic_feature_dim:
            raise ValueError("features must match backend acoustic_feature_dim.")
        return self.backend.decode_features(prepared, features)

    def _reference_condition(
        self,
        *,
        batch_size: int,
        reference_acoustic_codes: Tensor | None,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
    ) -> Tensor:
        if reference_acoustic_codes is not None and reference_features is not None:
            raise ValueError("provide reference_acoustic_codes or reference_features, not both.")
        features = reference_features
        if reference_acoustic_codes is not None:
            if reference_acoustic_codes.dim() != 3 or reference_acoustic_codes.size(-1) < 1:
                raise ValueError("reference_acoustic_codes must have shape [B, F, K].")
            if not is_signed_integer_dtype(reference_acoustic_codes.dtype):
                raise TypeError("reference_acoustic_codes must use a signed integer dtype.")
            if reference_mask is None:
                reference_mask = torch.ones(
                    reference_acoustic_codes.shape[:2],
                    device=reference_acoustic_codes.device,
                    dtype=torch.bool,
                )
            elif reference_mask.shape != reference_acoustic_codes.shape[:2]:
                raise ValueError("reference_mask must align with reference_acoustic_codes.")
            elif reference_mask.dtype != torch.bool:
                raise TypeError("reference_mask must be boolean.")
            safe_codes = reference_acoustic_codes.masked_fill(~reference_mask[..., None], 0)
            with torch.no_grad():
                features = self.backend.acoustic_codes_to_features(safe_codes)
        elif features is not None and reference_mask is not None:
            if reference_mask.shape != features.shape[:2]:
                raise ValueError("reference_mask must align with reference_features.")
            if reference_mask.dtype != torch.bool:
                raise TypeError("reference_mask must be boolean.")

        parameter = self.reference_conditioner.default_feature
        if features is not None:
            features = features.to(device=parameter.device, dtype=parameter.dtype)
        if reference_mask is not None:
            reference_mask = reference_mask.to(device=parameter.device)
        return self.reference_conditioner(
            features,
            mask=reference_mask,
            batch_size=batch_size,
        )

    def _semantic_input(
        self,
        value: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if value.dim() == 2:
            semantic = value[:, :, None]
        elif value.dim() == 3 and value.size(-1) == 1:
            semantic = value
        else:
            raise ValueError("semantic_codes must have shape [B, F] or [B, F, 1].")
        if semantic.size(0) < 1 or semantic.size(1) < 1:
            raise ValueError("semantic_codes must not be empty.")
        if not is_signed_integer_dtype(semantic.dtype):
            raise TypeError("semantic_codes must use a signed integer dtype.")

        reference = self.conditioner.embedding.weight
        prepared = semantic.to(device=reference.device, dtype=torch.long).contiguous()
        if mask is None:
            frame_mask = torch.ones(prepared.shape[:2], device=prepared.device, dtype=torch.bool)
        else:
            if mask.shape != prepared.shape[:2]:
                raise ValueError("mask must align with semantic_codes on [B, F].")
            if mask.dtype != torch.bool:
                raise TypeError("mask must be boolean.")
            frame_mask = mask.to(device=prepared.device)
        if not bool(frame_mask.any(dim=1).all()):
            raise ValueError("each semantic sequence must contain at least one valid frame.")

        valid = prepared[..., 0][frame_mask]
        if bool((valid < 0).any()):
            raise ValueError("valid semantic_codes must not contain negative IDs.")
        if bool((valid >= self.conditioner.embedding.num_embeddings).any()):
            raise ValueError("semantic_codes contain an ID outside the semantic codebook.")
        return prepared.masked_fill(~frame_mask[..., None], 0), frame_mask


def build_support(backend: CodecBackend, config: SemanticSupportConfig) -> SemanticCodecSupport:
    modules = build_route(
        config.route,
        backend,
        condition_dim=config.condition_dim,
        decoder=config.decoder,
        adapter=config.adapter,
        initialization=config.initialization,
        seed=config.seed,
    )
    mean = None if config.feature_mean is None else torch.tensor(config.feature_mean, dtype=torch.float32)
    std = None if config.feature_std is None else torch.tensor(config.feature_std, dtype=torch.float32)
    return SemanticCodecSupport(
        backend,
        modules,
        sampling=config.sampling,
        feature_mean=mean,
        feature_std=std,
    )


def save_artifact(path: str | Path, support: SemanticCodecSupport, config: SemanticSupportConfig) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if config.route is not support.route:
        raise ValueError("artifact config route must match support route.")
    torch.save(support.state_dict(), root / CHECKPOINT_NAME)
    data = {
        "schema_version": SCHEMA_VERSION,
        "config": _config_dict(config),
        "backend": {
            "name": support.backend.name,
            "sample_rate": support.sample_rate,
            "frame_rate": support.frame_rate,
            "semantic_vocab_size": int(support.semantic_codebook.size(0)),
            "acoustic_feature_dim": int(support.backend.acoustic_feature_dim),
            "acoustic_codebook_sizes": list(support.backend.acoustic_codebook_sizes),
        },
        "checkpoint": CHECKPOINT_NAME,
    }
    (root / CONFIG_NAME).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_artifact(
    path: str | Path,
    *,
    backend: CodecBackend | None = None,
    device: str | torch.device | None = None,
) -> SemanticCodecSupport:
    root = Path(path)
    data = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported semantic codec schema: {data.get('schema_version')!r}")
    backend_data = cast(Mapping[str, object], data["backend"])
    backend = _load_backend(backend_data, device=device) if backend is None else backend
    config = _config(data["config"])
    _validate_backend_metadata(backend_data, backend)
    support = build_support(backend, config)
    state = _load_state(root / str(data.get("checkpoint", CHECKPOINT_NAME)), device=device)
    support.load_state_dict(state)
    if device is not None:
        support.to(device=device)
    support.eval()
    return support


def _load_state(path: Path, *, device: str | torch.device | None) -> Mapping[str, Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, Mapping):
        raise TypeError("semantic codec checkpoint must contain a state dict mapping.")
    return cast(Mapping[str, Tensor], state)


def _load_backend(data: Mapping[str, object], *, device: str | torch.device | None) -> CodecBackend:
    name = data.get("name")
    if name == LongCatBackend.name:
        return LongCatBackend.from_pretrained(device=None if device is None else str(device))
    raise ValueError(f"unsupported codec backend: {name!r}")


def _config_dict(config: SemanticSupportConfig) -> dict[str, object]:
    data = asdict(config)
    data["route"] = config.route.value
    data["adapter"] = None if config.adapter is None else config.adapter.value
    data["initialization"] = config.initialization.value
    decoder = cast(dict[str, object], data["decoder"])
    decoder["rvq_predictor"] = config.decoder.rvq_predictor.value
    return cast(dict[str, object], data)


def _config(data: Mapping[str, Any]) -> SemanticSupportConfig:
    decoder = cast(Mapping[str, Any], data["decoder"])
    sampling = cast(Mapping[str, Any], data["sampling"])
    adapter = data.get("adapter")
    return SemanticSupportConfig(
        route=Route(cast(str, data["route"])),
        condition_dim=int(data["condition_dim"]),
        decoder=DecoderConfig(
            hidden_dim=cast(int | None, decoder["hidden_dim"]),
            layers=int(decoder["layers"]),
            heads=int(decoder["heads"]),
            ffn_ratio=int(decoder["ffn_ratio"]),
            rvq_predictor=RVQPredictor(cast(str, decoder.get("rvq_predictor", RVQPredictor.CODEBOOK_AR.value))),
            mtp_layers=int(decoder.get("mtp_layers", 2)),
            mtp_heads=int(decoder.get("mtp_heads", 4)),
            repa_feature_dim=cast(int | None, decoder.get("repa_feature_dim")),
            repa_student_layer=cast(int | None, decoder.get("repa_student_layer")),
            repa_loss_weight=float(decoder.get("repa_loss_weight", 0.0)),
        ),
        adapter=None if adapter is None else AdapterType(cast(str, adapter)),
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


def _validate_backend_metadata(data: Mapping[str, object], backend: CodecBackend) -> None:
    expected = {
        "name": backend.name,
        "sample_rate": backend.sample_rate,
        "frame_rate": backend.frame_rate,
        "semantic_vocab_size": int(backend.semantic_codebook.size(0)),
        "acoustic_feature_dim": backend.acoustic_feature_dim,
        "acoustic_codebook_sizes": list(backend.acoustic_codebook_sizes),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"backend metadata mismatch for {key}: {data.get(key)!r} != {value!r}")


def _feature_stat(backend: CodecBackend, value: Tensor | None, *, fill: float) -> Tensor:
    if value is None:
        return torch.full((1, 1, backend.acoustic_feature_dim), fill)
    if value.dim() == 1:
        value = value.view(1, 1, -1)
    if value.shape != (1, 1, backend.acoustic_feature_dim):
        raise ValueError("feature normalization must match backend acoustic_feature_dim.")
    if not torch.is_floating_point(value):
        raise TypeError("feature normalization tensors must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("feature normalization tensors must be finite.")
    return value.detach().clone()


def _float_tuple(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("feature normalization metadata must be a list.")
    return tuple(float(item) for item in value)
