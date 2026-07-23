from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from semantic_acoustic_codec._tensor import is_signed_integer_dtype
from semantic_acoustic_codec.config import AdapterType, DecoderConfig, Initialization, Route
from semantic_acoustic_codec.model.dit import DiTDecoder
from semantic_acoustic_codec.model.routes import RouteModules, build_route
from semantic_acoustic_codec.model.rvq import AcousticRVQDecoder
from semantic_acoustic_codec.runtime.protocol import TeacherCodec
from semantic_acoustic_codec.teacher import LongCatTeacher

SCHEMA_VERSION = 1
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
class SemanticCodecConfig:
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


class SemanticAcousticCodec(nn.Module):
    """Semantic-only codec runtime built from a teacher codec and acoustic decoder."""

    def __init__(
        self,
        teacher: TeacherCodec,
        modules: RouteModules,
        *,
        sampling: SamplingConfig | None = None,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.conditioner = modules.conditioner
        self.reference_conditioner = modules.reference_conditioner
        self.decoder = modules.decoder
        self.route = modules.route
        self.sampling = SamplingConfig() if sampling is None else sampling
        self.feature_mean = nn.Buffer(_feature_stat(teacher, feature_mean, fill=0.0))
        self.feature_std = nn.Buffer(_feature_stat(teacher, feature_std, fill=1.0))

    @property
    def sample_rate(self) -> int:
        return self.teacher.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.teacher.frame_rate

    @property
    def semantic_codebook(self) -> Tensor:
        return self.teacher.semantic_codebook

    @torch.no_grad()
    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        codes = self.teacher.encode(audio, sample_rate)
        if codes.dim() != 3 or codes.size(-1) < 1:
            raise ValueError("teacher encode must return codes with shape [B, F, K].")
        if not is_signed_integer_dtype(codes.dtype):
            raise TypeError("teacher encode must return signed integer codes.")
        return codes[..., :1].to(dtype=torch.long).contiguous()

    @torch.no_grad()
    def decode(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_acoustic_codes: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        features = self.sample_features(
            prepared,
            mask=frame_mask,
            reference_acoustic_codes=reference_acoustic_codes,
            reference_features=reference_features,
            reference_mask=reference_mask,
            generator=generator,
        )
        return self.teacher.decode_features(prepared, features)

    @torch.no_grad()
    def sample_features(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_acoustic_codes: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        condition = self.condition(
            prepared,
            mask=frame_mask,
            reference_acoustic_codes=reference_acoustic_codes,
            reference_features=reference_features,
            reference_mask=reference_mask,
        )
        if self.route is Route.FM:
            decoder = cast(DiTDecoder, self.decoder)
            features = decoder.sample(
                condition,
                mask=frame_mask,
                steps=self.sampling.flow_steps,
                generator=generator,
            )
        elif self.route is Route.RVQ:
            decoder = cast(AcousticRVQDecoder, self.decoder)
            acoustic_codes = decoder.generate(
                condition,
                mask=frame_mask,
                temperature=self.sampling.temperature,
                top_p=self.sampling.top_p,
                generator=generator,
            )
            features = self.teacher.acoustic_codes_to_features(acoustic_codes)
            features = features.to(device=condition.device, dtype=condition.dtype)
        else:
            raise AssertionError(f"unsupported route: {self.route}")
        if self.route is Route.FM:
            features = features * self.feature_std + self.feature_mean
        return features.masked_fill(~frame_mask[..., None], 0)

    @torch.no_grad()
    def sample_acoustic_codes(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_acoustic_codes: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.route is not Route.RVQ:
            raise RuntimeError("sample_acoustic_codes is only available for the RVQ route.")
        prepared, frame_mask = self._semantic_input(semantic_codes, mask)
        condition = self.condition(
            prepared,
            mask=frame_mask,
            reference_acoustic_codes=reference_acoustic_codes,
            reference_features=reference_features,
            reference_mask=reference_mask,
        )
        decoder = cast(AcousticRVQDecoder, self.decoder)
        return decoder.generate(
            condition,
            mask=frame_mask,
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
        if features.size(-1) != self.teacher.acoustic_feature_dim:
            raise ValueError("features must match teacher acoustic_feature_dim.")
        return self.teacher.decode_features(prepared, features)

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
                features = self.teacher.acoustic_codes_to_features(safe_codes)
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


def build_codec(teacher: TeacherCodec, config: SemanticCodecConfig) -> SemanticAcousticCodec:
    modules = build_route(
        config.route,
        teacher,
        condition_dim=config.condition_dim,
        decoder=config.decoder,
        adapter=config.adapter,
        initialization=config.initialization,
        seed=config.seed,
    )
    mean = None if config.feature_mean is None else torch.tensor(config.feature_mean, dtype=torch.float32)
    std = None if config.feature_std is None else torch.tensor(config.feature_std, dtype=torch.float32)
    return SemanticAcousticCodec(
        teacher,
        modules,
        sampling=config.sampling,
        feature_mean=mean,
        feature_std=std,
    )


def save_artifact(path: str | Path, codec: SemanticAcousticCodec, config: SemanticCodecConfig) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if config.route is not codec.route:
        raise ValueError("artifact config route must match codec route.")
    torch.save(codec.state_dict(), root / CHECKPOINT_NAME)
    data = {
        "schema_version": SCHEMA_VERSION,
        "config": _config_dict(config),
        "teacher": {
            "name": "longcat",
            "sample_rate": codec.sample_rate,
            "frame_rate": codec.frame_rate,
            "semantic_vocab_size": int(codec.semantic_codebook.size(0)),
            "acoustic_feature_dim": int(codec.teacher.acoustic_feature_dim),
            "acoustic_codebook_sizes": list(codec.teacher.acoustic_codebook_sizes),
        },
        "checkpoint": CHECKPOINT_NAME,
    }
    (root / CONFIG_NAME).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_artifact(
    path: str | Path,
    *,
    teacher: TeacherCodec | None = None,
    device: str | torch.device | None = None,
) -> SemanticAcousticCodec:
    root = Path(path)
    data = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported semantic codec schema: {data.get('schema_version')!r}")
    teacher = LongCatTeacher.from_pretrained(device=None if device is None else str(device)) if teacher is None else teacher
    config = _config(data["config"])
    _validate_teacher_metadata(cast(Mapping[str, object], data["teacher"]), teacher)
    codec = build_codec(teacher, config)
    state = _load_state(root / str(data.get("checkpoint", CHECKPOINT_NAME)), device=device)
    codec.load_state_dict(state)
    if device is not None:
        codec.to(device=device)
    codec.eval()
    return codec


def _load_state(path: Path, *, device: str | torch.device | None) -> Mapping[str, Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, Mapping):
        raise TypeError("semantic codec checkpoint must contain a state dict mapping.")
    return cast(Mapping[str, Tensor], state)


def _config_dict(config: SemanticCodecConfig) -> dict[str, object]:
    data = asdict(config)
    data["route"] = config.route.value
    data["adapter"] = None if config.adapter is None else config.adapter.value
    data["initialization"] = config.initialization.value
    return cast(dict[str, object], data)


def _config(data: Mapping[str, Any]) -> SemanticCodecConfig:
    decoder = cast(Mapping[str, Any], data["decoder"])
    sampling = cast(Mapping[str, Any], data["sampling"])
    adapter = data.get("adapter")
    return SemanticCodecConfig(
        route=Route(cast(str, data["route"])),
        condition_dim=int(data["condition_dim"]),
        decoder=DecoderConfig(
            hidden_dim=cast(int | None, decoder["hidden_dim"]),
            layers=int(decoder["layers"]),
            heads=int(decoder["heads"]),
            ffn_ratio=int(decoder["ffn_ratio"]),
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


def _validate_teacher_metadata(data: Mapping[str, object], teacher: TeacherCodec) -> None:
    expected = {
        "sample_rate": teacher.sample_rate,
        "frame_rate": teacher.frame_rate,
        "semantic_vocab_size": int(teacher.semantic_codebook.size(0)),
        "acoustic_feature_dim": teacher.acoustic_feature_dim,
        "acoustic_codebook_sizes": list(teacher.acoustic_codebook_sizes),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"teacher metadata mismatch for {key}: {data.get(key)!r} != {value!r}")


def _feature_stat(teacher: TeacherCodec, value: Tensor | None, *, fill: float) -> Tensor:
    if value is None:
        return torch.full((1, 1, teacher.acoustic_feature_dim), fill)
    if value.dim() == 1:
        value = value.view(1, 1, -1)
    if value.shape != (1, 1, teacher.acoustic_feature_dim):
        raise ValueError("feature normalization must match teacher acoustic_feature_dim.")
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
