from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from semantic_acoustic_codec.config import Route
from semantic_acoustic_codec.runtime import (
    SemanticAcousticCodec,
    SemanticCodecConfig,
    TeacherCodec,
    build_codec,
    save_artifact,
)
from semantic_acoustic_codec.teacher import LongCatTeacher


def export_legacy_s2s_oracle(
    checkpoint: str | Path,
    output_dir: str | Path,
    config: SemanticCodecConfig,
    *,
    teacher: TeacherCodec | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> SemanticAcousticCodec:
    """Convert a legacy speech-to-speech codec-oracle checkpoint to a SAC artifact."""

    if config.route not in {Route.FM, Route.RVQ}:
        raise ValueError("legacy speech-to-speech oracle export supports fm and rvq routes.")
    teacher = LongCatTeacher.from_pretrained(device=_device_name(device)) if teacher is None else teacher
    codec = build_codec(teacher, config)
    target_state = dict(codec.state_dict())
    source_state = _state_dict(_checkpoint_state(Path(checkpoint), device=device))
    mapped, unused = _mapped_state(source_state, route=config.route)
    _require_legacy_sections(mapped, route=config.route)
    if strict:
        missing = _missing(target_state, mapped, route=config.route)
        if missing:
            joined = ", ".join(missing[:8])
            raise KeyError(f"legacy speech-to-speech oracle checkpoint is missing SAC state keys: {joined}")
        _reject_trainable_leftovers(unused)

    target_state.update(mapped)
    codec.load_state_dict(target_state, strict=True)
    if device is not None:
        codec.to(device=device)
    codec.eval()
    save_artifact(output_dir, codec, _artifact_config(config, codec))
    return codec


def export_legacy_oracle_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    config: SemanticCodecConfig,
    *,
    teacher: TeacherCodec | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> SemanticAcousticCodec:
    """Alias for old callers while SAC owns the conversion implementation."""

    return export_legacy_s2s_oracle(
        checkpoint,
        output_dir,
        config,
        teacher=teacher,
        device=device,
        strict=strict,
    )


def legacy_s2s_oracle_state_dict(
    checkpoint: Mapping[str, object],
    *,
    route: Route,
) -> dict[str, Tensor]:
    """Map old S2S oracle Lightning keys to the SAC runtime keyspace."""

    mapped, _ = _mapped_state(_state_dict(checkpoint), route=route)
    _require_legacy_sections(mapped, route=route)
    return mapped


def _checkpoint_state(path: Path, *, device: str | torch.device | None) -> Mapping[str, object]:
    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location=device)
    if not isinstance(raw, Mapping):
        raise TypeError("legacy speech-to-speech oracle checkpoint must be a mapping.")
    return cast(Mapping[str, object], raw)


def _state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, Tensor]:
    value = checkpoint.get("state_dict", checkpoint)
    if not isinstance(value, Mapping):
        raise TypeError("legacy speech-to-speech oracle checkpoint state_dict must be a mapping.")
    state = cast(Mapping[str, object], value)
    if any(not isinstance(item, Tensor) for item in state.values()):
        raise TypeError("legacy speech-to-speech oracle state_dict values must be tensors.")
    return cast(Mapping[str, Tensor], state)


def _mapped_state(
    state: Mapping[str, Tensor],
    *,
    route: Route,
) -> tuple[dict[str, Tensor], set[str]]:
    mapped: dict[str, Tensor] = {}
    unused = set(state)
    for key, value in state.items():
        target = _target_key(key, route=route)
        if target is None:
            continue
        mapped[target] = value
        unused.discard(key)
    if not mapped:
        raise ValueError("legacy speech-to-speech oracle checkpoint produced an empty SAC state dict.")
    return mapped, unused


def _target_key(key: str, *, route: Route) -> str | None:
    for source, target in (
        ("model.semantic_audio_embedding.", "conditioner.embedding."),
        ("semantic_audio_embedding.", "conditioner.embedding."),
        ("model.semantic_audio_adapter.", "conditioner.adapter."),
        ("semantic_audio_adapter.", "conditioner.adapter."),
    ):
        if key.startswith(source):
            return target + key.removeprefix(source)

    if key == "target_mean":
        return "feature_mean"
    if key == "target_std":
        return "feature_std"

    if route is Route.FM:
        for source in ("model.acoustic_flow.decoder.", "acoustic_flow.decoder."):
            if key.startswith(source):
                return "decoder.decoder." + key.removeprefix(source)
    elif route is Route.RVQ:
        for source in ("model.acoustic_decoder.", "acoustic_decoder."):
            if key.startswith(source):
                return "decoder." + key.removeprefix(source)
    else:
        raise ValueError("legacy speech-to-speech oracle export supports fm and rvq routes.")
    return None


def _require_legacy_sections(mapped: Mapping[str, Tensor], *, route: Route) -> None:
    _require_prefix(mapped, "conditioner.embedding.")
    _require_prefix(mapped, "conditioner.adapter.")
    if route is Route.FM:
        _require_prefix(mapped, "decoder.decoder.")
        _require_key(mapped, "feature_mean")
        _require_key(mapped, "feature_std")
    elif route is Route.RVQ:
        _require_prefix(mapped, "decoder.")
    else:
        raise ValueError("legacy speech-to-speech oracle export supports fm and rvq routes.")


def _missing(
    state: Mapping[str, Tensor],
    mapped: Mapping[str, Tensor],
    *,
    route: Route,
) -> list[str]:
    optional = {"reference_conditioner."}
    if route is Route.RVQ:
        optional.update({"feature_mean", "feature_std"})
    return sorted(
        key
        for key in state
        if key not in mapped and not _optional_missing(key, optional)
    )


def _optional_missing(key: str, optional: set[str]) -> bool:
    return key in optional or any(key.startswith(prefix) for prefix in optional if prefix.endswith("."))


def _artifact_config(
    config: SemanticCodecConfig,
    codec: SemanticAcousticCodec,
) -> SemanticCodecConfig:
    return replace(
        config,
        feature_mean=tuple(float(item) for item in codec.feature_mean.detach().cpu().view(-1)),
        feature_std=tuple(float(item) for item in codec.feature_std.detach().cpu().view(-1)),
    )


def _reject_trainable_leftovers(unused: set[str]) -> None:
    leftovers = sorted(
        key
        for key in unused
        if key.startswith(("model.semantic_", "model.acoustic_", "semantic_", "acoustic_"))
    )
    if leftovers:
        joined = ", ".join(leftovers[:8])
        raise KeyError(f"unmapped legacy speech-to-speech oracle state keys: {joined}")


def _require_prefix(mapped: Mapping[str, Tensor], prefix: str) -> None:
    if not any(key.startswith(prefix) for key in mapped):
        raise ValueError(f"legacy speech-to-speech oracle checkpoint is missing {prefix}*")


def _require_key(mapped: Mapping[str, Tensor], key: str) -> None:
    if key not in mapped:
        raise ValueError(f"legacy speech-to-speech flow oracle checkpoint is missing {key}.")


def _device_name(device: str | torch.device | None) -> str | None:
    return None if device is None else str(device)


__all__ = [
    "export_legacy_oracle_checkpoint",
    "export_legacy_s2s_oracle",
    "legacy_s2s_oracle_state_dict",
]
