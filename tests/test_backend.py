from __future__ import annotations

from types import SimpleNamespace

import pytest
from anytrain.codec import AcousticLayout

import semantic_acoustic_generator.backend.loader as backend_loader
from semantic_acoustic_generator.backend import BackendConfig, load_backend


def test_backend_config_rejects_bicodec_semantic_global_contract() -> None:
    with pytest.raises(ValueError, match="semantic-global"):
        BackendConfig(name="bicodec")


def test_load_backend_keeps_default_loader_for_non_bicodec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    loaded = SimpleNamespace(
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
        acoustic_unit_length=None,
    )

    def load_semantic_acoustic(name: str, *, device: object) -> object:
        calls.append((name, device))
        return loaded

    monkeypatch.setattr(
        backend_loader,
        "load_semantic_acoustic",
        load_semantic_acoustic,
    )
    config = BackendConfig(name="longcat")

    assert load_backend(config, device="cpu") is loaded
    assert calls == [("longcat", "cpu")]


def test_load_backend_rejects_non_aligned_acoustic_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = SimpleNamespace(
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
        acoustic_unit_length=32,
    )
    monkeypatch.setattr(
        backend_loader,
        "load_semantic_acoustic",
        lambda *args, **kwargs: loaded,
    )

    with pytest.raises(ValueError, match="frame-aligned"):
        load_backend(BackendConfig(name="fixed-test"), device="cpu")


def test_backend_config_rejects_string_booleans() -> None:
    with pytest.raises(TypeError, match="backend.local_files_only must be a boolean"):
        BackendConfig(local_files_only="false")  # type: ignore[arg-type]


def test_backend_config_rejects_non_string_optional_values() -> None:
    with pytest.raises(TypeError, match="backend.revision must be a string"):
        BackendConfig(revision=123)  # type: ignore[arg-type]


def test_backend_config_rejects_empty_optional_strings() -> None:
    with pytest.raises(ValueError, match="backend.model_dir must be non-empty"):
        BackendConfig(model_dir="")


def test_load_backend_rejects_dynamic_mappings() -> None:
    with pytest.raises(TypeError, match="config must be a BackendConfig"):
        load_backend({"name": "longcat"}, device="cpu")  # type: ignore[arg-type]
