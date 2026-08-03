from __future__ import annotations

from typing import Any

import pytest

import semantic_acoustic_codec.backend.loader as backend_loader
from semantic_acoustic_codec.backend import BackendConfig, load_backend


def test_load_backend_passes_bicodec_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    loaded = object()

    class FakeBiCodec:
        @classmethod
        def from_pretrained(cls, **kwargs: Any) -> object:
            calls.append(kwargs)
            return loaded

    monkeypatch.setattr(backend_loader, "_bicodec_type", lambda: FakeBiCodec)
    config = BackendConfig(
        name="bicodec",
        model_dir="/models/bicodec",
        revision="0123456789abcdef",
        local_files_only=False,
        allow_unpinned_revision=True,
    )

    assert load_backend(config, device="cpu") is loaded
    assert calls == [
        {
            "model_dir": "/models/bicodec",
            "revision": "0123456789abcdef",
            "device": "cpu",
            "local_files_only": False,
            "allow_unpinned_revision": True,
        }
    ]


def test_load_backend_keeps_default_loader_for_non_bicodec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    loaded = object()

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
