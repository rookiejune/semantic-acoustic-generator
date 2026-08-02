from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

import semantic_acoustic_codec.backend.loader as backend_loader
from semantic_acoustic_codec.backend import load_backend


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
    config = OmegaConf.create(
        {
            "name": "bicodec",
            "model_dir": "/models/bicodec",
            "revision": "0123456789abcdef",
            "local_files_only": False,
            "allow_unpinned_revision": True,
        }
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
    config = OmegaConf.create({"name": "longcat"})

    assert load_backend(config, device="cpu") is loaded
    assert calls == [("longcat", "cpu")]
