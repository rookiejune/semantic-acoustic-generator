from __future__ import annotations

from importlib import import_module
from typing import Any

from semantic_acoustic_codec.runtime.protocol import CodecBackend

_LAZY = {
    "SamplingConfig",
    "SemanticCodecSupport",
    "SemanticSupportConfig",
    "build_support",
    "load_artifact",
    "save_artifact",
}

__all__ = [
    "SamplingConfig",
    "CodecBackend",
    "SemanticCodecSupport",
    "SemanticSupportConfig",
    "build_support",
    "load_artifact",
    "save_artifact",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(name)
    semantic = import_module("semantic_acoustic_codec.runtime.semantic")
    return getattr(semantic, name)
