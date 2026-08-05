from __future__ import annotations

import torch
from anytrain.codec import (
    AcousticLayout,
    SemanticAcousticCodec,
    load_semantic_acoustic,
)

from semantic_acoustic_generator.backend.config import BackendConfig


def load_backend(
    config: BackendConfig,
    device: str | torch.device | None,
) -> SemanticAcousticCodec:
    if not isinstance(config, BackendConfig):
        raise TypeError("config must be a BackendConfig.")
    backend = load_semantic_acoustic(config.name, device=device)
    _validate_frame_aligned(backend)
    return backend


def _validate_frame_aligned(backend: SemanticAcousticCodec) -> None:
    if backend.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
        raise ValueError(
            "semantic-acoustic-generator requires frame-aligned acoustic units."
        )
    if backend.acoustic_unit_length is not None:
        raise ValueError(
            "frame-aligned semantic-acoustic backends must not set acoustic_unit_length."
        )
