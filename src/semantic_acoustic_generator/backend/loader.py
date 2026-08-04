from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import torch
from anytrain.codec import load_semantic_acoustic

from semantic_acoustic_generator.backend.config import BackendConfig

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec


class _BiCodecFactory(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        *,
        model_dir: str | None,
        revision: str | None,
        device: str | torch.device | None,
        local_files_only: bool,
        allow_unpinned_revision: bool,
    ) -> SemanticAcousticCodec: ...


def load_backend(
    config: BackendConfig,
    device: str | torch.device | None,
) -> SemanticAcousticCodec:
    if not isinstance(config, BackendConfig):
        raise TypeError("config must be a BackendConfig.")
    name = config.name
    if name != "bicodec":
        return load_semantic_acoustic(name, device=device)
    return _bicodec_type().from_pretrained(
        model_dir=config.model_dir,
        revision=config.revision,
        device=device,
        local_files_only=config.local_files_only,
        allow_unpinned_revision=config.allow_unpinned_revision,
    )


def _bicodec_type() -> type[_BiCodecFactory]:
    from anytrain.codec.bicodec import BiCodec

    return cast(type[_BiCodecFactory], BiCodec)
