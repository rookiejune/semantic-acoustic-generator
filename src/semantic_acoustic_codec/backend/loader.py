from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from anytrain.codec import load_semantic_acoustic

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
    config: Any,
    device: str | torch.device | None,
) -> SemanticAcousticCodec:
    name = str(config.name)
    if name != "bicodec":
        return load_semantic_acoustic(name, device=device)
    return _bicodec_type().from_pretrained(
        model_dir=_optional_string(config, "model_dir"),
        revision=_optional_string(config, "revision"),
        device=device,
        local_files_only=bool(config.get("local_files_only", True)),
        allow_unpinned_revision=bool(config.get("allow_unpinned_revision", False)),
    )


def _optional_string(config: Any, key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    return str(value)


def _bicodec_type() -> type[_BiCodecFactory]:
    from anytrain.codec.bicodec import BiCodec

    return cast(type[_BiCodecFactory], BiCodec)
