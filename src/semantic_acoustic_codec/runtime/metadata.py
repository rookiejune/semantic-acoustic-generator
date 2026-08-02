from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec

    from semantic_acoustic_codec.runtime.semantic import SemanticCodecSupport

__all__ = [
    "support_metadata",
    "validate_backend_metadata",
    "validate_support_metadata",
]


def support_metadata(support: SemanticCodecSupport) -> dict[str, object]:
    return {
        "semantic_vocab_size": support.conditioner.semantic_codebook_size,
        "semantic_embedding_dim": support.conditioner.embedding.embedding_dim,
        "acoustic_feature_dim": support.acoustic_feature_dim,
        "acoustic_codebook_sizes": list(support.acoustic_codebook_sizes),
        "acoustic_layout": support.acoustic_layout.value,
        "acoustic_unit_length": support.acoustic_unit_length,
    }


def validate_backend_metadata(
    data: Mapping[str, object],
    backend: SemanticAcousticCodec,
) -> None:
    expected = {
        "name": backend.name,
        "sample_rate": backend.sample_rate,
        "frame_rate": float(backend.frame_rate),
        "semantic_frame_rate": float(backend.semantic_frame_rate),
        **_expected_support_metadata(backend),
    }
    _validate_metadata(data, expected)


def validate_support_metadata(
    data: Mapping[str, object],
    backend: SemanticAcousticCodec,
) -> None:
    _validate_metadata(data, _expected_support_metadata(backend))


def _expected_support_metadata(backend: SemanticAcousticCodec) -> dict[str, object]:
    return {
        "semantic_vocab_size": int(backend.semantic_codebook.size(0)),
        "semantic_embedding_dim": int(backend.semantic_codebook.size(1)),
        "acoustic_feature_dim": backend.acoustic_feature_dim,
        "acoustic_codebook_sizes": list(backend.acoustic_codebook_sizes),
        "acoustic_layout": backend.acoustic_layout.value,
        "acoustic_unit_length": backend.acoustic_unit_length,
    }


def _validate_metadata(
    data: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(
                f"backend metadata mismatch for {key}: {data.get(key)!r} != {value!r}"
            )
