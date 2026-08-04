from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from anytrain.codec import SemanticAcousticCodecSpec, semantic_acoustic_spec

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec

    from semantic_acoustic_generator.runtime.semantic import GeneratorSupport

__all__ = [
    "support_metadata",
    "validate_backend_metadata",
    "validate_support_metadata",
]


def support_metadata(support: GeneratorSupport) -> dict[str, object]:
    return _spec_metadata(support.codec_spec)


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
    return _spec_metadata(semantic_acoustic_spec(backend))


def _spec_metadata(spec: SemanticAcousticCodecSpec) -> dict[str, object]:
    return {
        "semantic_vocab_size": spec.semantic_codebook_sizes[0],
        "semantic_embedding_dim": spec.semantic_embedding_dim,
        "acoustic_feature_dim": spec.acoustic_feature_dim,
        "acoustic_codebook_sizes": list(spec.acoustic_codebook_sizes),
        "acoustic_layout": spec.acoustic_layout.value,
        "acoustic_unit_length": spec.acoustic_unit_length,
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
