from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from torch import Tensor


class LongCatBackend:
    """Adapter around anytrain LongCat matching the local codec backend protocol."""

    name = "longcat"

    def __init__(self, codec: Any) -> None:
        self.codec = codec
        decoders = list(getattr(codec, "decoders", {}).values())
        latent_dim = None if not decoders else getattr(decoders[0], "latent_dim", None)
        if not isinstance(latent_dim, int):
            raise TypeError("LongCat decoder must expose an integer latent_dim.")
        self._acoustic_feature_dim = latent_dim

    @classmethod
    def from_pretrained(cls, *, device: str | None = None) -> LongCatBackend:
        from anytrain.codec.longcat import LongCat

        return cls(LongCat.from_pretrained(device=device))

    @property
    def sample_rate(self) -> int:
        return int(self.codec.sample_rate)

    @property
    def frame_rate(self) -> float:
        encoder = self.codec.encoder
        return float(encoder.input_sample_rate / encoder.hop_length)

    @property
    def acoustic_feature_dim(self) -> int:
        return self._acoustic_feature_dim

    @property
    def semantic_codebook(self) -> Tensor:
        return self.codec.semantic_codebook

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.codec.codebook_sizes[1:])

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        return self.codec.encode(audio, sample_rate)

    def decode(self, codes: Tensor) -> Tensor:
        return self.codec.decode(codes)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        return self.codec.acoustic_codes_to_features(acoustic_codes)

    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor:
        return self.codec.decode_features(semantic_codes, acoustic_features)
