from __future__ import annotations

from typing import Protocol

from torch import Tensor


class TeacherCodec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def semantic_codebook(self) -> Tensor: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def decode(self, codes: Tensor) -> Tensor: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor: ...
