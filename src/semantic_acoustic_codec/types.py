from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from semantic_acoustic_codec._tensor import is_signed_integer_dtype


@dataclass(eq=False)
class SemanticCodecBatch:
    semantic_codes: Tensor
    acoustic_codes: Tensor
    mask: Tensor
    semantic_pad_id: int
    acoustic_pad_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.semantic_codes.dim() != 3 or self.semantic_codes.size(-1) != 1:
            raise ValueError("semantic_codes must have shape [B, F, 1].")
        if self.acoustic_codes.dim() != 3 or self.acoustic_codes.size(-1) < 1:
            raise ValueError("acoustic_codes must have shape [B, F, K].")
        if self.semantic_codes.shape[:2] != self.acoustic_codes.shape[:2]:
            raise ValueError("semantic and acoustic codes must align on [B, F].")
        if self.mask.shape != self.semantic_codes.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [B, F].")
        if not is_signed_integer_dtype(self.semantic_codes.dtype):
            raise TypeError("semantic_codes must use a signed integer dtype.")
        if not is_signed_integer_dtype(self.acoustic_codes.dtype):
            raise TypeError("acoustic_codes must use a signed integer dtype.")
        if self.semantic_codes.size(0) < 1 or self.semantic_codes.size(1) < 1:
            raise ValueError("SemanticCodecBatch must not be empty.")
        if not bool(self.mask.any(dim=1).all()):
            raise ValueError("each batch row must contain at least one valid frame.")
        _check_positive_int(self.semantic_pad_id, name="semantic_pad_id")
        if len(self.acoustic_pad_ids) != self.acoustic_codes.size(-1):
            raise ValueError("acoustic_pad_ids must match acoustic codebooks.")
        for index, pad_id in enumerate(self.acoustic_pad_ids):
            _check_positive_int(pad_id, name=f"acoustic_pad_ids[{index}]")
        if bool((self.semantic_codes < 0).any()):
            raise ValueError("semantic_codes must not contain negative IDs.")
        if bool((self.acoustic_codes < 0).any()):
            raise ValueError("acoustic_codes must not contain negative IDs.")
        semantic = self.semantic_codes[..., 0]
        if bool((semantic[self.mask] >= self.semantic_pad_id).any()):
            raise ValueError("valid semantic_codes contain an ID outside the semantic codebook.")
        if bool((semantic[~self.mask] != self.semantic_pad_id).any()):
            raise ValueError("padded semantic_codes must equal semantic_pad_id.")
        pad_ids = torch.tensor(
            self.acoustic_pad_ids,
            device=self.acoustic_codes.device,
            dtype=self.acoustic_codes.dtype,
        )
        if bool((self.acoustic_codes[self.mask] >= pad_ids).any()):
            raise ValueError("valid acoustic_codes contain an ID outside their codebooks.")
        if bool((self.acoustic_codes[~self.mask] != pad_ids).any()):
            raise ValueError("padded acoustic_codes must equal acoustic_pad_ids.")

    @property
    def semantic_codebook_size(self) -> int:
        return self.semantic_pad_id

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]:
        return self.acoustic_pad_ids


def _check_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = ["SemanticCodecBatch"]
