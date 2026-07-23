from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from anydataset.types import AudioItem, AudioView, Modality, Role
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from semantic_acoustic_codec._tensor import is_signed_integer_dtype

LONGCAT_VIEW = AudioView.LONGCAT
PAD_ID = -1


@dataclass(eq=False)
class SemanticCodecBatch:
    semantic_codes: Tensor
    acoustic_codes: Tensor
    mask: Tensor

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

    @property
    def semantic_tokens(self) -> Tensor:
        return self.semantic_codes[..., 0].masked_fill(~self.mask, 0)

    @property
    def safe_acoustic_codes(self) -> Tensor:
        return self.acoustic_codes.masked_fill(~self.mask[..., None], 0)


def codes(sample: Mapping[Any, Any], *, role: Role = Role.TARGET) -> Tensor:
    item = sample[(role, Modality.AUDIO)]
    if not isinstance(item, AudioItem):
        raise TypeError("LongCat sample audio must be an AudioItem.")
    value = item.views[LONGCAT_VIEW]
    if not isinstance(value, Tensor) or value.dim() != 2:
        raise ValueError("LongCat prepared codes must have shape [frame, codebook].")
    if value.size(0) < 1:
        raise ValueError("LongCat prepared code sequence must not be empty.")
    if value.size(1) < 2:
        raise ValueError("LongCat prepared codes must include semantic and acoustic codebooks.")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError("LongCat prepared codes must use a signed integer dtype.")
    if bool((value < 0).any()):
        raise ValueError("LongCat prepared codes must not contain negative IDs.")
    return value.to(dtype=torch.long).contiguous()


def split_codes(value: Tensor) -> tuple[Tensor, Tensor]:
    if value.dim() != 2 or value.size(1) < 2:
        raise ValueError("LongCat codes must have shape [frame, semantic+acoustic].")
    return value[:, :1].contiguous(), value[:, 1:].contiguous()


def collate(samples: Sequence[Mapping[Any, Any]], *, role: Role = Role.TARGET) -> SemanticCodecBatch:
    if not samples:
        raise ValueError("cannot collate an empty semantic codec batch.")
    split = [split_codes(codes(sample, role=role)) for sample in samples]
    semantic = pad_sequence(
        [item[0] for item in split],
        batch_first=True,
        padding_value=PAD_ID,
    )
    acoustic = pad_sequence(
        [item[1] for item in split],
        batch_first=True,
        padding_value=PAD_ID,
    )
    mask = semantic[..., 0] != PAD_ID
    return SemanticCodecBatch(
        semantic_codes=semantic,
        acoustic_codes=acoustic,
        mask=mask,
    )
