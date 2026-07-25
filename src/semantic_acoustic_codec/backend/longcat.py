from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from semantic_acoustic_codec._tensor import is_signed_integer_dtype
from semantic_acoustic_codec.types import SemanticCodecBatch

if TYPE_CHECKING:
    from anydataset.types import Role


LONGCAT_CODEBOOK_SIZES = (8192, 8100, 8100, 8100)


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


def codes(sample: Mapping[Any, Any], *, role: Role | None = None) -> Tensor:
    AudioItem, AudioView, Modality, Role = _anydataset_types()
    item = sample[(Role.TARGET if role is None else role, Modality.AUDIO)]
    if not isinstance(item, AudioItem):
        raise TypeError("LongCat sample audio must be an AudioItem.")
    value = item.views[AudioView.LONGCAT]
    _check_codes(value, source="LongCat prepared codes")
    return value.to(dtype=torch.long).contiguous()


def split_codes(value: Tensor) -> tuple[Tensor, Tensor]:
    _check_codes(value, source="LongCat codes")
    return value[:, :1].contiguous(), value[:, 1:].contiguous()


def batch_codes(
    values: Sequence[Tensor],
    *,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> SemanticCodecBatch:
    if not values:
        raise ValueError("cannot batch an empty semantic codec code sequence.")
    acoustic_pads = tuple(int(pad_id) for pad_id in acoustic_pad_ids)
    _check_pad_id(semantic_pad_id, name="semantic_pad_id")
    _check_pad_ids(acoustic_pads)
    split = [split_codes(value) for value in values]
    if any(item[1].size(1) != len(acoustic_pads) for item in split):
        raise ValueError("LongCat acoustic codes must match acoustic_pad_ids.")
    semantic = _pad_semantic([item[0] for item in split], semantic_pad_id=semantic_pad_id)
    mask = semantic[..., 0] != semantic_pad_id
    acoustic = _pad_acoustic([item[1] for item in split], mask=mask, acoustic_pad_ids=acoustic_pads)
    return SemanticCodecBatch(
        semantic_codes=semantic,
        acoustic_codes=acoustic,
        mask=mask,
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pads,
    )


def batch_samples(
    samples: Sequence[Mapping[Any, Any]],
    *,
    role: Role | None = None,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> SemanticCodecBatch:
    if not samples:
        raise ValueError("cannot batch an empty semantic codec sample sequence.")
    return batch_codes(
        [codes(sample, role=role) for sample in samples],
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
    )


def _check_codes(value: object, *, source: str) -> None:
    if not isinstance(value, Tensor) or value.dim() != 2:
        raise ValueError(f"{source} must have shape [frame, codebook].")
    if value.size(0) < 1:
        raise ValueError(f"{source} sequence must not be empty.")
    if value.size(1) < 2:
        raise ValueError(f"{source} must include semantic and acoustic codebooks.")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{source} must use a signed integer dtype.")
    if bool((value < 0).any()):
        raise ValueError(f"{source} must not contain negative IDs.")


def _pad_semantic(values: Sequence[Tensor], *, semantic_pad_id: int) -> Tensor:
    return pad_sequence(values, batch_first=True, padding_value=semantic_pad_id)


def _pad_acoustic(values: Sequence[Tensor], *, mask: Tensor, acoustic_pad_ids: tuple[int, ...]) -> Tensor:
    padded = pad_sequence(values, batch_first=True, padding_value=0)
    pad_ids = torch.tensor(acoustic_pad_ids, device=padded.device, dtype=padded.dtype)
    return torch.where(mask[..., None], padded, pad_ids)


def _check_pad_ids(values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError("acoustic_pad_ids must contain at least one acoustic codebook.")
    for index, value in enumerate(values):
        _check_pad_id(value, name=f"acoustic_pad_ids[{index}]")


def _check_pad_id(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _anydataset_types() -> tuple[type[Any], Any, Any, Any]:
    from anydataset.types import AudioItem, AudioView, Modality, Role

    return AudioItem, AudioView, Modality, Role


__all__ = [
    "LongCatBackend",
    "LONGCAT_CODEBOOK_SIZES",
    "batch_codes",
    "batch_samples",
    "codes",
    "split_codes",
]
