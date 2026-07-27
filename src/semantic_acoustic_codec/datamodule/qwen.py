from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, cast

from anydataset.types import AudioView, Role
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor
from torch.utils.data import Dataset
from zhuyin.datasets.qwen_tts_speech import (
    QwenCodec,
    qwen_tts_speaker_codec_grid,
)

from semantic_acoustic_codec.backend.longcat import split_codes as split_longcat_codes


@dataclass(frozen=True)
class QwenCodecSample:
    index: int
    text_index: int
    source_index: int
    role: Role
    utterance_id: str
    speaker_id: str
    text: str
    codes: SemanticAcousticCodes


@dataclass(frozen=True)
class QwenCodecPairSample:
    target_index: int
    reference_index: int
    target: QwenCodecSample
    reference: QwenCodecSample


class QwenCodecColumnDataset(Dataset[QwenCodecSample]):
    """Expose one role and speaker column from a prepared Qwen codec grid."""

    def __init__(
        self,
        *,
        codec: QwenCodec | str,
        root: str | PathLike[str] | None,
        split: str,
        role: Role,
        speaker_id: str,
    ) -> None:
        super().__init__()
        self.codec = QwenCodec(codec)
        self.view = (
            AudioView.BICODEC
            if self.codec is QwenCodec.BICODEC
            else AudioView.LONGCAT
        )
        self.grid = qwen_tts_speaker_codec_grid(
            codec=self.codec,
            root=root,
            split=split,
        )
        self.split = split
        self.role = role
        if speaker_id not in self.grid.speaker_ids:
            raise ValueError(
                f"speaker {speaker_id!r} is not present in Qwen grid speakers "
                f"{self.grid.speaker_ids!r}."
            )
        self.speaker_id = speaker_id
        self.text_indices = tuple(
            index for index, row in enumerate(self.grid.row_specs) if row.role is role
        )
        if not self.text_indices:
            raise ValueError(f"Qwen codec grid has no rows for role {role.value!r}.")

    def __len__(self) -> int:
        return len(self.text_indices)

    def __getitem__(self, index: int) -> QwenCodecSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Qwen codec column sample index out of range.")
        text_index = self.text_indices[index]
        block = self.grid.select(
            text=text_index,
            speaker=self.speaker_id,
        ).load(view=self.view)
        source_index = block.source_indices[0]
        role = block.roles[0]
        text = block.texts[0]
        if not text:
            raise ValueError("Qwen codec grid text must not be empty.")
        return QwenCodecSample(
            index=index,
            text_index=text_index,
            source_index=source_index,
            role=role,
            utterance_id=(
                f"{self.split}-{source_index:08d}-{role.value}-{text_index:08d}-"
                f"{self.speaker_id}"
            ),
            speaker_id=self.speaker_id,
            text=text,
            codes=_codes(block.audio.views[self.view], self.codec, block.lengths),
        )


class QwenCodecPairDataset(Dataset[QwenCodecPairSample]):
    """Pair every column sample with the next valid cross-text sample."""

    def __init__(self, source: QwenCodecColumnDataset) -> None:
        super().__init__()
        self.source = source
        samples = tuple(source[index] for index in range(len(source)))
        self.samples = samples
        self.reference_indices = _reference_indices(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> QwenCodecPairSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Qwen codec pair sample index out of range.")
        reference_index = self.reference_indices[index]
        return QwenCodecPairSample(
            target_index=index,
            reference_index=reference_index,
            target=self.samples[index],
            reference=self.samples[reference_index],
        )


def _codes(
    value: object,
    codec: QwenCodec,
    lengths: Tensor,
) -> SemanticAcousticCodes:
    length = int(lengths[0, 0].item())
    if codec is QwenCodec.LONGCAT:
        if not isinstance(value, Tensor):
            raise TypeError("Qwen LongCat view must be a Tensor.")
        combined = value[0, 0, :length].contiguous()
        semantic, acoustic = split_longcat_codes(combined)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)
    if codec is QwenCodec.BICODEC:
        if not isinstance(value, Mapping):
            raise TypeError("Qwen BiCodec view must be a semantic/acoustic mapping.")
        fields = cast(Mapping[str, Any], value)
        semantic = fields["semantic"]
        acoustic = fields["acoustic"]
        if not isinstance(semantic, Tensor) or not isinstance(acoustic, Tensor):
            raise TypeError("Qwen BiCodec semantic and acoustic values must be Tensors.")
        return SemanticAcousticCodes(
            semantic=semantic[0, 0, :length].contiguous(),
            acoustic=acoustic[0, 0].contiguous(),
        )
    raise ValueError(f"unsupported Qwen codec: {codec!r}.")


def _reference_indices(samples: tuple[QwenCodecSample, ...]) -> tuple[int, ...]:
    if len(samples) < 2:
        raise ValueError("cross-text pairing requires at least two Qwen codec samples.")
    references: list[int] = []
    for target_index, target in enumerate(samples):
        for offset in range(1, len(samples)):
            reference_index = (target_index + offset) % len(samples)
            reference = samples[reference_index]
            if reference.speaker_id != target.speaker_id:
                continue
            if reference.text_index == target.text_index:
                continue
            if reference.utterance_id == target.utterance_id:
                continue
            if reference.text == target.text:
                continue
            references.append(reference_index)
            break
        else:
            raise ValueError(
                f"Qwen codec sample {target.utterance_id!r} has no same-speaker "
                "reference from a different text row with a different utterance id and text."
            )
    return tuple(references)


__all__ = [
    "QwenCodecColumnDataset",
    "QwenCodecPairDataset",
    "QwenCodecPairSample",
    "QwenCodecSample",
]
