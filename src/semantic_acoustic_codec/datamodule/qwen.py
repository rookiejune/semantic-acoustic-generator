from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, cast

from anydataset.types import AudioItem, AudioView, Role, TextItem, TextMeta, TextView
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor
from torch.utils.data import Dataset
from zhuyin.datasets.wmt19 import qwen_tts

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


@dataclass(frozen=True)
class _QwenCodecSampleInfo:
    index: int
    text_index: int
    source_index: int
    role: Role
    utterance_id: str
    speaker_id: str
    text: str


class QwenCodecColumnDataset(Dataset[QwenCodecSample]):
    """Expose one role and speaker column from a prepared Qwen codec grid."""

    def __init__(
        self,
        *,
        codec: qwen_tts.Codec | str,
        root: str | PathLike[str] | None,
        split: str,
        role: Role,
        speaker_id: str,
    ) -> None:
        super().__init__()
        self.codec = qwen_tts.Codec(codec)
        self.view = (
            AudioView.BICODEC
            if self.codec is qwen_tts.Codec.BICODEC
            else AudioView.LONGCAT
        )
        self.grid = qwen_tts.speaker_grid(
            codec=self.codec,
            root=root,
            split=split,
        ).load()
        self.split = split
        self.role = role
        if speaker_id not in self.grid.speaker_ids:
            raise ValueError(
                f"speaker {speaker_id!r} is not present in Qwen grid speakers "
                f"{self.grid.speaker_ids!r}."
            )
        self.speaker_id = speaker_id
        self.speaker_index = self.grid.speaker_ids.index(speaker_id)
        self.text_indices = tuple(
            index for index, row in enumerate(self.grid.row_specs) if row.role is role
        )
        if not self.text_indices:
            raise ValueError(f"Qwen codec grid has no rows for role {role.value!r}.")

    def __len__(self) -> int:
        return len(self.text_indices)

    def __getitem__(self, index: int) -> QwenCodecSample:
        info = self.info(index)
        block = self.grid.select(
            text=info.text_index,
            speaker=self.speaker_id,
        ).load(view=self.view)
        return QwenCodecSample(
            index=info.index,
            text_index=info.text_index,
            source_index=info.source_index,
            role=info.role,
            utterance_id=info.utterance_id,
            speaker_id=info.speaker_id,
            text=info.text,
            codes=_codes(block.audio.views[self.view], self.codec, block.lengths),
        )

    def info(self, index: int) -> _QwenCodecSampleInfo:
        index = _index(index, size=len(self), source="Qwen codec column sample")
        text_index = self.text_indices[index]
        row = self.grid.row_specs[text_index]
        text_item = self._text_item(text_index)
        source_index = text_item.meta.get(TextMeta.SOURCE_INDEX)
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise TypeError("Qwen codec text source index must be an integer.")
        if source_index != row.source_index:
            raise ValueError(
                f"Qwen codec text row {text_index} has source index {source_index!r}; "
                f"expected {row.source_index!r}."
            )
        actual_speaker = text_item.views.get(TextView.SPEAKERS)
        if actual_speaker != self.speaker_id:
            raise ValueError(
                f"Qwen codec text row {text_index} has speaker {actual_speaker!r}; "
                f"expected {self.speaker_id!r}."
            )
        text = text_item.views.get(TextView.TEXT)
        if not isinstance(text, str) or not text:
            raise ValueError("Qwen codec grid text must be a non-empty string.")
        return _QwenCodecSampleInfo(
            index=index,
            text_index=text_index,
            source_index=source_index,
            role=row.role,
            utterance_id=(
                f"{self.split}-{source_index:08d}-{row.role.value}-{text_index:08d}-"
                f"{self.speaker_id}"
            ),
            speaker_id=self.speaker_id,
            text=text,
        )

    def raw_length(self, index: int) -> int:
        index = _index(index, size=len(self), source="Qwen codec column sample")
        value = self._audio_item(self.text_indices[index]).views.get(self.view)
        if self.codec is qwen_tts.Codec.LONGCAT:
            if not isinstance(value, Tensor) or value.dim() != 2:
                raise ValueError("Qwen LongCat view must expose [time, codebook].")
            return _positive_length(value.size(0))
        if self.codec is qwen_tts.Codec.BICODEC:
            if not isinstance(value, Mapping):
                raise TypeError("Qwen BiCodec view must be a semantic/acoustic mapping.")
            semantic = cast(Mapping[str, Any], value)["semantic"]
            if not isinstance(semantic, Tensor) or semantic.dim() != 2:
                raise ValueError("Qwen BiCodec semantic view must expose [time, codebook].")
            return _positive_length(semantic.size(0))
        raise ValueError(f"unsupported Qwen codec: {self.codec!r}.")

    def _text_item(self, text_index: int) -> TextItem:
        value = self._cell(text_index).get(self.grid.text_ref)
        if not isinstance(value, TextItem):
            raise TypeError("Qwen codec grid text cell must contain a TextItem.")
        return value

    def _audio_item(self, text_index: int) -> AudioItem:
        value = self._cell(text_index).get(self.grid.audio_ref)
        if not isinstance(value, AudioItem):
            raise TypeError("Qwen codec grid audio cell must contain an AudioItem.")
        return value

    def _cell(self, text_index: int) -> Mapping[Any, Any]:
        flat_index = text_index * len(self.grid.speaker_ids) + self.speaker_index
        return self.grid.cells[flat_index]


class QwenCodecPairDataset(Dataset[QwenCodecPairSample]):
    """Pair every column sample with the next valid cross-text sample."""

    def __init__(self, source: QwenCodecColumnDataset, *, sample_count: int | None = None) -> None:
        super().__init__()
        self.source = source
        size = len(source) if sample_count is None else _sample_count(sample_count, source=source)
        if size < 2:
            raise ValueError("cross-text pairing requires at least two Qwen codec samples.")
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> QwenCodecPairSample:
        index = _index(index, size=len(self), source="Qwen codec pair sample")
        target = self.source[index]
        for offset in range(1, len(self)):
            reference_index = (index + offset) % len(self)
            reference = self.source[reference_index]
            if _is_reference(target, reference):
                return QwenCodecPairSample(
                    target_index=index,
                    reference_index=reference_index,
                    target=target,
                    reference=reference,
                )
        raise ValueError(
            f"Qwen codec sample {target.utterance_id!r} has no same-speaker "
            "reference from a different text row with a different utterance id and text."
        )

    def raw_length(self, index: int) -> int:
        index = _index(index, size=len(self), source="Qwen codec pair sample")
        target = self.source.info(index)
        for offset in range(1, len(self)):
            reference_index = (index + offset) % len(self)
            reference = self.source.info(reference_index)
            if _is_reference_info(target, reference):
                return max(
                    self.source.raw_length(index),
                    self.source.raw_length(reference_index),
                )
        raise ValueError(
            f"Qwen codec sample {target.utterance_id!r} has no same-speaker "
            "reference from a different text row with a different utterance id and text."
        )


def _codes(
    value: object,
    codec: qwen_tts.Codec,
    lengths: Tensor,
) -> SemanticAcousticCodes:
    length = _length(lengths)
    if codec is qwen_tts.Codec.LONGCAT:
        if not isinstance(value, Tensor):
            raise TypeError("Qwen LongCat view must be a Tensor.")
        if value.dim() < 4 or value.size(0) < 1 or value.size(1) < 1:
            raise ValueError("Qwen LongCat view must expose [text, speaker, time, codebook].")
        if length > value.size(2):
            raise ValueError("Qwen LongCat length metadata exceeds available codec frames.")
        combined = value[0, 0, :length].contiguous()
        semantic, acoustic = split_longcat_codes(combined)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)
    if codec is qwen_tts.Codec.BICODEC:
        if not isinstance(value, Mapping):
            raise TypeError("Qwen BiCodec view must be a semantic/acoustic mapping.")
        fields = cast(Mapping[str, Any], value)
        semantic = fields["semantic"]
        acoustic = fields["acoustic"]
        if not isinstance(semantic, Tensor) or not isinstance(acoustic, Tensor):
            raise TypeError("Qwen BiCodec semantic and acoustic values must be Tensors.")
        if semantic.dim() < 4 or semantic.size(0) < 1 or semantic.size(1) < 1:
            raise ValueError("Qwen BiCodec semantic view must expose [text, speaker, time, codebook].")
        if acoustic.dim() < 4 or acoustic.size(0) < 1 or acoustic.size(1) < 1 or acoustic.size(2) < 1:
            raise ValueError("Qwen BiCodec acoustic view must expose [text, speaker, unit, codebook].")
        if length > semantic.size(2):
            raise ValueError("Qwen BiCodec length metadata exceeds available semantic frames.")
        return SemanticAcousticCodes(
            semantic=semantic[0, 0, :length].contiguous(),
            acoustic=acoustic[0, 0].contiguous(),
        )
    raise ValueError(f"unsupported Qwen codec: {codec!r}.")


def _length(lengths: Tensor) -> int:
    if lengths.dim() != 2 or lengths.size(0) < 1 or lengths.size(1) < 1:
        raise ValueError("Qwen codec lengths must expose [text, speaker].")
    length = int(lengths[0, 0].item())
    if length < 1:
        raise ValueError("Qwen codec length metadata must be positive.")
    return length


def _positive_length(value: int) -> int:
    if value < 1:
        raise ValueError("Qwen codec length metadata must be positive.")
    return value


def _is_reference(target: QwenCodecSample, reference: QwenCodecSample) -> bool:
    return (
        reference.speaker_id == target.speaker_id
        and reference.text_index != target.text_index
        and reference.utterance_id != target.utterance_id
        and reference.text != target.text
    )


def _is_reference_info(
    target: _QwenCodecSampleInfo,
    reference: _QwenCodecSampleInfo,
) -> bool:
    return (
        reference.speaker_id == target.speaker_id
        and reference.text_index != target.text_index
        and reference.utterance_id != target.utterance_id
        and reference.text != target.text
    )


def _index(value: int, *, size: int, source: str) -> int:
    index = operator.index(value)
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise IndexError(f"{source} index out of range.")
    return index


def _sample_count(value: int, *, source: QwenCodecColumnDataset) -> int:
    if isinstance(value, bool):
        raise TypeError("sample_count must be an integer or None.")
    size = operator.index(value)
    if size < 1:
        raise ValueError("sample_count must be positive.")
    return min(size, len(source))


__all__ = [
    "QwenCodecColumnDataset",
    "QwenCodecPairDataset",
    "QwenCodecPairSample",
    "QwenCodecSample",
]
