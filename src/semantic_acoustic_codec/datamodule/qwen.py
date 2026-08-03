from __future__ import annotations

import math
import operator
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)
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
    raw_length: int


@dataclass(frozen=True)
class _LoadedCell:
    info: _QwenCodecSampleInfo
    codec_view: object


_INFO_CACHE_SIZE = 1024
_REFERENCE_CACHE_SIZE = 1024
_T = TypeVar("_T")


@runtime_checkable
class _CostRowDataset(Protocol):
    def cost_row(self, index: int) -> object: ...


@runtime_checkable
class _CostRow(Protocol):
    def item(self, ref: tuple[Role, Modality]) -> object: ...


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
        self.view = AudioView.BICODEC if self.codec is qwen_tts.Codec.BICODEC else AudioView.LONGCAT
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
        self._info_cache: OrderedDict[int, _QwenCodecSampleInfo] = OrderedDict()

    def __len__(self) -> int:
        return len(self.text_indices)

    def __getitem__(self, index: int) -> QwenCodecSample:
        return self._sample(self._read(index))

    def _sample(self, cell: _LoadedCell) -> QwenCodecSample:
        info = cell.info
        return QwenCodecSample(
            index=info.index,
            text_index=info.text_index,
            source_index=info.source_index,
            role=info.role,
            utterance_id=info.utterance_id,
            speaker_id=info.speaker_id,
            text=info.text,
            codes=_codes(cell.codec_view, self.codec),
        )

    def info(self, index: int) -> _QwenCodecSampleInfo:
        index = _index(index, size=len(self), source="Qwen codec column sample")
        cached = self._info_cache.get(index)
        if cached is not None:
            self._info_cache.move_to_end(index)
            return cached
        return self._read(index).info

    def _read(self, index: int) -> _LoadedCell:
        index = _index(index, size=len(self), source="Qwen codec column sample")
        text_index = self.text_indices[index]
        text_item, audio_item = self._items(self._cell(text_index))
        codec_view = audio_item.views.get(self.view)
        info = self._sample_info(index, text_index, text_item, codec_view)
        self._cache_info(info)
        return _LoadedCell(info=info, codec_view=codec_view)

    def raw_length(self, index: int) -> int:
        return self.info(index).raw_length

    def duration(self, index: int) -> float:
        index = _index(index, size=len(self), source="Qwen codec column sample")
        cells = self.grid.cells
        if not isinstance(cells, _CostRowDataset):
            raise TypeError("Qwen codec grid cells must expose metadata-only cost_row().")
        row = cells.cost_row(self._flat_index(self.text_indices[index]))
        if not isinstance(row, _CostRow):
            raise TypeError("Qwen codec cost rows must expose item().")
        item = row.item(self.grid.audio_ref)
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("Qwen codec cost row is missing audio metadata.")
        metadata = item[1]
        if not isinstance(metadata, Mapping):
            raise TypeError("Qwen codec cost-row audio metadata must be a mapping.")
        value = metadata.get(AudioMeta.DURATION.value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Qwen codec cost-row audio duration must be a number.")
        duration = float(value)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Qwen codec cost-row audio duration must be finite and positive.")
        return duration

    def _sample_info(
        self,
        index: int,
        text_index: int,
        text_item: TextItem,
        codec_view: object,
    ) -> _QwenCodecSampleInfo:
        row = self.grid.row_specs[text_index]
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
            raw_length=_raw_length(codec_view, self.codec),
        )

    def _items(self, cell: Mapping[Any, Any]) -> tuple[TextItem, AudioItem]:
        text = cell.get(self.grid.text_ref)
        if not isinstance(text, TextItem):
            raise TypeError("Qwen codec grid text cell must contain a TextItem.")
        audio = cell.get(self.grid.audio_ref)
        if not isinstance(audio, AudioItem):
            raise TypeError("Qwen codec grid audio cell must contain an AudioItem.")
        audio_speaker = audio.meta.get(AudioMeta.SPEAKER_ID)
        if audio_speaker is not None and audio_speaker != self.speaker_id:
            raise ValueError(
                f"Qwen codec audio cell has speaker {audio_speaker!r}; "
                f"expected {self.speaker_id!r}."
            )
        return text, audio

    def _cache_info(self, info: _QwenCodecSampleInfo) -> None:
        self._info_cache[info.index] = info
        self._info_cache.move_to_end(info.index)
        if len(self._info_cache) > _INFO_CACHE_SIZE:
            self._info_cache.popitem(last=False)

    def _cell(self, text_index: int) -> Mapping[Any, Any]:
        return self.grid.cells[self._flat_index(text_index)]

    def _flat_index(self, text_index: int) -> int:
        return text_index * len(self.grid.speaker_ids) + self.speaker_index


class QwenCodecPairDataset(Dataset[QwenCodecPairSample]):
    """Pair every column sample with the next valid cross-text sample."""

    def __init__(self, source: QwenCodecColumnDataset, *, sample_count: int | None = None) -> None:
        super().__init__()
        self.source = source
        size = len(source) if sample_count is None else _sample_count(sample_count, source=source)
        if size < 2:
            raise ValueError("cross-text pairing requires at least two Qwen codec samples.")
        self.size = size
        self._reference_indices: OrderedDict[int, int] = OrderedDict()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> QwenCodecPairSample:
        index = _index(index, size=len(self), source="Qwen codec pair sample")
        target_cell = self.source._read(index)
        reference_index = self._reference_indices.get(index)
        if reference_index is None:
            reference_index, reference_cell = _find_reference(
                target_cell.info,
                size=len(self),
                load=self.source._read,
                describe=_cell_info,
            )
            self._cache_reference(index, reference_index)
        else:
            self._reference_indices.move_to_end(index)
            reference_cell = self.source._read(reference_index)
        target = self.source._sample(target_cell)
        reference = self.source._sample(reference_cell)
        return QwenCodecPairSample(
            target_index=index,
            reference_index=reference_index,
            target=target,
            reference=reference,
        )

    def raw_length(self, index: int) -> int:
        index = _index(index, size=len(self), source="Qwen codec pair sample")
        target = self.source.info(index)
        reference_index = self._reference_indices.get(index)
        if reference_index is None:
            reference_index, reference = _find_reference(
                target,
                size=len(self),
                load=self.source.info,
                describe=_info_value,
            )
            self._cache_reference(index, reference_index)
        else:
            self._reference_indices.move_to_end(index)
            reference = self.source.info(reference_index)
        return max(target.raw_length, reference.raw_length)

    def duration(self, index: int) -> float:
        index = _index(index, size=len(self), source="Qwen codec pair sample")
        # The worker may skip duplicate-text candidates; planning only needs a stable proxy.
        reference_index = (index + 1) % len(self)
        return max(
            self.source.duration(index),
            self.source.duration(reference_index),
        )

    def _cache_reference(self, index: int, reference_index: int) -> None:
        self._reference_indices[index] = reference_index
        self._reference_indices.move_to_end(index)
        if len(self._reference_indices) > _REFERENCE_CACHE_SIZE:
            self._reference_indices.popitem(last=False)


def _codes(
    value: object,
    codec: qwen_tts.Codec,
) -> SemanticAcousticCodes:
    if codec is qwen_tts.Codec.LONGCAT:
        _raw_length(value, codec)
        combined = cast(Tensor, value).contiguous()
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
        _raw_length(value, codec)
        if acoustic.dim() != 2 or acoustic.size(0) < 1:
            raise ValueError("Qwen BiCodec acoustic view must expose [unit, codebook].")
        return SemanticAcousticCodes(
            semantic=semantic.contiguous(),
            acoustic=acoustic.contiguous(),
        )
    raise ValueError(f"unsupported Qwen codec: {codec!r}.")


def _raw_length(value: object, codec: qwen_tts.Codec) -> int:
    if codec is qwen_tts.Codec.LONGCAT:
        if not isinstance(value, Tensor) or value.dim() != 2:
            raise ValueError("Qwen LongCat view must expose [time, codebook].")
        return _positive_length(value.size(0))
    if codec is qwen_tts.Codec.BICODEC:
        if not isinstance(value, Mapping):
            raise TypeError("Qwen BiCodec view must be a semantic/acoustic mapping.")
        semantic = cast(Mapping[str, Any], value)["semantic"]
        if not isinstance(semantic, Tensor) or semantic.dim() != 2:
            raise ValueError("Qwen BiCodec semantic view must expose [time, codebook].")
        return _positive_length(semantic.size(0))
    raise ValueError(f"unsupported Qwen codec: {codec!r}.")


def _positive_length(value: int) -> int:
    if value < 1:
        raise ValueError("Qwen codec length metadata must be positive.")
    return value


def _find_reference(
    target: _QwenCodecSampleInfo,
    *,
    size: int,
    load: Callable[[int], _T],
    describe: Callable[[_T], _QwenCodecSampleInfo],
) -> tuple[int, _T]:
    for offset in range(1, size):
        reference_index = (target.index + offset) % size
        value = load(reference_index)
        if _is_reference_info(target, describe(value)):
            return reference_index, value
    raise ValueError(
        f"Qwen codec sample {target.utterance_id!r} has no same-speaker "
        "reference from a different text row with a different utterance id and text."
    )


def _cell_info(value: _LoadedCell) -> _QwenCodecSampleInfo:
    return value.info


def _info_value(value: _QwenCodecSampleInfo) -> _QwenCodecSampleInfo:
    return value


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
