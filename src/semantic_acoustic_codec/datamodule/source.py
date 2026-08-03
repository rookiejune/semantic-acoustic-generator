from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

from anydataset.types import Role
from anytrain.codec import SemanticAcousticCodes
from torch.utils.data import Dataset

from semantic_acoustic_codec._compat import StrEnum, auto
from semantic_acoustic_codec.datamodule.qwen import (
    QwenCodecColumnDataset,
    QwenCodecPairDataset,
    QwenCodecPairSample,
    QwenCodecSample,
)
from semantic_acoustic_codec.types import SemanticCodecPairMetadata


class DataSource(StrEnum):
    QWEN_FIXED_SPEAKER = auto()
    QWEN_CROSS_TEXT = auto()


class Overlong(StrEnum):
    ERROR = auto()
    FILTER = auto()
    TRUNCATE = auto()


@runtime_checkable
class _RawLengthDataset(Protocol):
    def raw_length(self, index: int) -> int: ...


@dataclass(frozen=True)
class AdaptedSample:
    """Source-neutral codec units consumed by batching and collation."""

    target: SemanticAcousticCodes
    reference: SemanticAcousticCodes | None = None
    metadata: SemanticCodecPairMetadata | None = None

    def __post_init__(self) -> None:
        if (self.reference is None) != (self.metadata is None):
            raise ValueError("reference units and pair metadata must be provided together.")

    @property
    def raw_length(self) -> int:
        target = self.target.semantic.size(0)
        if self.reference is None:
            return target
        return max(target, self.reference.semantic.size(0))

    def pair(self) -> tuple[SemanticAcousticCodes, SemanticCodecPairMetadata]:
        if self.reference is None or self.metadata is None:
            raise RuntimeError("paired source adapter must provide reference units and metadata.")
        return self.reference, self.metadata


class SourceAdapter(ABC):
    """Adapt one configured dataset source into the shared codec sample contract."""

    source: ClassVar[DataSource]
    paired: ClassVar[bool]

    @abstractmethod
    def dataset(
        self,
        *,
        codec: str,
        root: str | None,
        split: str,
        role: Role,
        speaker_id: str,
        sample_limit: int | None,
    ) -> Dataset[Any]:
        raise NotImplementedError

    @abstractmethod
    def adapt(self, sample: object) -> AdaptedSample:
        raise NotImplementedError

    def raw_length(self, dataset: Dataset[Any], index: int) -> int:
        value = (
            dataset.raw_length(index)
            if isinstance(dataset, _RawLengthDataset)
            else self.adapt(dataset[index]).raw_length
        )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("dataset raw_length() must return an integer.")
        if value < 1:
            raise ValueError("dataset raw_length() must be positive.")
        return value


class _QwenAdapter(SourceAdapter):
    def _column(
        self,
        *,
        codec: str,
        root: str | None,
        split: str,
        role: Role,
        speaker_id: str,
    ) -> QwenCodecColumnDataset:
        return QwenCodecColumnDataset(
            codec=codec,
            root=None if root is None else Path(root).expanduser(),
            split=split,
            role=role,
            speaker_id=speaker_id,
        )


class _QwenFixedSpeakerAdapter(_QwenAdapter):
    source = DataSource.QWEN_FIXED_SPEAKER
    paired = False

    def dataset(
        self,
        *,
        codec: str,
        root: str | None,
        split: str,
        role: Role,
        speaker_id: str,
        sample_limit: int | None,
    ) -> Dataset[Any]:
        del sample_limit
        return cast(
            Dataset[Any],
            self._column(
                codec=codec,
                root=root,
                split=split,
                role=role,
                speaker_id=speaker_id,
            ),
        )

    def adapt(self, sample: object) -> AdaptedSample:
        if not isinstance(sample, QwenCodecSample):
            raise TypeError("qwen_fixed_speaker samples must use QwenCodecSample.")
        return AdaptedSample(target=sample.codes)


class _QwenCrossTextAdapter(_QwenAdapter):
    source = DataSource.QWEN_CROSS_TEXT
    paired = True

    def dataset(
        self,
        *,
        codec: str,
        root: str | None,
        split: str,
        role: Role,
        speaker_id: str,
        sample_limit: int | None,
    ) -> Dataset[Any]:
        column = self._column(
            codec=codec,
            root=root,
            split=split,
            role=role,
            speaker_id=speaker_id,
        )
        return cast(
            Dataset[Any],
            QwenCodecPairDataset(column, sample_count=sample_limit),
        )

    def adapt(self, sample: object) -> AdaptedSample:
        if not isinstance(sample, QwenCodecPairSample):
            raise TypeError("qwen_cross_text samples must use QwenCodecPairSample.")
        return AdaptedSample(
            target=sample.target.codes,
            reference=sample.reference.codes,
            metadata=_metadata(sample),
        )


_ADAPTERS: dict[DataSource, SourceAdapter] = {
    adapter.source: adapter
    for adapter in (
        _QwenFixedSpeakerAdapter(),
        _QwenCrossTextAdapter(),
    )
}


def source_adapter(source: DataSource | str) -> SourceAdapter:
    return _ADAPTERS[DataSource(source)]


def _metadata(pair: QwenCodecPairSample) -> SemanticCodecPairMetadata:
    return SemanticCodecPairMetadata(
        target_index=pair.target_index,
        reference_index=pair.reference_index,
        target_text_index=pair.target.text_index,
        reference_text_index=pair.reference.text_index,
        target_source_index=pair.target.source_index,
        reference_source_index=pair.reference.source_index,
        target_role=pair.target.role.value,
        reference_role=pair.reference.role.value,
        target_utterance_id=pair.target.utterance_id,
        reference_utterance_id=pair.reference.utterance_id,
        target_speaker_id=pair.target.speaker_id,
        reference_speaker_id=pair.reference.speaker_id,
        target_text=pair.target.text,
        reference_text=pair.reference.text,
    )


__all__ = [
    "AdaptedSample",
    "DataSource",
    "Overlong",
    "SourceAdapter",
    "source_adapter",
]
