from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

from anydataset.dataset import MapStyleABC
from anydataset.types import Role
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from lightning import pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

from semantic_acoustic_codec.backend.longcat import batch_codes
from semantic_acoustic_codec.backend.longcat import codes as longcat_codes
from semantic_acoustic_codec.backend.longcat import split_codes as split_longcat_codes
from semantic_acoustic_codec.datamodule.qwen import (
    QwenCodecColumnDataset,
    QwenCodecPairDataset,
    QwenCodecPairSample,
    QwenCodecSample,
)
from semantic_acoustic_codec.datamodule.structured import collate_structured_codes
from semantic_acoustic_codec.types import SemanticCodecBatch, SemanticCodecPairMetadata


@dataclass(frozen=True)
class BatchingConfig:
    enabled: bool = True
    max_batch_seconds: float = 8.0
    planning_window: int = 256
    prefetch_factor: int = 4
    drop_distributed_tail: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("batching.enabled must be a boolean.")
        _positive_number(self.max_batch_seconds, name="max_batch_seconds")
        for name, value in (
            ("planning_window", self.planning_window),
            ("prefetch_factor", self.prefetch_factor),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not isinstance(self.drop_distributed_tail, bool):
            raise TypeError("drop_distributed_tail must be a boolean.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("batching.seed must be an integer.")


@dataclass(frozen=True)
class DataConfig:
    source: str = "qwen_cross_text"
    root: str | None = None
    split: str = "train"
    validation_split: str | None = None
    validation_sample_limit: int | None = None
    role: str = "target"
    speaker_id: str = "vivian"
    sample_index: int = 0
    max_seconds: float | None = None
    overlong: str = "error"
    sample_limit: int | None = None
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    batching: BatchingConfig = field(default_factory=BatchingConfig)

    def __post_init__(self) -> None:
        if self.source not in {"wmt19_tts_codec", "qwen_fixed_speaker", "qwen_cross_text"}:
            raise NotImplementedError(
                f"data.source={self.source!r} is not wired yet; supported sources are "
                "'wmt19_tts_codec', 'qwen_fixed_speaker', and 'qwen_cross_text'."
            )
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("split must be a non-empty string.")
        if self.validation_split is not None:
            if not isinstance(self.validation_split, str) or not self.validation_split:
                raise ValueError("validation_split must be a non-empty string or None.")
            if self.validation_split == self.split:
                raise ValueError("validation_split must differ from the training split.")
        elif self.validation_sample_limit is not None:
            raise ValueError("validation_sample_limit requires validation_split.")
        try:
            role = Role(self.role)
        except ValueError as error:
            raise ValueError("role must be 'source' or 'target'.") from error
        if role not in {Role.SOURCE, Role.TARGET}:
            raise ValueError("role must be 'source' or 'target'.")
        if not isinstance(self.speaker_id, str) or not self.speaker_id:
            raise ValueError("speaker_id must be a non-empty string.")
        if isinstance(self.sample_index, bool) or not isinstance(self.sample_index, int):
            raise TypeError("sample_index must be an integer.")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative.")
        if self.max_seconds is not None:
            _positive_number(self.max_seconds, name="max_seconds")
        if self.overlong not in {"error", "filter", "truncate"}:
            raise ValueError("overlong must be 'error', 'filter', or 'truncate'.")
        if (
            self.max_seconds is not None
            and self.batching.enabled
            and self.max_seconds > self.batching.max_batch_seconds
        ):
            raise ValueError(
                "max_seconds must not exceed batching.max_batch_seconds when batching is enabled."
            )
        for name, value, minimum in (
            ("batch_size", self.batch_size, 1),
            ("num_workers", self.num_workers, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < minimum:
                qualifier = "positive" if minimum == 1 else "non-negative"
                raise ValueError(f"{name} must be {qualifier}.")
        for name, value in (
            ("sample_limit", self.sample_limit),
            ("validation_sample_limit", self.validation_sample_limit),
        ):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer or None.")
                if value <= 0:
                    raise ValueError(f"{name} must be positive.")
        for name, value in (
            ("pin_memory", self.pin_memory),
            ("persistent_workers", self.persistent_workers),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")


class _PreparedDataset(MapStyleABC):
    """Expose preplanned source indexes without retaining materialized samples."""

    def __init__(
        self,
        source: Dataset[Any],
        *,
        indexes: Sequence[int],
        costs: Sequence[int],
    ) -> None:
        if len(indexes) != len(costs):
            raise ValueError("prepared dataset indexes and costs must have equal length.")
        self.source = source
        self.indexes = tuple(indexes)
        self.costs = tuple(costs)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> Any:
        return self.source[self.indexes[index]]

    def cost(self, index: int) -> int:
        return self.costs[index]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: DataConfig,
        *,
        codec: str,
        acoustic_layout: AcousticLayout,
        frame_rate: float,
        semantic_pad_id: int,
        acoustic_pad_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self.data = data
        self.codec = codec
        self.acoustic_layout = acoustic_layout
        self.frame_rate = frame_rate
        self.semantic_pad_id = semantic_pad_id
        self.acoustic_pad_ids = tuple(acoustic_pad_ids)
        self.dataset: _PreparedDataset | None = None
        self.val_dataset: _PreparedDataset | None = None
        self.validation_data: DataConfig | None = None
        self.filtered_samples = 0
        self.validation_filtered_samples = 0

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.dataset is None:
            self.dataset, self.filtered_samples = self._prepare_dataset(
                self.data,
                label="training",
            )
        if self.data.validation_split is not None and self.val_dataset is None:
            self.validation_data = replace(
                self.data,
                split=self.data.validation_split,
                validation_split=None,
                sample_limit=self.data.validation_sample_limit,
                validation_sample_limit=None,
            )
            self.val_dataset, self.validation_filtered_samples = self._prepare_dataset(
                self.validation_data,
                label="validation",
            )

    def _prepare_dataset(
        self,
        data: DataConfig,
        *,
        label: str,
    ) -> tuple[_PreparedDataset, int]:
        source = _dataset(data, codec=self.codec)
        size = len(cast(Sized, cast(object, source)))
        sample_limit = data.sample_limit
        if sample_limit is not None:
            size = min(sample_limit, size)
        candidates = range(size)
        indexes: list[int] = []
        costs: list[int] = []
        filtered = 0
        max_seconds = _max_seconds(data)
        if data.overlong == "filter" and max_seconds is None:
            raise ValueError("duration filtering requires a hard limit.")
        max_frames = None if max_seconds is None else _frames(max_seconds, self.frame_rate)
        inspect_lengths = data.batching.enabled or data.overlong == "filter"
        for index in candidates:
            frames = 1
            if inspect_lengths:
                frames = _raw_length(source[index], source=data.source)
                if data.overlong == "filter" and max_frames is not None and frames > max_frames:
                    filtered += 1
                    continue
                if data.overlong == "truncate" and max_frames is not None:
                    frames = min(frames, max_frames)
            indexes.append(index)
            costs.append(frames)
        if not indexes:
            raise ValueError("semantic codec duration filter removed every sample.")
        if filtered:
            if max_seconds is None:
                raise RuntimeError("duration filtering requires a hard limit.")
            warnings.warn(
                f"filtered {filtered} {label} semantic codec samples longer "
                f"than {max_seconds:g} seconds.",
                stacklevel=2,
            )
        return _PreparedDataset(source, indexes=indexes, costs=costs), filtered

    def _collate(self, data: DataConfig):
        return partial(
            collate_samples,
            data=data,
            acoustic_layout=self.acoustic_layout,
            frame_rate=self.frame_rate,
            semantic_pad_id=self.semantic_pad_id,
            acoustic_pad_ids=self.acoustic_pad_ids,
        )

    def train_dataloader(self):
        if self.dataset is None:
            raise RuntimeError("semantic codec DataModule.setup() must run first.")
        data = self.data
        batching = data.batching
        collate_fn = self._collate(data)
        persistent_workers = data.persistent_workers and data.num_workers > 0
        if not batching.enabled:
            return DataLoader(
                self.dataset,
                batch_size=data.batch_size,
                shuffle=True,
                num_workers=data.num_workers,
                pin_memory=data.pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        return self.dataset.dataloader(
            cost_fn=self.dataset.cost,
            max_batch_memory=_frames(batching.max_batch_seconds, self.frame_rate),
            max_batch_samples=data.batch_size,
            planning_window=batching.planning_window,
            drop_distributed_tail=batching.drop_distributed_tail,
            shuffle=True,
            seed=batching.seed,
            epoch=0 if self.trainer is None else self.trainer.current_epoch,
            num_workers=data.num_workers,
            pin_memory=data.pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            prefetch_factor=batching.prefetch_factor if data.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader[SemanticCodecBatch] | list[DataLoader[SemanticCodecBatch]]:
        if self.data.validation_split is None:
            return []
        if self.val_dataset is None or self.validation_data is None:
            raise RuntimeError("semantic codec DataModule.setup() must run first.")
        data = self.validation_data
        return DataLoader(
            self.val_dataset,
            batch_size=data.batch_size,
            shuffle=False,
            num_workers=data.num_workers,
            pin_memory=data.pin_memory,
            persistent_workers=data.persistent_workers and data.num_workers > 0,
            collate_fn=self._collate(data),
        )

    def feature_stats_dataloader(self) -> DataLoader[SemanticCodecBatch]:
        """Iterate the effective training subset once without shuffle or dynamic batching."""
        if self.dataset is None:
            raise RuntimeError("semantic codec DataModule.setup() must run first.")
        data = self.data
        return DataLoader(
            self.dataset,
            batch_size=data.batch_size,
            shuffle=False,
            num_workers=data.num_workers,
            pin_memory=data.pin_memory,
            persistent_workers=data.persistent_workers and data.num_workers > 0,
            collate_fn=self._collate(data),
        )

    def sample_batch(self) -> SemanticCodecBatch:
        """Load one fixed sample from the effective training subset."""
        if self.dataset is None:
            raise RuntimeError("semantic codec DataModule.setup() must run first.")
        size = len(cast(Sized, cast(object, self.dataset)))
        index = self.data.sample_index
        if index >= size:
            raise IndexError(
                f"sample_index {index} is outside the effective training subset of {size} samples."
            )
        return self._collate(self.data)([self.dataset[index]])


def load_codes(
    data: DataConfig,
    *,
    codec: str,
    frame_rate: float,
    acoustic_layout: AcousticLayout,
) -> SemanticAcousticCodes:
    dataset = _dataset(data, codec=codec)
    return sample_codes(
        dataset[data.sample_index],
        data=data,
        frame_rate=frame_rate,
        acoustic_layout=acoustic_layout,
    )


def load_batch(
    data: DataConfig,
    *,
    codec: str,
    frame_rate: float,
    acoustic_layout: AcousticLayout,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> SemanticCodecBatch:
    dataset = _dataset(data, codec=codec)
    return collate_samples(
        [dataset[data.sample_index]],
        data=data,
        acoustic_layout=acoustic_layout,
        frame_rate=frame_rate,
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
    )


def sample_codes(
    sample: Mapping[Any, Any] | QwenCodecSample | QwenCodecPairSample,
    *,
    data: DataConfig,
    frame_rate: float,
    acoustic_layout: AcousticLayout,
) -> SemanticAcousticCodes:
    value = _sample_codes(sample, source=data.source)
    return _limit_codes(
        value,
        data=data,
        frame_rate=frame_rate,
        acoustic_layout=acoustic_layout,
    )


def _limit_codes(
    value: SemanticAcousticCodes,
    *,
    data: DataConfig,
    frame_rate: float,
    acoustic_layout: AcousticLayout,
) -> SemanticAcousticCodes:
    max_seconds = _max_seconds(data)
    if max_seconds is None:
        return value
    max_frames = _frames(max_seconds, frame_rate)
    if value.semantic.size(0) <= max_frames:
        return value
    if data.overlong == "truncate":
        semantic = value.semantic[:max_frames].contiguous()
        acoustic = (
            value.acoustic[:max_frames].contiguous()
            if acoustic_layout is AcousticLayout.FRAME_ALIGNED
            else value.acoustic
        )
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)
    raise ValueError(
        f"prepared semantic sequence has {value.semantic.size(0)} frames, exceeding "
        f"the {max_frames}-frame ({max_seconds:g}s) hard limit; "
        f"overlong policy is {data.overlong!r}."
    )


def length(
    sample: Mapping[Any, Any] | QwenCodecSample | QwenCodecPairSample,
    *,
    data: DataConfig,
    frame_rate: float,
    acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
) -> int:
    target = sample_codes(
        sample,
        data=data,
        frame_rate=frame_rate,
        acoustic_layout=acoustic_layout,
    ).semantic.size(0)
    if not isinstance(sample, QwenCodecPairSample):
        return target
    reference = _limit_codes(
        sample.reference.codes,
        data=data,
        frame_rate=frame_rate,
        acoustic_layout=acoustic_layout,
    ).semantic.size(0)
    return max(target, reference)


def collate_samples(
    samples: Sequence[Mapping[Any, Any] | QwenCodecSample | QwenCodecPairSample],
    *,
    data: DataConfig,
    acoustic_layout: AcousticLayout,
    frame_rate: float,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> SemanticCodecBatch:
    if not samples:
        raise ValueError("cannot collate an empty semantic codec batch.")
    target = collate_structured_codes(
        [
            sample_codes(
                sample,
                data=data,
                frame_rate=frame_rate,
                acoustic_layout=acoustic_layout,
            )
            for sample in samples
        ],
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
        acoustic_layout=acoustic_layout,
    )
    if data.source != "qwen_cross_text":
        return target
    pairs = tuple(_pair(sample) for sample in samples)
    reference = collate_structured_codes(
        [
            _limit_codes(
                pair.reference.codes,
                data=data,
                frame_rate=frame_rate,
                acoustic_layout=acoustic_layout,
            )
            for pair in pairs
        ],
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
        acoustic_layout=acoustic_layout,
    )
    return SemanticCodecBatch(
        semantic_codes=target.semantic_codes,
        acoustic_codes=target.acoustic_codes,
        mask=target.mask,
        semantic_pad_id=target.semantic_pad_id,
        acoustic_pad_ids=target.acoustic_pad_ids,
        acoustic_mask=target.acoustic_mask,
        acoustic_layout=target.acoustic_layout,
        reference_semantic_codes=reference.semantic_codes,
        reference_acoustic_codes=reference.acoustic_codes,
        reference_mask=reference.mask,
        reference_acoustic_mask=reference.acoustic_mask,
        metadata=tuple(_metadata(pair) for pair in pairs),
    )


def collate_codes(
    values: Sequence[Tensor],
    *,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> SemanticCodecBatch:
    return batch_codes(values, semantic_pad_id=semantic_pad_id, acoustic_pad_ids=acoustic_pad_ids)


def single_batch_loader(
    value: SemanticCodecBatch,
) -> DataLoader[SemanticCodecBatch]:
    dataset = cast(Dataset[SemanticCodecBatch], cast(object, [value]))
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=_single_batch,
    )
    return cast(DataLoader[SemanticCodecBatch], cast(object, loader))


def _single_batch(values: Sequence[SemanticCodecBatch]) -> SemanticCodecBatch:
    if len(values) != 1:
        raise ValueError("single-batch loader requires exactly one batch value.")
    return values[0]


def _raw_length(
    sample: Mapping[Any, Any] | QwenCodecSample | QwenCodecPairSample,
    *,
    source: str,
) -> int:
    target = _sample_codes(sample, source=source)
    if source != "qwen_cross_text":
        return target.semantic.size(0)
    reference = _pair(sample).reference.codes.semantic.size(0)
    return max(target.semantic.size(0), reference)


def _max_seconds(data: DataConfig) -> float | None:
    if not data.batching.enabled:
        return data.max_seconds
    if data.max_seconds is None:
        return data.batching.max_batch_seconds
    return min(data.max_seconds, data.batching.max_batch_seconds)


def _frames(seconds: float, frame_rate: float) -> int:
    _positive_number(frame_rate, name="frame_rate")
    frames = round(seconds * frame_rate)
    if frames < 1:
        raise ValueError("duration limit must contain at least one codec frame.")
    return frames


def _positive_number(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _path(value: str | None) -> Path | None:
    return None if value is None else Path(value).expanduser()


def _dataset(data: DataConfig, *, codec: str) -> Dataset[Any]:
    if data.source == "wmt19_tts_codec":
        return cast(
            Dataset[Any],
            wmt19_tts_codec(
                codec="longcat",
                root=_path(data.root),
                split=data.split,
            ),
        )
    column = QwenCodecColumnDataset(
        codec=codec,
        root=_path(data.root),
        split=data.split,
        role=Role(data.role),
        speaker_id=data.speaker_id,
    )
    if data.source == "qwen_fixed_speaker":
        return cast(Dataset[Any], column)
    if data.source == "qwen_cross_text":
        return cast(
            Dataset[Any],
            QwenCodecPairDataset(column),
        )
    raise AssertionError(f"unsupported data source: {data.source}")


def _sample_codes(
    sample: Mapping[Any, Any] | QwenCodecSample | QwenCodecPairSample,
    *,
    source: str,
) -> SemanticAcousticCodes:
    if source == "wmt19_tts_codec":
        semantic, acoustic = split_longcat_codes(longcat_codes(cast(Mapping[Any, Any], sample)))
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)
    if source == "qwen_fixed_speaker":
        if not isinstance(sample, QwenCodecSample):
            raise TypeError("qwen_fixed_speaker samples must use QwenCodecSample.")
        return sample.codes
    if source == "qwen_cross_text":
        return _pair(sample).target.codes
    raise AssertionError(f"unsupported data source: {source}")


def _pair(
    sample: Mapping[Any, Any] | QwenCodecSample | QwenCodecPairSample,
) -> QwenCodecPairSample:
    if not isinstance(sample, QwenCodecPairSample):
        raise TypeError("qwen_cross_text samples must use QwenCodecPairSample.")
    return sample


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
    "BatchingConfig",
    "DataConfig",
    "DataModule",
    "collate_codes",
    "collate_samples",
    "length",
    "load_batch",
    "load_codes",
    "sample_codes",
    "single_batch_loader",
]
