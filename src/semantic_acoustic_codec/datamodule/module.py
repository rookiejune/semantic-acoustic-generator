from __future__ import annotations

import math
import operator
import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Protocol, cast, overload, runtime_checkable

from anydataset.dataset import IndexSelection, MapStyleABC
from anydataset.types import Role
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from lightning import pytorch as pl
from torch.utils.data import DataLoader, Dataset

from semantic_acoustic_codec.datamodule.source import (
    DataSource,
    Overlong,
    SourceAdapter,
    source_adapter,
)
from semantic_acoustic_codec.datamodule.structured import collate_structured_codes
from semantic_acoustic_codec.types import SemanticCodecBatch


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
    source: str = DataSource.QWEN_CROSS_TEXT.value
    root: str | None = None
    split: str = "train"
    validation_split: str | None = None
    validation_sample_limit: int | None = None
    role: str = "target"
    speaker_id: str = "vivian"
    sample_index: int = 0
    max_seconds: float | None = None
    overlong: str = Overlong.ERROR.value
    sample_limit: int | None = None
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    batching: BatchingConfig = field(default_factory=BatchingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError("source must be a string.")
        try:
            _ = self.source_type
        except ValueError as error:
            supported = ", ".join(repr(source.value) for source in DataSource)
            raise NotImplementedError(
                f"data.source={self.source!r} is not wired yet; supported sources are {supported}."
            ) from error
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
        if not isinstance(self.overlong, str):
            raise TypeError("overlong must be a string.")
        try:
            _ = self.overlong_policy
        except ValueError as error:
            raise ValueError("overlong must be 'error', 'filter', or 'truncate'.") from error
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

    @property
    def source_type(self) -> DataSource:
        return DataSource(self.source)

    @property
    def overlong_policy(self) -> Overlong:
        return Overlong(self.overlong)


@runtime_checkable
class _IndexOrderedDataset(Protocol):
    @property
    def index_order(self) -> MapStyleABC: ...


class _PreparedDataset(MapStyleABC):
    """Expose preplanned source indexes without retaining materialized samples."""

    def __init__(
        self,
        source: Dataset[Any],
        *,
        indexes: Sequence[int],
        costs: Sequence[int],
        shuffle_group_samples: int,
    ) -> None:
        if len(indexes) != len(costs):
            raise ValueError("prepared dataset indexes and costs must have equal length.")
        if not isinstance(source, _IndexOrderedDataset):
            raise TypeError("semantic codec source must expose a map-style index_order.")
        self.source = source
        self.indexes = tuple(indexes)
        self.index_order = IndexSelection(source.index_order, self.indexes)
        self.costs = costs
        self.shuffle_group_samples = shuffle_group_samples

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> Any:
        return self.source[self.indexes[index]]

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        yield from _coalesce_groups(
            self.index_order._shuffle(
                shuffle=shuffle,
                seed=seed,
                epoch=epoch,
                num_replicas=num_replicas,
                rank=rank,
            ),
            minimum=self.shuffle_group_samples,
        )


def _coalesce_groups(
    groups: Iterator[Sequence[int]],
    *,
    minimum: int,
) -> Iterator[Sequence[int]]:
    pending: list[int] = []
    for group in groups:
        if not group:
            continue
        if not pending and len(group) >= minimum:
            yield group
            continue
        pending.extend(group)
        if len(pending) >= minimum:
            yield tuple(pending)
            pending.clear()
    if pending:
        yield tuple(pending)


class _DurationCosts(Sequence[int]):
    """Compute duration-based planner costs without loading sample payloads."""

    def __init__(
        self,
        source: Dataset[Any],
        *,
        indexes: Sequence[int],
        adapter: SourceAdapter,
        max_frames: int | None,
        batch_frames: int,
        frame_rate: float,
        truncate: bool,
        cache_size: int,
    ) -> None:
        self.source = source
        self.indexes = indexes
        self.adapter = adapter
        self.max_frames = max_frames
        self.batch_frames = batch_frames
        self.frame_rate = frame_rate
        self.truncate = truncate
        self.cache_size = cache_size
        self._cache: OrderedDict[int, int] = OrderedDict()

    def __len__(self) -> int:
        return len(self.indexes)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        if isinstance(index, slice):
            return tuple(self[offset] for offset in range(*index.indices(len(self))))
        resolved = operator.index(index)
        if resolved < 0:
            resolved += len(self)
        if resolved < 0 or resolved >= len(self):
            raise IndexError("cost index out of range.")
        cached = self._cache.get(resolved)
        if cached is not None:
            self._cache.move_to_end(resolved)
            return cached
        duration = self.adapter.duration(self.source, self.indexes[resolved])
        frames = _planning_frames(duration, self.frame_rate)
        if self.truncate and self.max_frames is not None:
            frames = min(frames, self.max_frames)
        frames = min(frames, self.batch_frames)
        self._cache[resolved] = frames
        self._cache.move_to_end(resolved)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frames


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
        self.adapter = source_adapter(data.source_type)
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
        if stage in {None, "fit"} and self.dataset is None:
            self.dataset, self.filtered_samples = self._prepare_dataset(
                self.data,
                label="training",
            )
        if (
            stage in {None, "fit", "validate"}
            and self.data.validation_split is not None
            and self.val_dataset is None
        ):
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
        source = self.adapter.dataset(
            codec=self.codec,
            root=data.root,
            split=data.split,
            role=Role(data.role),
            speaker_id=data.speaker_id,
            sample_limit=data.sample_limit,
        )
        size = len(cast(Sized, cast(object, source)))
        sample_limit = data.sample_limit
        if sample_limit is not None:
            size = min(sample_limit, size)
        candidates = range(size)
        max_seconds = data.max_seconds
        policy = data.overlong_policy
        if policy is Overlong.FILTER and max_seconds is None:
            raise ValueError("duration filtering requires a hard limit.")
        max_frames = None if max_seconds is None else _frames(max_seconds, self.frame_rate)
        batch_frames = (
            _frames(data.batching.max_batch_seconds, self.frame_rate)
            if data.batching.enabled
            else None
        )
        if policy is not Overlong.FILTER:
            lazy_indexes = tuple(candidates)
            lazy_costs: Sequence[int]
            if batch_frames is None:
                lazy_costs = (1,) * len(lazy_indexes)
            else:
                lazy_costs = _DurationCosts(
                    source,
                    indexes=lazy_indexes,
                    adapter=self.adapter,
                    max_frames=max_frames,
                    batch_frames=batch_frames,
                    frame_rate=self.frame_rate,
                    truncate=policy is Overlong.TRUNCATE,
                    cache_size=data.batching.planning_window,
                )
            return _PreparedDataset(
                source,
                indexes=lazy_indexes,
                costs=lazy_costs,
                shuffle_group_samples=data.batch_size,
            ), 0

        indexes: list[int] = []
        costs: list[int] = []
        filtered = 0
        for index in candidates:
            frames = self.adapter.raw_length(source, index)
            if max_frames is not None and frames > max_frames:
                filtered += 1
                continue
            if batch_frames is not None:
                frames = min(frames, batch_frames)
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
        return _PreparedDataset(
            source,
            indexes=indexes,
            costs=costs,
            shuffle_group_samples=data.batch_size,
        ), filtered

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
            costs=self.dataset.costs,
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

    def val_dataloader(
        self,
    ) -> DataLoader[SemanticCodecBatch] | list[DataLoader[SemanticCodecBatch]]:
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
    adapter = source_adapter(data.source_type)
    dataset = _dataset(data, codec=codec, adapter=adapter)
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
    adapter = source_adapter(data.source_type)
    dataset = _dataset(data, codec=codec, adapter=adapter)
    return collate_samples(
        [dataset[data.sample_index]],
        data=data,
        acoustic_layout=acoustic_layout,
        frame_rate=frame_rate,
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
    )


def sample_codes(
    sample: object,
    *,
    data: DataConfig,
    frame_rate: float,
    acoustic_layout: AcousticLayout,
) -> SemanticAcousticCodes:
    value = source_adapter(data.source_type).adapt(sample).target
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
    max_seconds = data.max_seconds
    if max_seconds is None:
        return value
    max_frames = _frames(max_seconds, frame_rate)
    if value.semantic.size(0) <= max_frames:
        return value
    if data.overlong_policy is Overlong.TRUNCATE:
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
    sample: object,
    *,
    data: DataConfig,
    frame_rate: float,
    acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
) -> int:
    adapted = source_adapter(data.source_type).adapt(sample)
    target = _limit_codes(
        adapted.target,
        data=data,
        frame_rate=frame_rate,
        acoustic_layout=acoustic_layout,
    ).semantic.size(0)
    if adapted.reference is None:
        return target
    reference = _limit_codes(
        adapted.reference,
        data=data,
        frame_rate=frame_rate,
        acoustic_layout=acoustic_layout,
    ).semantic.size(0)
    return max(target, reference)


def collate_samples(
    samples: Sequence[object],
    *,
    data: DataConfig,
    acoustic_layout: AcousticLayout,
    frame_rate: float,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> SemanticCodecBatch:
    if not samples:
        raise ValueError("cannot collate an empty semantic codec batch.")
    adapter = source_adapter(data.source_type)
    adapted = tuple(adapter.adapt(sample) for sample in samples)
    target = collate_structured_codes(
        [
            _limit_codes(
                sample.target,
                data=data,
                frame_rate=frame_rate,
                acoustic_layout=acoustic_layout,
            )
            for sample in adapted
        ],
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
        acoustic_layout=acoustic_layout,
    )
    if not adapter.paired:
        return target
    pairs = tuple(sample.pair() for sample in adapted)
    reference = collate_structured_codes(
        [
            _limit_codes(
                pair[0],
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
        metadata=tuple(pair[1] for pair in pairs),
    )


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


def _frames(seconds: float, frame_rate: float) -> int:
    _positive_number(frame_rate, name="frame_rate")
    frames = round(seconds * frame_rate)
    if frames < 1:
        raise ValueError("duration limit must contain at least one codec frame.")
    return frames


def _planning_frames(duration: float, frame_rate: float) -> int:
    _positive_number(frame_rate, name="frame_rate")
    return max(1, math.ceil(duration * frame_rate))


def _positive_number(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _dataset(
    data: DataConfig,
    *,
    codec: str,
    adapter: SourceAdapter,
) -> Dataset[Any]:
    return adapter.dataset(
        codec=codec,
        root=data.root,
        split=data.split,
        role=Role(data.role),
        speaker_id=data.speaker_id,
        sample_limit=data.sample_limit,
    )


__all__ = [
    "BatchingConfig",
    "DataConfig",
    "DataModule",
    "collate_samples",
    "length",
    "load_batch",
    "load_codes",
    "sample_codes",
    "single_batch_loader",
]
