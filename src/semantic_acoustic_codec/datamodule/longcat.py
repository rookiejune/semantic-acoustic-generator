from __future__ import annotations

import math
import warnings
from collections.abc import Sized
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

from lightning import pytorch as pl
from torch.utils.data import DataLoader, Dataset, Subset
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

from semantic_acoustic_codec.backend.longcat import batch_codes
from semantic_acoustic_codec.backend.longcat import codes as longcat_codes
from semantic_acoustic_codec.types import SemanticCodecBatch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal

    from torch import Tensor


@dataclass(frozen=True)
class LBAConfig:
    enabled: bool = True
    max_batch_seconds: float = 8.0
    max_padding_ratio: float = 0.05
    prefetch_batches: int = 4
    planner_mode: Literal["quality", "throughput"] = "quality"
    drop_last_flush: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("lba.enabled must be a boolean.")
        _positive_number(self.max_batch_seconds, name="max_batch_seconds")
        if isinstance(self.max_padding_ratio, bool) or not isinstance(
            self.max_padding_ratio, (int, float)
        ):
            raise TypeError("max_padding_ratio must be a number.")
        if not math.isfinite(self.max_padding_ratio) or not (0 <= self.max_padding_ratio <= 1):
            raise ValueError("max_padding_ratio must be between 0 and 1.")
        if isinstance(self.prefetch_batches, bool) or not isinstance(self.prefetch_batches, int):
            raise TypeError("prefetch_batches must be an integer.")
        if self.prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative.")
        if self.planner_mode not in {"quality", "throughput"}:
            raise ValueError("planner_mode must be 'quality' or 'throughput'.")
        if not isinstance(self.drop_last_flush, bool):
            raise TypeError("drop_last_flush must be a boolean.")


@dataclass(frozen=True)
class DataConfig:
    root: str | None = None
    split: str = "train"
    sample_index: int = 0
    max_seconds: float | None = None
    overlong: str = "error"
    sample_limit: int | None = None
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    lba: LBAConfig = field(default_factory=LBAConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("split must be a non-empty string.")
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
            and self.lba.enabled
            and self.max_seconds > self.lba.max_batch_seconds
        ):
            raise ValueError("max_seconds must not exceed lba.max_batch_seconds when LBA is enabled.")
        for name, value, minimum in (
            ("batch_size", self.batch_size, 1),
            ("num_workers", self.num_workers, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < minimum:
                qualifier = "positive" if minimum == 1 else "non-negative"
                raise ValueError(f"{name} must be {qualifier}.")
        if self.sample_limit is not None:
            if isinstance(self.sample_limit, bool) or not isinstance(self.sample_limit, int):
                raise TypeError("sample_limit must be an integer or None.")
            if self.sample_limit <= 0:
                raise ValueError("sample_limit must be positive.")
        for name, value in (
            ("pin_memory", self.pin_memory),
            ("persistent_workers", self.persistent_workers),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: DataConfig,
        *,
        frame_rate: float,
        output_dir: Path,
        semantic_pad_id: int,
        acoustic_pad_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self.data = data
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.semantic_pad_id = semantic_pad_id
        self.acoustic_pad_ids = tuple(acoustic_pad_ids)
        self.dataset: Dataset[Any] | None = None
        self.filtered_samples = 0

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.dataset is not None:
            return
        data = self.data
        dataset: Dataset[Any] = wmt19_tts_codec(
            codec="longcat",
            root=_path(data.root),
            split=data.split,
        )
        sample_limit = data.sample_limit
        if sample_limit is not None:
            dataset = Subset(dataset, range(min(sample_limit, len(dataset))))
        if data.overlong == "filter":
            dataset, self.filtered_samples = _filter(dataset, data=data, frame_rate=self.frame_rate)
            if self.filtered_samples:
                max_seconds = _max_seconds(data)
                if max_seconds is None:
                    raise RuntimeError("duration filtering requires a hard limit.")
                warnings.warn(
                    f"filtered {self.filtered_samples} semantic codec samples longer "
                    f"than {max_seconds:g} seconds.",
                    stacklevel=2,
                )
        self.dataset = dataset

    def train_dataloader(self):
        if self.dataset is None:
            raise RuntimeError("semantic codec DataModule.setup() must run first.")
        data = self.data
        lba = data.lba
        collate_fn = partial(
            collate_samples,
            data=data,
            frame_rate=self.frame_rate,
            semantic_pad_id=self.semantic_pad_id,
            acoustic_pad_ids=self.acoustic_pad_ids,
        )
        persistent_workers = data.persistent_workers and data.num_workers > 0
        if not lba.enabled:
            return DataLoader(
                self.dataset,
                batch_size=data.batch_size,
                shuffle=True,
                num_workers=data.num_workers,
                pin_memory=data.pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )

        from lba import LBA

        return LBA(
            self.dataset,
            batch_size=data.batch_size,
            shuffle=True,
            num_workers=data.num_workers,
            pin_memory=data.pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            len_fn=partial(length, data=data, frame_rate=self.frame_rate),
            max_padded_length=_frames(lba.max_batch_seconds, self.frame_rate),
            max_padding_ratio=lba.max_padding_ratio,
            prefetch_batches=lba.prefetch_batches,
            planner_mode=lba.planner_mode,
            drop_last_flush=lba.drop_last_flush,
            log_dir=self.output_dir / "lba",
        )


def load_codes(data: DataConfig, *, frame_rate: float) -> Tensor:
    dataset = wmt19_tts_codec(codec="longcat", root=_path(data.root), split=data.split)
    return sample_codes(dataset[data.sample_index], data=data, frame_rate=frame_rate)


def sample_codes(sample: Mapping[Any, Any], *, data: DataConfig, frame_rate: float) -> Tensor:
    value = longcat_codes(sample)
    max_seconds = _max_seconds(data)
    if max_seconds is None:
        return value
    max_frames = _frames(max_seconds, frame_rate)
    if value.size(0) <= max_frames:
        return value
    if data.overlong == "truncate":
        return value[:max_frames].contiguous()
    raise ValueError(
        f"prepared LongCat sequence has {value.size(0)} frames, exceeding "
        f"the {max_frames}-frame ({max_seconds:g}s) hard limit; "
        f"overlong policy is {data.overlong!r}."
    )


def length(sample: Mapping[Any, Any], *, data: DataConfig, frame_rate: float) -> int:
    return sample_codes(sample, data=data, frame_rate=frame_rate).size(0)


def collate_samples(
    samples: Sequence[Mapping[Any, Any]],
    *,
    data: DataConfig,
    frame_rate: float,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> SemanticCodecBatch:
    if not samples:
        raise ValueError("cannot collate an empty semantic codec batch.")
    return collate_codes(
        [sample_codes(sample, data=data, frame_rate=frame_rate) for sample in samples],
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
    )


def collate_codes(
    values: Sequence[Tensor],
    *,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> SemanticCodecBatch:
    return batch_codes(values, semantic_pad_id=semantic_pad_id, acoustic_pad_ids=acoustic_pad_ids)


def single_batch_loader(
    value: Tensor,
    *,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> DataLoader[SemanticCodecBatch]:
    dataset = cast(Dataset[Tensor], cast(object, [value]))
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=partial(
            collate_codes,
            semantic_pad_id=semantic_pad_id,
            acoustic_pad_ids=acoustic_pad_ids,
        ),
    )
    return cast(DataLoader[SemanticCodecBatch], cast(object, loader))


def _filter(
    dataset: Dataset[Any],
    *,
    data: DataConfig,
    frame_rate: float,
) -> tuple[Dataset[Any], int]:
    max_seconds = _max_seconds(data)
    if max_seconds is None:
        return dataset, 0
    max_frames = _frames(max_seconds, frame_rate)
    size = len(cast(Sized, cast(object, dataset)))
    indices = [index for index in range(size) if longcat_codes(dataset[index]).size(0) <= max_frames]
    dropped = size - len(indices)
    if not indices:
        raise ValueError("semantic codec duration filter removed every sample.")
    return Subset(dataset, indices), dropped


def _max_seconds(data: DataConfig) -> float | None:
    if not data.lba.enabled:
        return data.max_seconds
    if data.max_seconds is None:
        return data.lba.max_batch_seconds
    return min(data.max_seconds, data.lba.max_batch_seconds)


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


__all__ = [
    "DataConfig",
    "DataModule",
    "LBAConfig",
    "collate_codes",
    "collate_samples",
    "length",
    "load_codes",
    "sample_codes",
    "single_batch_loader",
]
