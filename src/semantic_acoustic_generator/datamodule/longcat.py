"""LongCat collation plus compatibility exports for the former data module location."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from semantic_acoustic_generator.backend.longcat import batch_codes
from semantic_acoustic_generator.datamodule.module import (
    BatchingConfig,
    DataConfig,
    DataModule,
    collate_samples,
    length,
    load_batch,
    load_codes,
    sample_codes,
    single_batch_loader,
)
from semantic_acoustic_generator.types import GeneratorBatch


def collate_codes(
    values: Sequence[Tensor],
    *,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
) -> GeneratorBatch:
    return batch_codes(values, semantic_pad_id=semantic_pad_id, acoustic_pad_ids=acoustic_pad_ids)


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
