from __future__ import annotations

from typing import Any

import torch
from anytrain.lightning import DataUnits
from lightning import pytorch as pl

from semantic_acoustic_codec.types import SemanticCodecBatch


class SemanticFrameUnits:
    """Expose semantic-frame valid/padded counts for DataThroughputCallback."""

    def __call__(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> DataUnits:
        del trainer, pl_module, outputs, batch_idx
        if not isinstance(batch, SemanticCodecBatch):
            raise TypeError("DataThroughput expects a SemanticCodecBatch.")
        mask = batch.mask
        if mask.dtype != torch.bool:
            raise TypeError("SemanticCodecBatch.mask must be boolean.")
        return DataUnits(
            valid=float(mask.sum().item()),
            padded=float(mask.numel()),
            unit="frames",
        )
