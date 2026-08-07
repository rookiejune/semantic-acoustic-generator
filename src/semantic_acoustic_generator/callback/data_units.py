from __future__ import annotations

from typing import Any

from anytrain.lightning.schedule import UnitBatch
from lightning import pytorch as pl

from semantic_acoustic_generator.types import GeneratorBatch


class SemanticFrameUnits:
    """Expose semantic-frame valid/padded counts to anytrain unit callbacks."""

    def __call__(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> UnitBatch:
        del trainer, pl_module, outputs, batch_idx
        if not isinstance(batch, GeneratorBatch):
            raise TypeError("SemanticFrameUnits expects a GeneratorBatch.")
        return UnitBatch(
            valid=float(batch.semantic_valid_frames),
            padded=float(batch.semantic_padded_frames),
            unit="frames",
        )


__all__ = ["SemanticFrameUnits"]
