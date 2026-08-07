from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import lightning.pytorch as pl
from anytrain.module.quantization import codebook_usage, quantization_usage_metrics
from torch import Tensor

from semantic_acoustic_generator.types import GeneratorBatch


@dataclass(frozen=True)
class _Monitor:
    indices: Tensor
    codebook_sizes: tuple[int, ...]
    active_codebook_mask: Tensor | None = None


class CodebookUsageLogger(pl.Callback):
    """Log target semantic and acoustic codebook usage."""

    def __init__(
        self,
        *,
        every_n_steps: int | None = 100,
        tag: str = "codebook",
        log_histograms: bool = False,
    ) -> None:
        super().__init__()
        if every_n_steps is not None:
            _positive_int(every_n_steps, name="every_n_steps")
        if not isinstance(tag, str) or not tag.strip("/"):
            raise ValueError("tag must be a non-empty string.")
        if not isinstance(log_histograms, bool):
            raise TypeError("log_histograms must be a boolean.")
        self.every_n_steps = every_n_steps
        self.tag = tag.strip("/")
        self.log_histograms = log_histograms
        self._last_logged_step: int | None = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del batch_idx
        step = int(trainer.global_step)
        if (
            self.every_n_steps is not None
            and (step % self.every_n_steps != 0 or self._last_logged_step == step)
        ):
            return
        self._last_logged_step = step
        values: dict[str, Tensor] = {}
        for name, monitor in _monitors(outputs, batch, pl_module).items():
            usage = codebook_usage(
                monitor.indices,
                codebook_sizes=monitor.codebook_sizes,
                active_codebook_mask=monitor.active_codebook_mask,
            )
            if int(getattr(trainer, "world_size", 1)) > 1:
                reduce = getattr(getattr(trainer, "strategy", None), "reduce", None)
                if not callable(reduce):
                    raise RuntimeError(
                        "distributed codebook usage logging requires trainer.strategy.reduce."
                    )
                usage.counts = reduce(usage.counts, reduce_op="sum")
                usage.active_count = reduce(usage.active_count, reduce_op="sum")
                usage.total_count = reduce(usage.total_count, reduce_op="sum")
            prefix = f"{self.tag}/{name}"
            values.update(
                {
                    f"{prefix}/{metric}": value
                    for metric, value in quantization_usage_metrics(usage).items()
                }
            )
        if values:
            pl_module.log_dict(
                values,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )


def _monitors(
    outputs: object,
    batch: object,
    pl_module: pl.LightningModule,
) -> Mapping[str, _Monitor]:
    del outputs, pl_module
    if not isinstance(batch, GeneratorBatch):
        raise TypeError("CodebookUsageLogger requires a GeneratorBatch.")
    semantic_indices = batch.semantic_codes[..., 0].masked_fill(
        ~batch.mask,
        -1,
    )
    acoustic_mask = batch.acoustic_mask.unsqueeze(-1).expand_as(batch.acoustic_codes)
    acoustic_indices = batch.acoustic_codes.masked_fill(~acoustic_mask, -1)
    return {
        "semantic": _Monitor(
            indices=semantic_indices,
            codebook_sizes=(batch.semantic_pad_id,),
            active_codebook_mask=batch.mask,
        ),
        "acoustic": _Monitor(
            indices=acoustic_indices,
            codebook_sizes=batch.acoustic_pad_ids,
            active_codebook_mask=acoustic_mask,
        ),
    }


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = ["CodebookUsageLogger"]
