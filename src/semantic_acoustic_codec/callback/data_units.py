from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from anytrain.lightning.schedule import UnitBatch, UnitProvider
from lightning import pytorch as pl

from semantic_acoustic_codec.types import SemanticCodecBatch


class SemanticFrameUnits:
    """Expose semantic-frame valid/padded counts for unit-aware callbacks."""

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
        if not isinstance(batch, SemanticCodecBatch):
            raise TypeError("SemanticFrameUnits expects a SemanticCodecBatch.")
        mask = batch.mask
        if mask.dtype != torch.bool:
            raise TypeError("SemanticCodecBatch.mask must be boolean.")
        return UnitBatch(
            valid=float(mask.sum().item()),
            padded=float(mask.numel()),
            unit="frames",
        )


@dataclass(frozen=True)
class _Measurement:
    elapsed: float
    valid: float
    padded: float | None


class UnitThroughputCallback(pl.Callback):
    """Log valid-unit throughput and padding ratios for train batches."""

    def __init__(
        self,
        *,
        unit_provider: UnitProvider,
        log_every_n_steps: int = 100,
        warmup_steps: int = 20,
        measure_window_steps: int = 100,
        sync_cuda: bool = True,
    ) -> None:
        super().__init__()
        _require_positive_int(log_every_n_steps, "log_every_n_steps")
        _require_non_negative_int(warmup_steps, "warmup_steps")
        _require_positive_int(measure_window_steps, "measure_window_steps")
        if not callable(unit_provider):
            raise TypeError("unit_provider must be callable.")
        if not isinstance(sync_cuda, bool):
            raise TypeError("sync_cuda must be a boolean.")
        self.unit_provider = unit_provider
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self.measure_window_steps = measure_window_steps
        self.sync_cuda = sync_cuda
        self._batch_started_at: float | None = None
        self._measurements: deque[_Measurement] = deque(maxlen=measure_window_steps)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if self._batch_started_at is not None:
            raise RuntimeError("A training batch started before the previous batch ended.")
        _sync_cuda(self.sync_cuda)
        self._batch_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._batch_started_at is None:
            return

        _sync_cuda(self.sync_cuda)
        elapsed = time.perf_counter() - self._batch_started_at
        self._batch_started_at = None
        _require_positive(elapsed, "measured batch time")

        units = self.unit_provider(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
        measurement = _Measurement(
            elapsed=elapsed,
            valid=float(units.valid),
            padded=None if units.padded is None else float(units.padded),
        )
        self._measurements.append(measurement)

        step = int(getattr(trainer, "global_step", 0))
        if step < self.warmup_steps or step % self.log_every_n_steps != 0:
            return
        pl_module.log_dict(
            _metrics(units.unit, current=measurement, window=self._measurements),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._batch_started_at = None

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._batch_started_at = None


def _metrics(
    unit: str,
    *,
    current: _Measurement,
    window: deque[_Measurement],
) -> dict[str, float]:
    window_elapsed = math.fsum(item.elapsed for item in window)
    window_valid = math.fsum(item.valid for item in window)
    metrics = {
        f"data/{unit}_per_second": current.valid / current.elapsed,
        f"data/{unit}_per_second_window": window_valid / window_elapsed,
    }
    if current.padded is not None:
        metrics[f"data/{unit}_padding_ratio"] = _padding_ratio(
            valid=current.valid,
            padded=current.padded,
        )
    window_padded = _window_padded(window)
    if window_padded is not None:
        metrics[f"data/{unit}_padding_ratio_window"] = _padding_ratio(
            valid=window_valid,
            padded=window_padded,
        )
    return metrics


def _window_padded(window: deque[_Measurement]) -> float | None:
    total = 0.0
    for item in window:
        if item.padded is None:
            return None
        total += item.padded
    return total


def _padding_ratio(*, valid: float, padded: float) -> float:
    if padded <= 0:
        raise ValueError("padded unit count must be positive.")
    if valid > padded:
        raise ValueError(f"valid unit count ({valid}) cannot exceed padded count ({padded}).")
    return 1.0 - valid / padded


def _sync_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
