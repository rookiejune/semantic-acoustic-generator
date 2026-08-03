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
        return UnitBatch(
            valid=float(batch.semantic_valid_frames),
            padded=float(batch.semantic_padded_frames),
            unit="frames",
        )


@dataclass(frozen=True)
class _Measurement:
    elapsed: float
    valid: float
    padded: float | None


@dataclass(frozen=True)
class _PendingCudaMeasurement:
    start: torch.cuda.Event
    end: torch.cuda.Event
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
        self._cuda_batch_started_at: torch.cuda.Event | None = None
        self._pending_cuda: deque[_PendingCudaMeasurement] = deque(maxlen=measure_window_steps)
        self._measurements: deque[_Measurement] = deque(maxlen=measure_window_steps)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, batch, batch_idx
        if self._batch_started_at is not None or self._cuda_batch_started_at is not None:
            raise RuntimeError("A training batch started before the previous batch ended.")
        if self.sync_cuda and torch.cuda.is_available() and _uses_cuda(pl_module):
            started = torch.cuda.Event(enable_timing=True)
            started.record()
            self._cuda_batch_started_at = started
        else:
            self._batch_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._batch_started_at is None and self._cuda_batch_started_at is None:
            return

        units = self.unit_provider(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
        step = int(getattr(trainer, "global_step", 0))
        should_log = step >= self.warmup_steps and step % self.log_every_n_steps == 0
        measurement = self._finish_measurement(units, should_log=should_log)
        if measurement is None:
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
        self._reset_active_timer()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._reset_active_timer()
        self._pending_cuda.clear()

    def _finish_measurement(
        self,
        units: UnitBatch,
        *,
        should_log: bool,
    ) -> _Measurement | None:
        padded = None if units.padded is None else float(units.padded)
        started = self._cuda_batch_started_at
        if started is not None:
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            self._cuda_batch_started_at = None
            self._pending_cuda.append(
                _PendingCudaMeasurement(
                    start=started,
                    end=ended,
                    valid=float(units.valid),
                    padded=padded,
                )
            )
            if not should_log:
                return None
            ended.synchronize()
            current = self._resolve_cuda_measurements()
            if current is None:
                raise RuntimeError("CUDA throughput measurement window is empty.")
            return current

        started_at = self._batch_started_at
        if started_at is None:
            raise RuntimeError("Training batch timer is not active.")
        self._batch_started_at = None
        measurement = _Measurement(
            elapsed=time.perf_counter() - started_at,
            valid=float(units.valid),
            padded=padded,
        )
        _require_positive(measurement.elapsed, "measured batch time")
        self._measurements.append(measurement)
        return measurement if should_log else None

    def _resolve_cuda_measurements(self) -> _Measurement | None:
        current: _Measurement | None = None
        while self._pending_cuda:
            pending = self._pending_cuda.popleft()
            current = _Measurement(
                elapsed=float(pending.start.elapsed_time(pending.end)) / 1000.0,
                valid=pending.valid,
                padded=pending.padded,
            )
            _require_positive(current.elapsed, "measured batch time")
            self._measurements.append(current)
        return current

    def _reset_active_timer(self) -> None:
        self._batch_started_at = None
        self._cuda_batch_started_at = None


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


def _uses_cuda(module: pl.LightningModule) -> bool:
    device = getattr(module, "device", None)
    return isinstance(device, (str, torch.device)) and torch.device(device).type == "cuda"


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
