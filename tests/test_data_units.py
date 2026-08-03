from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from anytrain.codec import AcousticLayout
from anytrain.lightning.schedule import UnitBatch

from semantic_acoustic_codec.callback import SemanticFrameUnits, UnitThroughputCallback
from semantic_acoustic_codec.types import SemanticCodecBatch


def test_semantic_frame_units_reports_mask_padding() -> None:
    batch = SemanticCodecBatch(
        semantic_codes=torch.tensor([[[1], [2], [8]], [[3], [8], [8]]], dtype=torch.long),
        acoustic_codes=torch.tensor([[[1], [1], [5]], [[2], [5], [5]]], dtype=torch.long),
        mask=torch.tensor([[True, True, False], [True, False, False]]),
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
        acoustic_mask=torch.tensor([[True, True, False], [True, False, False]]),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )

    units = SemanticFrameUnits()(
        trainer=None,  # type: ignore[arg-type]
        pl_module=None,  # type: ignore[arg-type]
        outputs=None,
        batch=batch,
        batch_idx=0,
    )

    assert units.unit == "frames"
    assert units.valid == 3.0
    assert units.padded == 6.0


def test_semantic_frame_units_uses_cached_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch()

    def fail(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("SemanticFrameUnits must not reduce the device mask.")

    monkeypatch.setattr(batch.mask, "sum", fail)

    units = SemanticFrameUnits()(
        trainer=None,  # type: ignore[arg-type]
        pl_module=None,  # type: ignore[arg-type]
        outputs=None,
        batch=batch,
        batch_idx=0,
    )

    assert units.valid == 3.0
    assert units.padded == 6.0


def test_cuda_throughput_synchronizes_only_when_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[_FakeEvent] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def event(*, enable_timing: bool) -> _FakeEvent:
        assert enable_timing is True
        value = _FakeEvent(timestamp_ms=float(len(events) * 10))
        events.append(value)
        return value

    monkeypatch.setattr(torch.cuda, "Event", event)
    callback = UnitThroughputCallback(
        unit_provider=lambda **_: UnitBatch(valid=5, padded=8, unit="frames"),
        log_every_n_steps=2,
        warmup_steps=0,
        measure_window_steps=2,
    )
    trainer: Any = SimpleNamespace(global_step=1)
    module: Any = _LogModule()

    callback.on_train_batch_start(trainer, module, None, 0)
    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert sum(item.synchronize_calls for item in events) == 0
    assert module.metrics == []

    trainer.global_step = 2
    callback.on_train_batch_start(trainer, module, None, 1)
    callback.on_train_batch_end(trainer, module, None, None, 1)

    assert sum(item.synchronize_calls for item in events) == 1
    assert events[-1].synchronize_calls == 1
    assert len(module.metrics) == 1
    metrics = module.metrics[0]
    assert metrics["data/frames_per_second"] == pytest.approx(500.0)
    assert metrics["data/frames_per_second_window"] == pytest.approx(500.0)
    assert metrics["data/frames_padding_ratio"] == pytest.approx(3 / 8)


def test_throughput_uses_cpu_timer_for_cpu_module_with_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_event(*, enable_timing: bool) -> None:
        del enable_timing
        raise AssertionError("CPU training must not initialize CUDA events.")

    monkeypatch.setattr(torch.cuda, "Event", fail_event)
    callback = UnitThroughputCallback(
        unit_provider=lambda **_: UnitBatch(valid=2, padded=2, unit="frames"),
        log_every_n_steps=1,
        warmup_steps=0,
        measure_window_steps=1,
    )
    trainer: Any = SimpleNamespace(global_step=1)
    module: Any = _LogModule()
    module.device = torch.device("cpu")

    callback.on_train_batch_start(trainer, module, None, 0)
    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert len(module.metrics) == 1


class _FakeEvent:
    def __init__(self, *, timestamp_ms: float) -> None:
        self.timestamp_ms = timestamp_ms
        self.record_calls = 0
        self.synchronize_calls = 0

    def record(self) -> None:
        self.record_calls += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def elapsed_time(self, end: _FakeEvent) -> float:
        return end.timestamp_ms - self.timestamp_ms


class _LogModule:
    def __init__(self) -> None:
        self.device = torch.device("cuda")
        self.metrics: list[dict[str, float]] = []

    def log_dict(self, metrics: dict[str, float], **kwargs: Any) -> None:
        assert kwargs == {
            "on_step": True,
            "on_epoch": False,
            "logger": True,
            "sync_dist": False,
        }
        self.metrics.append(metrics)


def _batch() -> SemanticCodecBatch:
    return SemanticCodecBatch(
        semantic_codes=torch.tensor([[[1], [2], [8]], [[3], [8], [8]]], dtype=torch.long),
        acoustic_codes=torch.tensor([[[1], [1], [5]], [[2], [5], [5]]], dtype=torch.long),
        mask=torch.tensor([[True, True, False], [True, False, False]]),
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
        acoustic_mask=torch.tensor([[True, True, False], [True, False, False]]),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )
