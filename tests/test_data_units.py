from __future__ import annotations

from typing import Any

import pytest
import torch
from anytrain.codec import AcousticLayout

from semantic_acoustic_generator.callback import SemanticFrameUnits
from semantic_acoustic_generator.types import GeneratorBatch


def test_semantic_frame_units_reports_mask_padding() -> None:
    batch = GeneratorBatch(
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


def _batch() -> GeneratorBatch:
    return GeneratorBatch(
        semantic_codes=torch.tensor([[[1], [2], [8]], [[3], [8], [8]]], dtype=torch.long),
        acoustic_codes=torch.tensor([[[1], [1], [5]], [[2], [5], [5]]], dtype=torch.long),
        mask=torch.tensor([[True, True, False], [True, False, False]]),
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
        acoustic_mask=torch.tensor([[True, True, False], [True, False, False]]),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )
