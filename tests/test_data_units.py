from __future__ import annotations

import torch
from anytrain.codec import AcousticLayout

from semantic_acoustic_codec.callback import SemanticFrameUnits
from semantic_acoustic_codec.types import SemanticCodecBatch


def test_semantic_frame_units_reports_mask_padding() -> None:
    batch = SemanticCodecBatch(
        semantic_codes=torch.tensor([[[1], [2], [8]], [[3], [8], [8]]], dtype=torch.long),
        acoustic_codes=torch.tensor([[[1], [1], [5]], [[2], [5], [5]]], dtype=torch.long),
        mask=torch.tensor([[True, True, False], [True, False, False]]),
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
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
