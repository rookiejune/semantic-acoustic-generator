from __future__ import annotations

import pytest
import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes

from semantic_acoustic_codec.datamodule import collate_structured_codes
from semantic_acoustic_codec.types import SemanticCodecBatch, SemanticCodecPairMetadata


def test_fixed_length_structured_batch_keeps_independent_axes() -> None:
    values = [
        SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2], [3]], dtype=torch.long),
            acoustic=torch.tensor([[4], [5], [6], [7]], dtype=torch.long),
        ),
        SemanticAcousticCodes(
            semantic=torch.tensor([[8], [9]], dtype=torch.long),
            acoustic=torch.tensor([[10], [11], [12], [13]], dtype=torch.long),
        ),
    ]

    batch = collate_structured_codes(
        values,
        semantic_pad_id=100,
        acoustic_pad_ids=(200,),
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
    )

    assert batch.acoustic_layout is AcousticLayout.FIXED_LENGTH
    assert batch.semantic_codes.tolist() == [[[1], [2], [3]], [[8], [9], [100]]]
    assert batch.acoustic_codes.tolist() == [
        [[4], [5], [6], [7]],
        [[10], [11], [12], [13]],
    ]
    assert batch.mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.acoustic_mask is not None
    assert batch.acoustic_mask.tolist() == [[True, True, True, True], [True, True, True, True]]


def test_frame_aligned_structured_batch_rejects_axis_mismatch() -> None:
    values = [
        SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2]], dtype=torch.long),
            acoustic=torch.tensor([[3]], dtype=torch.long),
        )
    ]

    with pytest.raises(ValueError, match="frame-aligned"):
        collate_structured_codes(
            values,
            semantic_pad_id=100,
            acoustic_pad_ids=(200,),
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
        )


def test_fixed_length_batch_requires_shared_target_and_reference_batch_axes() -> None:
    semantic = torch.tensor([[[1], [2]]], dtype=torch.long)
    acoustic = torch.tensor([[[3]], [[4]]], dtype=torch.long)
    semantic_mask = torch.ones(1, 2, dtype=torch.bool)
    acoustic_mask = torch.ones(2, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="target semantic and acoustic codes must share"):
        SemanticCodecBatch(
            semantic_codes=semantic,
            acoustic_codes=acoustic,
            mask=semantic_mask,
            semantic_pad_id=100,
            acoustic_pad_ids=(200,),
            acoustic_mask=acoustic_mask,
            acoustic_layout=AcousticLayout.FIXED_LENGTH,
        )

    target = collate_structured_codes(
        [SemanticAcousticCodes(semantic=semantic[0], acoustic=acoustic[0])],
        semantic_pad_id=100,
        acoustic_pad_ids=(200,),
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
    )
    metadata = SemanticCodecPairMetadata(
        target_index=0,
        reference_index=1,
        target_text_index=0,
        reference_text_index=1,
        target_source_index=0,
        reference_source_index=1,
        target_role="target",
        reference_role="target",
        target_utterance_id="target",
        reference_utterance_id="reference",
        target_speaker_id="speaker",
        reference_speaker_id="speaker",
        target_text="target text",
        reference_text="reference text",
    )
    with pytest.raises(ValueError, match="reference semantic and acoustic codes must share"):
        SemanticCodecBatch(
            semantic_codes=target.semantic_codes,
            acoustic_codes=target.acoustic_codes,
            mask=target.mask,
            semantic_pad_id=target.semantic_pad_id,
            acoustic_pad_ids=target.acoustic_pad_ids,
            acoustic_mask=target.target_acoustic_mask,
            acoustic_layout=target.acoustic_layout,
            reference_semantic_codes=semantic,
            reference_acoustic_codes=acoustic,
            reference_mask=semantic_mask,
            reference_acoustic_mask=acoustic_mask,
            metadata=(metadata,),
        )


def test_reference_batch_requires_complete_units_and_row_metadata() -> None:
    target = collate_structured_codes(
        [
            SemanticAcousticCodes(
                semantic=torch.tensor([[1], [2]], dtype=torch.long),
                acoustic=torch.tensor([[3]], dtype=torch.long),
            )
        ],
        semantic_pad_id=100,
        acoustic_pad_ids=(200,),
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
    )

    with pytest.raises(ValueError, match="must be provided together"):
        SemanticCodecBatch(
            semantic_codes=target.semantic_codes,
            acoustic_codes=target.acoustic_codes,
            mask=target.mask,
            semantic_pad_id=target.semantic_pad_id,
            acoustic_pad_ids=target.acoustic_pad_ids,
            acoustic_mask=target.target_acoustic_mask,
            acoustic_layout=target.acoustic_layout,
            reference_semantic_codes=target.semantic_codes,
        )

    with pytest.raises(ValueError, match="one item per batch row"):
        SemanticCodecBatch(
            semantic_codes=target.semantic_codes,
            acoustic_codes=target.acoustic_codes,
            mask=target.mask,
            semantic_pad_id=target.semantic_pad_id,
            acoustic_pad_ids=target.acoustic_pad_ids,
            acoustic_mask=target.target_acoustic_mask,
            acoustic_layout=target.acoustic_layout,
            reference_semantic_codes=target.semantic_codes,
            reference_acoustic_codes=target.acoustic_codes,
            reference_mask=target.mask,
            reference_acoustic_mask=target.target_acoustic_mask,
        )


def test_batch_shape_validation_precedes_codebook_axis_access() -> None:
    with pytest.raises(ValueError, match="acoustic_codes must have shape"):
        SemanticCodecBatch(
            semantic_codes=torch.tensor([[[1]]], dtype=torch.long),
            acoustic_codes=torch.tensor(1, dtype=torch.long),
            mask=torch.ones(1, 1, dtype=torch.bool),
            semantic_pad_id=100,
            acoustic_pad_ids=(200,),
        )


def test_semantic_codec_batch_to_moves_reference_tensors() -> None:
    target = collate_structured_codes(
        [
            SemanticAcousticCodes(
                semantic=torch.tensor([[1], [2]], dtype=torch.long),
                acoustic=torch.tensor([[3], [4]], dtype=torch.long),
            )
        ],
        semantic_pad_id=100,
        acoustic_pad_ids=(200,),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )
    metadata = SemanticCodecPairMetadata(
        target_index=0,
        reference_index=1,
        target_text_index=0,
        reference_text_index=1,
        target_source_index=0,
        reference_source_index=1,
        target_role="target",
        reference_role="target",
        target_utterance_id="target",
        reference_utterance_id="reference",
        target_speaker_id="speaker",
        reference_speaker_id="speaker",
        target_text="target text",
        reference_text="reference text",
    )
    batch = SemanticCodecBatch(
        semantic_codes=target.semantic_codes,
        acoustic_codes=target.acoustic_codes,
        mask=target.mask,
        semantic_pad_id=target.semantic_pad_id,
        acoustic_pad_ids=target.acoustic_pad_ids,
        acoustic_mask=target.target_acoustic_mask,
        acoustic_layout=target.acoustic_layout,
        reference_semantic_codes=target.semantic_codes.clone(),
        reference_acoustic_codes=target.acoustic_codes.clone(),
        reference_mask=target.mask.clone(),
        reference_acoustic_mask=target.target_acoustic_mask.clone(),
        metadata=(metadata,),
    )

    moved = batch.to("cpu")

    assert moved is not batch
    assert moved.metadata == batch.metadata
    assert moved.reference_semantic_codes is not None
    assert moved.reference_acoustic_codes is not None
    assert moved.reference_mask is not None
    assert moved.reference_acoustic_mask is not None
    assert moved.reference_semantic_codes.device.type == "cpu"
    assert moved.reference_acoustic_codes.device.type == "cpu"
    assert moved.reference_mask.device.type == "cpu"
    assert moved.reference_acoustic_mask.device.type == "cpu"
