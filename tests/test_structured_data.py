from __future__ import annotations

import pytest
import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes

import semantic_acoustic_generator.types as codec_types
from semantic_acoustic_generator.datamodule import collate_structured_codes
from semantic_acoustic_generator.pl_module.semantic import GeneratorModule
from semantic_acoustic_generator.types import GeneratorBatch, PairMetadata


def test_structured_batch_rejects_fixed_length_axes() -> None:
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

    with pytest.raises(ValueError, match="frame-aligned"):
        collate_structured_codes(
            values,
            semantic_pad_id=100,
            acoustic_pad_ids=(200,),
            acoustic_layout=AcousticLayout.FIXED_LENGTH,
        )


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


def test_batch_requires_shared_target_and_reference_batch_axes() -> None:
    semantic = torch.tensor([[[1], [2]]], dtype=torch.long)
    acoustic = torch.tensor([[[3], [4]], [[5], [6]]], dtype=torch.long)
    semantic_mask = torch.ones(1, 2, dtype=torch.bool)
    acoustic_mask = torch.ones(2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="target semantic and acoustic codes must share"):
        GeneratorBatch(
            semantic_codes=semantic,
            acoustic_codes=acoustic,
            mask=semantic_mask,
            semantic_pad_id=100,
            acoustic_pad_ids=(200,),
            acoustic_mask=acoustic_mask,
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
        )

    target = collate_structured_codes(
        [SemanticAcousticCodes(semantic=semantic[0], acoustic=acoustic[0])],
        semantic_pad_id=100,
        acoustic_pad_ids=(200,),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )
    metadata = PairMetadata(
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
        GeneratorBatch(
            semantic_codes=target.semantic_codes,
            acoustic_codes=target.acoustic_codes,
            mask=target.mask,
            semantic_pad_id=target.semantic_pad_id,
            acoustic_pad_ids=target.acoustic_pad_ids,
            acoustic_mask=target.acoustic_mask,
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
                acoustic=torch.tensor([[3], [4]], dtype=torch.long),
            )
        ],
        semantic_pad_id=100,
        acoustic_pad_ids=(200,),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )

    with pytest.raises(ValueError, match="must be provided together"):
        GeneratorBatch(
            semantic_codes=target.semantic_codes,
            acoustic_codes=target.acoustic_codes,
            mask=target.mask,
            semantic_pad_id=target.semantic_pad_id,
            acoustic_pad_ids=target.acoustic_pad_ids,
            acoustic_mask=target.acoustic_mask,
            acoustic_layout=target.acoustic_layout,
            reference_semantic_codes=target.semantic_codes,
        )

    with pytest.raises(ValueError, match="one item per batch row"):
        GeneratorBatch(
            semantic_codes=target.semantic_codes,
            acoustic_codes=target.acoustic_codes,
            mask=target.mask,
            semantic_pad_id=target.semantic_pad_id,
            acoustic_pad_ids=target.acoustic_pad_ids,
            acoustic_mask=target.acoustic_mask,
            acoustic_layout=target.acoustic_layout,
            reference_semantic_codes=target.semantic_codes,
            reference_acoustic_codes=target.acoustic_codes,
            reference_mask=target.mask,
            reference_acoustic_mask=target.acoustic_mask,
        )


def test_batch_shape_validation_precedes_codebook_axis_access() -> None:
    with pytest.raises(ValueError, match="acoustic_codes must have shape"):
        GeneratorBatch(
            semantic_codes=torch.tensor([[[1]]], dtype=torch.long),
            acoustic_codes=torch.tensor(1, dtype=torch.long),
            mask=torch.ones(1, 1, dtype=torch.bool),
            semantic_pad_id=100,
            acoustic_pad_ids=(200,),
            acoustic_mask=torch.ones(1, 1, dtype=torch.bool),
        )


def test_semantic_codec_batch_to_moves_reference_tensors() -> None:
    batch = _paired_batch()

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


def test_batch_tensor_transforms_do_not_repeat_value_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _paired_batch()
    to_calls: list[dict[str, object]] = []

    def fail_validation(**_: object) -> None:
        raise AssertionError("trusted tensor transforms must not repeat batch validation")

    def fake_to(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        del args
        to_calls.append(kwargs)
        return tensor

    def fail_count(*_: object, **__: object) -> int:
        raise AssertionError("trusted tensor transforms must preserve cached frame counts")

    monkeypatch.setattr(codec_types, "_validate_side", fail_validation)
    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(torch.Tensor, "sum", fail_count)
    monkeypatch.setattr(torch.Tensor, "numel", fail_count)

    moved = batch.to(torch.device("cpu"), non_blocking=True)

    assert moved is not batch
    assert moved.metadata is batch.metadata
    assert moved.semantic_valid_frames == batch.semantic_valid_frames == 2
    assert moved.semantic_padded_frames == batch.semantic_padded_frames == 2
    assert moved.acoustic_valid_units == batch.acoustic_valid_units == 2
    assert len(to_calls) == 8
    assert all(call == {"device": torch.device("cpu"), "non_blocking": True} for call in to_calls)


def test_dataloader_pin_memory_uses_batch_tensor_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _paired_batch()
    pinned_ids: list[int] = []

    def fail_validation(**_: object) -> None:
        raise AssertionError("pinning a validated batch must not repeat batch validation")

    def fake_pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        pinned_ids.append(id(tensor))
        return tensor.clone()

    def fail_count(*_: object, **__: object) -> int:
        raise AssertionError("pinning must preserve cached frame counts")

    monkeypatch.setattr(codec_types, "_validate_side", fail_validation)
    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin_memory)
    monkeypatch.setattr(torch.Tensor, "sum", fail_count)
    monkeypatch.setattr(torch.Tensor, "numel", fail_count)

    pinned = torch.utils.data._utils.pin_memory.pin_memory(batch)

    assert isinstance(pinned, GeneratorBatch)
    assert pinned is not batch
    assert pinned.metadata is batch.metadata
    assert pinned.semantic_valid_frames == batch.semantic_valid_frames == 2
    assert pinned.semantic_padded_frames == batch.semantic_padded_frames == 2
    assert pinned.acoustic_valid_units == batch.acoustic_valid_units == 2
    assert len(pinned_ids) == 8
    assert pinned.semantic_codes is not batch.semantic_codes
    assert pinned.reference_acoustic_codes is not batch.reference_acoustic_codes


def test_lightning_transfer_requests_non_blocking_batch_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _paired_batch()
    calls: list[tuple[torch.device | str, bool]] = []

    def fake_to(
        self: GeneratorBatch,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> GeneratorBatch:
        calls.append((device, non_blocking))
        return self

    monkeypatch.setattr(GeneratorBatch, "to", fake_to)
    module = object.__new__(GeneratorModule)
    device = torch.device("cuda")

    moved = module.transfer_batch_to_device(batch, device, dataloader_idx=0)

    assert moved is batch
    assert calls == [(device, True)]


def _paired_batch() -> GeneratorBatch:
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
    metadata = PairMetadata(
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
    return GeneratorBatch(
        semantic_codes=target.semantic_codes,
        acoustic_codes=target.acoustic_codes,
        mask=target.mask,
        semantic_pad_id=target.semantic_pad_id,
        acoustic_pad_ids=target.acoustic_pad_ids,
        acoustic_mask=target.acoustic_mask,
        acoustic_layout=target.acoustic_layout,
        reference_semantic_codes=target.semantic_codes.clone(),
        reference_acoustic_codes=target.acoustic_codes.clone(),
        reference_mask=target.mask.clone(),
        reference_acoustic_mask=target.acoustic_mask.clone(),
        metadata=(metadata,),
    )
