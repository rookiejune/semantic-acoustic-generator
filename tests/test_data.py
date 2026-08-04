from __future__ import annotations

import pytest
import torch

try:
    from anydataset.types import AudioItem, AudioView, Modality, Role
except TypeError as exc:
    if "SPEAKER_ID" not in str(exc):
        raise
    pytest.skip(
        "anydataset TextMeta currently defines SPEAKER_ID twice; data tests require that third_party fix.",
        allow_module_level=True,
    )

from semantic_acoustic_generator.backend.longcat import batch_samples, codes, split_codes
from semantic_acoustic_generator.types import GeneratorBatch


def sample(value: torch.Tensor):
    return {
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: value},
        )
    }


def test_longcat_codes_split_target_sample() -> None:
    value = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    semantic, acoustic = split_codes(codes(sample(value)))

    assert torch.equal(semantic, torch.tensor([[1], [4]]))
    assert torch.equal(acoustic, torch.tensor([[2, 3], [5, 6]]))


def test_collate_pads_right_side_only() -> None:
    batch = batch_samples(
        [
            sample(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)),
            sample(torch.tensor([[7, 8, 9]], dtype=torch.long)),
        ],
        semantic_pad_id=10,
        acoustic_pad_ids=(11, 12),
    )

    assert isinstance(batch, GeneratorBatch)
    assert batch.semantic_codes.tolist() == [[[1], [4]], [[7], [10]]]
    assert batch.acoustic_codes.tolist() == [[[2, 3], [5, 6]], [[8, 9], [11, 12]]]
    assert batch.mask.tolist() == [[True, True], [True, False]]
    assert batch.acoustic_mask.tolist() == [[True, True], [True, False]]
    assert batch.semantic_pad_id == 10
    assert batch.acoustic_pad_ids == (11, 12)


def test_rejects_missing_acoustic_codebook() -> None:
    with pytest.raises(ValueError, match="semantic and acoustic"):
        codes(sample(torch.tensor([[1], [2]], dtype=torch.long)))
