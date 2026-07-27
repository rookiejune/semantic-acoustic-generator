from __future__ import annotations

import pytest
import torch
from anydataset.dataset import SpeakerAudioGrid, SpeakerAudioRow
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)
from anytrain.codec import AcousticLayout

from semantic_acoustic_codec.datamodule import DataConfig, DataModule, LBAConfig
from semantic_acoustic_codec.datamodule import qwen as qwen_data


def test_data_config_defaults_to_cross_text_grid_column() -> None:
    data = DataConfig()

    assert data.source == "qwen_cross_text"
    assert data.role == "target"
    assert data.speaker_id == "vivian"


def test_qwen_fixed_speaker_source_batches_fixed_length_units(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        [
            ("hello", torch.tensor([[1], [2], [3]]), torch.tensor([[4], [5], [6], [7]])),
        ]
    )
    monkeypatch.setattr(qwen_data, "qwen_tts_speaker_codec_grid", lambda **_: grid)
    data = DataConfig(
        source="qwen_fixed_speaker",
        root=str(tmp_path / "prepared"),
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        lba=LBAConfig(enabled=False),
    )
    module = _module(data, tmp_path)

    module.setup()
    batch = next(iter(module.train_dataloader()))

    assert batch.acoustic_layout is AcousticLayout.FIXED_LENGTH
    assert batch.semantic_codes.tolist() == [[[1], [2], [3]]]
    assert batch.acoustic_codes.tolist() == [[[4], [5], [6], [7]]]
    assert batch.mask.tolist() == [[True, True, True]]
    assert batch.acoustic_mask.tolist() == [[True, True, True, True]]


def test_qwen_cross_text_source_batches_explicit_pair_and_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        [
            ("target text", torch.tensor([[1], [2]]), torch.tensor([[4], [5], [6], [7]])),
            ("reference text", torch.tensor([[2], [3], [4]]), torch.tensor([[8], [9], [10]])),
        ]
    )
    monkeypatch.setattr(qwen_data, "qwen_tts_speaker_codec_grid", lambda **_: grid)
    data = DataConfig(
        source="qwen_cross_text",
        root=str(tmp_path / "prepared"),
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
        lba=LBAConfig(enabled=False),
    )
    module = _module(data, tmp_path)

    module.setup()
    batch = next(iter(module.train_dataloader()))

    assert batch.has_reference
    assert len(batch.metadata) == 2
    pairs = {
        (item.target_text_index, item.reference_text_index)
        for item in batch.metadata
    }
    assert pairs == {(0, 1), (1, 0)}
    assert all(
        item.target_speaker_id == item.reference_speaker_id == "vivian"
        for item in batch.metadata
    )
    assert {
        (item.target_text, item.reference_text) for item in batch.metadata
    } == {
        ("target text", "reference text"),
        ("reference text", "target text"),
    }


def test_qwen_cross_text_filter_checks_target_and_reference_raw_lengths(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        [
            ("text zero", torch.ones(2, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
            ("text one", torch.ones(2, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
            ("text two", torch.ones(4, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
            ("text three", torch.ones(4, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
        ]
    )
    monkeypatch.setattr(qwen_data, "qwen_tts_speaker_codec_grid", lambda **_: grid)
    data = DataConfig(
        source="qwen_cross_text",
        root=str(tmp_path / "prepared"),
        max_seconds=1.0,
        overlong="filter",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        lba=LBAConfig(enabled=False),
    )
    module = _module(data, tmp_path, frame_rate=2.0)

    with pytest.warns(UserWarning, match="filtered 3"):
        module.setup()
    batch = next(iter(module.train_dataloader()))

    assert module.filtered_samples == 3
    assert len(batch.metadata) == 1
    assert batch.metadata[0].target_text == "text zero"
    assert batch.metadata[0].reference_text == "text one"


def _module(data: DataConfig, tmp_path, *, frame_rate: float = 50.0) -> DataModule:
    return DataModule(
        data,
        codec="bicodec",
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
        frame_rate=frame_rate,
        output_dir=tmp_path / "out",
        semantic_pad_id=10,
        acoustic_pad_ids=(20,),
    )


def _grid(
    values: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> SpeakerAudioGrid:
    cells = [
        {
            (Role.DEFAULT, Modality.TEXT): TextItem(
                views={TextView.TEXT: text, TextView.SPEAKERS: "vivian"},
                meta={TextMeta.SOURCE_INDEX: index},
            ),
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={
                    AudioView.BICODEC: {
                        "semantic": semantic,
                        "acoustic": acoustic,
                    }
                }
            ),
        }
        for index, (text, semantic, acoustic) in enumerate(values)
    ]
    return SpeakerAudioGrid(
        cells,
        ("vivian",),
        row_specs=tuple(
            SpeakerAudioRow(source_index=index, role=Role.TARGET)
            for index in range(len(cells))
        ),
    )
