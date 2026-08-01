from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from anydataset.dataset.speaker import SpeakerAudioGrid, SpeakerAudioRow
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)
from anytrain.codec import AcousticLayout, SemanticAcousticCodes

from semantic_acoustic_codec.datamodule import BatchingConfig, DataConfig, DataModule
from semantic_acoustic_codec.datamodule import qwen as qwen_data


def test_data_config_defaults_to_cross_text_grid_column() -> None:
    data = DataConfig()

    assert data.source == "qwen_cross_text"
    assert data.role == "target"
    assert data.speaker_id == "vivian"
    assert data.validation_split is None
    with pytest.raises(ValueError, match="requires validation_split"):
        DataConfig(validation_sample_limit=2)


def test_qwen_fixed_speaker_source_batches_fixed_length_units(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        [
            ("hello", torch.tensor([[1], [2], [3]]), torch.tensor([[4], [5], [6], [7]])),
        ]
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    data = DataConfig(
        source="qwen_fixed_speaker",
        root=str(tmp_path / "prepared"),
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        batching=BatchingConfig(enabled=False),
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
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    data = DataConfig(
        source="qwen_cross_text",
        root=str(tmp_path / "prepared"),
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
        batching=BatchingConfig(enabled=False),
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


def test_qwen_cross_text_uses_anydataset_cost_batching(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        [
            (
                f"text {index}",
                torch.ones(2, 1, dtype=torch.long),
                torch.ones(4, 1, dtype=torch.long),
            )
            for index in range(4)
        ]
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    data = DataConfig(
        source="qwen_cross_text",
        root=str(tmp_path / "prepared"),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(
            max_batch_seconds=2.0,
            planning_window=4,
        ),
    )
    module = _module(data, tmp_path, frame_rate=2.0)
    module.trainer = SimpleNamespace(current_epoch=3)

    module.setup()
    loader = module.train_dataloader()
    batches = list(loader)

    assert type(loader).__module__ == "anydataset.dataset.batching"
    assert loader.batch_sampler.epoch == 3
    assert [len(batch.metadata) for batch in batches] == [2, 2]
    assert sorted(
        item.target_index
        for batch in batches
        for item in batch.metadata
    ) == [0, 1, 2, 3]


def test_dynamic_batch_planning_does_not_load_qwen_codec_samples(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        [
            (
                f"text {index}",
                torch.ones(index + 2, 1, dtype=torch.long),
                torch.ones(4, 1, dtype=torch.long),
            )
            for index in range(4)
        ]
    )
    loads: list[AudioView] = []
    original_select = SpeakerAudioGrid.select

    class TrackingSelection:
        def __init__(self, selection) -> None:
            self.selection = selection

        def load(self, *, view=None):
            loads.append(view)
            return self.selection.load(view=view)

    def select(self, *, text=None, speaker=None):
        selection = original_select(self, text=text, speaker=speaker)
        if self is grid:
            return TrackingSelection(selection)
        return selection

    monkeypatch.setattr(SpeakerAudioGrid, "select", select)
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    data = DataConfig(
        source="qwen_cross_text",
        root=str(tmp_path / "prepared"),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(max_batch_seconds=2.0, planning_window=4),
    )
    module = _module(data, tmp_path, frame_rate=2.0)

    module.setup()

    assert loads == []
    next(iter(module.train_dataloader()))
    assert loads


def test_dynamic_batch_budget_does_not_create_hard_duration_limit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        [
            (
                "long text",
                torch.ones(4, 1, dtype=torch.long),
                torch.ones(4, 1, dtype=torch.long),
            )
        ]
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    data = DataConfig(
        source="qwen_fixed_speaker",
        root=str(tmp_path / "prepared"),
        max_seconds=None,
        overlong="truncate",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(max_batch_seconds=1.0, planning_window=1),
    )
    module = _module(data, tmp_path, frame_rate=1.0)

    module.setup()
    assert module.dataset is not None
    assert module.dataset.costs == (1,)
    batch = next(iter(module.train_dataloader()))

    assert batch.semantic_codes.shape == (1, 4, 1)
    assert batch.mask.tolist() == [[True, True, True, True]]


def test_qwen_pair_dataset_construction_does_not_load_codec_samples() -> None:
    codes = SemanticAcousticCodes(
        semantic=torch.ones(2, 1, dtype=torch.long),
        acoustic=torch.ones(2, 1, dtype=torch.long),
    )
    samples = [
        qwen_data.QwenCodecSample(
            index=index,
            text_index=index,
            source_index=index,
            role=Role.TARGET,
            utterance_id=f"utterance-{index}",
            speaker_id="vivian",
            text=f"text {index}",
            codes=codes,
        )
        for index in range(2)
    ]

    class CountingSource:
        def __init__(self) -> None:
            self.loads: list[int] = []

        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> qwen_data.QwenCodecSample:
            self.loads.append(index)
            return samples[index]

    source = CountingSource()
    pairs = qwen_data.QwenCodecPairDataset(source)  # type: ignore[arg-type]

    assert source.loads == []
    pair = pairs[0]
    assert source.loads == [0, 1]
    assert pair.target.text != pair.reference.text


def test_qwen_pair_sample_count_fences_reference_pool() -> None:
    codes = SemanticAcousticCodes(
        semantic=torch.ones(2, 1, dtype=torch.long),
        acoustic=torch.ones(2, 1, dtype=torch.long),
    )
    samples = [
        qwen_data.QwenCodecSample(
            index=index,
            text_index=index,
            source_index=index,
            role=Role.TARGET,
            utterance_id=f"utterance-{index}",
            speaker_id="vivian",
            text=f"text {index}",
            codes=codes,
        )
        for index in range(4)
    ]

    class CountingSource:
        def __init__(self) -> None:
            self.loads: list[int] = []

        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> qwen_data.QwenCodecSample:
            self.loads.append(index)
            return samples[index]

    source = CountingSource()
    pairs = qwen_data.QwenCodecPairDataset(source, sample_count=2)  # type: ignore[arg-type]

    pair = pairs[1]

    assert len(pairs) == 2
    assert pair.target_index == 1
    assert pair.reference_index == 0
    assert source.loads == [1, 0]


def test_datamodule_exposes_deterministic_held_out_split(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grids = {
        "train": _grid(
            [
                ("train zero", torch.ones(2, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
                ("train one", torch.ones(2, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
            ]
        ),
        "heldout": _grid(
            [
                ("heldout zero", torch.ones(2, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
                ("heldout one", torch.ones(2, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
            ]
        ),
    }
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **kwargs: SimpleNamespace(load=lambda: grids[kwargs["split"]]),
    )
    data = DataConfig(
        source="qwen_cross_text",
        root=str(tmp_path / "prepared"),
        validation_split="heldout",
        validation_sample_limit=2,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(enabled=False),
    )
    module = _module(data, tmp_path)

    module.setup("fit")
    first = next(iter(module.val_dataloader()))
    second = next(iter(module.val_dataloader()))

    assert module.validation_data is not None
    assert module.validation_data.split == "heldout"
    assert [item.target_text for item in first.metadata] == [
        "heldout zero",
        "heldout one",
    ]
    assert first.metadata == second.metadata


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
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    data = DataConfig(
        source="qwen_cross_text",
        root=str(tmp_path / "prepared"),
        max_seconds=1.0,
        overlong="filter",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(enabled=False),
    )
    module = _module(data, tmp_path, frame_rate=2.0)

    with pytest.warns(UserWarning, match="filtered 3"):
        module.setup()
    batch = next(iter(module.train_dataloader()))
    fixed = module.sample_batch()

    assert module.filtered_samples == 3
    assert len(batch.metadata) == 1
    assert batch.metadata[0].target_text == "text zero"
    assert batch.metadata[0].reference_text == "text one"
    assert fixed.metadata == batch.metadata


def _module(data: DataConfig, tmp_path, *, frame_rate: float = 50.0) -> DataModule:
    return DataModule(
        data,
        codec="bicodec",
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
        frame_rate=frame_rate,
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
