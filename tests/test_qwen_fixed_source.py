from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from anydataset.dataset import MapStyleABC
from anydataset.dataset.speaker import SpeakerAudioGrid, SpeakerAudioRow
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)
from anytrain.codec import AcousticLayout

from semantic_acoustic_generator import datamodule
from semantic_acoustic_generator.datamodule import BatchingConfig, DataConfig, DataModule
from semantic_acoustic_generator.datamodule import longcat as longcat_data
from semantic_acoustic_generator.datamodule import module as module_data
from semantic_acoustic_generator.datamodule import qwen as qwen_data
from semantic_acoustic_generator.datamodule.source import DataSource, Overlong


def test_data_config_defaults_to_cross_text_grid_column() -> None:
    data = DataConfig()

    assert data.source == "qwen_cross_text"
    assert data.source_type is DataSource.QWEN_CROSS_TEXT
    assert data.overlong_policy is Overlong.ERROR
    assert data.role == "target"
    assert data.speaker_id == "vivian"
    assert data.validation_split is None
    with pytest.raises(ValueError, match="requires validation_split"):
        DataConfig(validation_sample_limit=2)


def test_longcat_module_preserves_former_data_module_imports() -> None:
    assert datamodule.DataConfig is module_data.DataConfig
    assert datamodule.DataModule is module_data.DataModule
    assert longcat_data.DataConfig is module_data.DataConfig
    assert longcat_data.DataModule is module_data.DataModule


def test_qwen_fixed_speaker_source_batches_frame_aligned_units(
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

    assert batch.acoustic_layout is AcousticLayout.FRAME_ALIGNED
    assert batch.semantic_codes.tolist() == [[[1], [2], [3]]]
    assert batch.acoustic_codes.tolist() == [[[4], [5], [6]]]
    assert batch.mask.tolist() == [[True, True, True]]
    assert batch.acoustic_mask.tolist() == [[True, True, True]]


def test_qwen_bicodec_source_is_rejected_as_semantic_global(tmp_path) -> None:
    with pytest.raises(ValueError, match="semantic/global"):
        qwen_data.QwenCodecColumnDataset(
            codec="bicodec",
            root=tmp_path,
            split="train",
            role=Role.TARGET,
            speaker_id="vivian",
        )


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
    pairs = {(item.target_text_index, item.reference_text_index) for item in batch.metadata}
    assert pairs == {(0, 1), (1, 0)}
    assert all(
        item.target_speaker_id == item.reference_speaker_id == "vivian" for item in batch.metadata
    )
    assert {(item.target_text, item.reference_text) for item in batch.metadata} == {
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
        ],
        frame_rate=2.0,
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
    assert sorted(item.target_index for batch in batches for item in batch.metadata) == [0, 1, 2, 3]


def test_dynamic_batch_planning_does_not_load_qwen_codec_samples(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    cost_accesses: list[int] = []
    grid = _grid(
        [
            (
                f"text {index}",
                torch.ones(index + 2, 1, dtype=torch.long),
                torch.ones(4, 1, dtype=torch.long),
            )
            for index in range(4)
        ],
        frame_rate=2.0,
        accesses=accesses,
        cost_accesses=cost_accesses,
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
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(max_batch_seconds=2.0, planning_window=4),
    )
    module = _module(data, tmp_path, frame_rate=2.0)

    module.setup()

    assert accesses == []
    assert module.dataset is not None
    assert module.dataset.costs[0] == 3
    assert accesses == []
    assert cost_accesses == [0, 1]
    assert module.dataset.costs[0] == 3
    assert accesses == []
    assert cost_accesses == [0, 1]


def test_qwen_column_sample_reads_its_cell_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    grid = _grid(
        [
            (
                "single text",
                torch.tensor([[1], [2]], dtype=torch.long),
                torch.tensor([[3], [4]], dtype=torch.long),
            )
        ],
        accesses=accesses,
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    source = qwen_data.QwenCodecColumnDataset(
        codec="longcat",
        root=tmp_path,
        split="train",
        role=Role.TARGET,
        speaker_id="vivian",
    )

    sample = source[0]

    assert accesses == [0]
    assert sample.text == "single text"
    assert sample.codes.semantic.tolist() == [[1], [2]]


def test_qwen_pair_length_reuses_bounded_column_info_cache(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    grid = _grid(
        [
            ("text zero", torch.ones(2, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
            ("text one", torch.ones(3, 1, dtype=torch.long), torch.ones(4, 1, dtype=torch.long)),
        ],
        accesses=accesses,
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    source = qwen_data.QwenCodecColumnDataset(
        codec="longcat",
        root=tmp_path,
        split="train",
        role=Role.TARGET,
        speaker_id="vivian",
    )
    pairs = qwen_data.QwenCodecPairDataset(source)

    assert pairs.raw_length(0) == 3
    assert accesses == [0, 1]
    assert pairs.raw_length(1) == 3
    assert accesses == [0, 1]


def test_qwen_pair_reference_cache_is_bounded(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen_data, "_REFERENCE_CACHE_SIZE", 2)
    grid = _grid([(f"text {index}", torch.ones(2, 1), torch.ones(4, 1)) for index in range(5)])
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    source = qwen_data.QwenCodecColumnDataset(
        codec="longcat",
        root=tmp_path,
        split="train",
        role=Role.TARGET,
        speaker_id="vivian",
    )
    pairs = qwen_data.QwenCodecPairDataset(source)

    assert len(vars(pairs)["_reference_indices"]) == 0
    for index in range(5):
        pairs.raw_length(index)
    assert len(vars(pairs)["_reference_indices"]) == 2


def test_dynamic_batch_cost_cache_is_bounded_by_planning_window(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    cost_accesses: list[int] = []
    grid = _grid(
        [(f"text {index}", torch.ones(index + 1, 1), torch.ones(4, 1)) for index in range(3)],
        accesses=accesses,
        cost_accesses=cost_accesses,
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    data = DataConfig(
        source="qwen_fixed_speaker",
        root=str(tmp_path / "prepared"),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(planning_window=1),
    )
    module = _module(data, tmp_path)
    module.setup()
    assert module.dataset is not None

    costs = module.dataset.costs
    assert costs[0] == 1
    assert costs[0] == 1
    assert accesses == []
    assert cost_accesses == [0]
    assert costs[1] == 2
    assert costs[0] == 1
    assert accesses == []
    assert cost_accesses == [0, 1, 0]
    assert tuple(vars(costs)["_cache"]) == (0,)


def test_dynamic_batch_duration_proxy_counts_at_least_one_frame(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        [("short text", torch.ones(1, 1), torch.ones(4, 1))],
        frame_rate=200.0,
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
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(planning_window=1),
    )
    module = _module(data, tmp_path, frame_rate=50.0)

    module.setup()

    assert module.dataset is not None
    assert module.dataset.costs[0] == 1


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
        ],
        frame_rate=1.0,
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
    assert module.dataset.costs[0] == 1
    batch = next(iter(module.train_dataloader()))

    assert batch.semantic_codes.shape == (1, 4, 1)
    assert batch.mask.tolist() == [[True, True, True, True]]


def test_qwen_pair_dataset_construction_does_not_load_codec_samples(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    grid = _grid(
        [
            ("text zero", torch.ones(2, 1), torch.ones(4, 1)),
            ("text one", torch.ones(2, 1), torch.ones(4, 1)),
        ],
        accesses=accesses,
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    source = qwen_data.QwenCodecColumnDataset(
        codec="longcat",
        root=tmp_path,
        split="train",
        role=Role.TARGET,
        speaker_id="vivian",
    )
    pairs = qwen_data.QwenCodecPairDataset(source)

    assert accesses == []
    pair = pairs[0]
    assert accesses == [0, 1]
    assert pair.target.text != pair.reference.text


def test_qwen_pair_sample_count_fences_reference_pool(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    grid = _grid(
        [(f"text {index}", torch.ones(2, 1), torch.ones(4, 1)) for index in range(4)],
        accesses=accesses,
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    source = qwen_data.QwenCodecColumnDataset(
        codec="longcat",
        root=tmp_path,
        split="train",
        role=Role.TARGET,
        speaker_id="vivian",
    )
    pairs = qwen_data.QwenCodecPairDataset(source, sample_count=2)

    pair = pairs[1]

    assert len(pairs) == 2
    assert pair.target_index == 1
    assert pair.reference_index == 0
    assert accesses == [1, 0]


def test_qwen_column_and_pair_preserve_payload_local_index_groups_without_loading(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    shuffle_calls: list[tuple[bool, int, int, int, int]] = []
    groups = (tuple(range(8)), tuple(range(8, 16)))
    cells = _GroupedTrackingCells(
        [{} for _ in range(16)],
        accesses=accesses,
        cost_accesses=None,
        durations=(1.0,) * 16,
        groups=groups,
        shuffle_calls=shuffle_calls,
    )
    grid = SpeakerAudioGrid(
        cells,
        ("alice", "vivian"),
        row_specs=(
            SpeakerAudioRow(source_index=0, role=Role.SOURCE),
            SpeakerAudioRow(source_index=0, role=Role.TARGET),
            SpeakerAudioRow(source_index=1, role=Role.SOURCE),
            SpeakerAudioRow(source_index=1, role=Role.TARGET),
            SpeakerAudioRow(source_index=2, role=Role.SOURCE),
            SpeakerAudioRow(source_index=2, role=Role.TARGET),
            SpeakerAudioRow(source_index=3, role=Role.SOURCE),
            SpeakerAudioRow(source_index=3, role=Role.TARGET),
        ),
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )

    column = qwen_data.QwenCodecColumnDataset(
        codec="longcat",
        root=tmp_path,
        split="train",
        role=Role.TARGET,
        speaker_id="vivian",
    )
    pairs = qwen_data.QwenCodecPairDataset(column, sample_count=3)
    prepared = module_data._PreparedDataset(
        pairs,
        indexes=range(3),
        costs=(1, 1, 1),
        shuffle_group_samples=3,
    )
    column_groups = list(
        column.index_order._shuffle(
            shuffle=True,
            seed=7,
            epoch=2,
            num_replicas=1,
            rank=0,
        )
    )
    pair_groups = list(
        pairs.index_order._shuffle(
            shuffle=True,
            seed=7,
            epoch=2,
            num_replicas=1,
            rank=0,
        )
    )
    prepared_groups = list(
        prepared._shuffle(
            shuffle=True,
            seed=7,
            epoch=2,
            num_replicas=1,
            rank=0,
        )
    )

    assert column.index_order.indices == (3, 7, 11, 15)
    assert column_groups == [[0, 1], [2, 3]]
    assert pair_groups == [[0, 1], [2]]
    assert prepared_groups == [(0, 1, 2)]
    assert shuffle_calls == [
        (True, 7, 2, 1, 0),
        (True, 7, 2, 1, 0),
        (True, 7, 2, 1, 0),
    ]
    assert accesses == []


def test_qwen_column_rejects_cells_without_anydataset_index_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = SpeakerAudioGrid(
        [{}],
        ("vivian",),
        row_specs=(SpeakerAudioRow(source_index=0, role=Role.TARGET),),
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )

    with pytest.raises(TypeError, match="cells must be a MapStyleABC"):
        qwen_data.QwenCodecColumnDataset(
            codec="longcat",
            root=tmp_path,
            split="train",
            role=Role.TARGET,
            speaker_id="vivian",
        )


def test_qwen_pair_reads_candidates_once_and_materializes_only_the_pair(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    materialized: list[object] = []
    grid = _grid(
        [
            ("duplicate", torch.ones(2, 1), torch.ones(4, 1)),
            ("duplicate", torch.ones(4, 1), torch.ones(4, 1)),
            ("selected", torch.ones(3, 1), torch.ones(4, 1)),
        ],
        accesses=accesses,
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    original_codes = qwen_data._codes

    def codes(value, codec):
        materialized.append(value)
        return original_codes(value, codec)

    monkeypatch.setattr(qwen_data, "_codes", codes)
    source = qwen_data.QwenCodecColumnDataset(
        codec="longcat",
        root=tmp_path,
        split="train",
        role=Role.TARGET,
        speaker_id="vivian",
    )

    pair = qwen_data.QwenCodecPairDataset(source)[0]

    assert accesses == [0, 2]
    assert len(materialized) == 2
    assert pair.target.text == "duplicate"
    assert pair.reference.text == "selected"


def test_qwen_pair_duration_uses_the_actual_cross_text_reference_without_payload_reads(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[int] = []
    cost_accesses: list[int] = []
    grid = _grid(
        [
            ("duplicate", torch.ones(1, 1), torch.ones(4, 1)),
            ("duplicate", torch.ones(2, 1), torch.ones(4, 1)),
            ("selected", torch.ones(8, 1), torch.ones(4, 1)),
        ],
        frame_rate=1.0,
        accesses=accesses,
        cost_accesses=cost_accesses,
    )
    monkeypatch.setattr(
        qwen_data.qwen_tts,
        "speaker_grid",
        lambda **_: SimpleNamespace(load=lambda: grid),
    )
    source = qwen_data.QwenCodecColumnDataset(
        codec="longcat",
        root=tmp_path,
        split="train",
        role=Role.TARGET,
        speaker_id="vivian",
    )

    duration = qwen_data.QwenCodecPairDataset(source).duration(0)

    assert duration == 8.0
    assert accesses == []
    assert cost_accesses == [0, 2]


def test_datamodule_exposes_deterministic_held_out_split(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grids = {
        "train": _grid(
            [
                (
                    "train zero",
                    torch.ones(2, 1, dtype=torch.long),
                    torch.ones(4, 1, dtype=torch.long),
                ),
                (
                    "train one",
                    torch.ones(2, 1, dtype=torch.long),
                    torch.ones(4, 1, dtype=torch.long),
                ),
            ]
        ),
        "heldout": _grid(
            [
                (
                    "heldout zero",
                    torch.ones(2, 1, dtype=torch.long),
                    torch.ones(4, 1, dtype=torch.long),
                ),
                (
                    "heldout one",
                    torch.ones(2, 1, dtype=torch.long),
                    torch.ones(4, 1, dtype=torch.long),
                ),
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


def test_validate_setup_does_not_load_training_split(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grids = {
        "heldout": _grid(
            [
                ("heldout zero", torch.ones(2, 1), torch.ones(4, 1)),
                ("heldout one", torch.ones(2, 1), torch.ones(4, 1)),
            ]
        )
    }
    loaded_splits: list[str] = []

    def speaker_grid(**kwargs):
        split = kwargs["split"]
        loaded_splits.append(split)
        return SimpleNamespace(load=lambda: grids[split])

    monkeypatch.setattr(qwen_data.qwen_tts, "speaker_grid", speaker_grid)
    data = DataConfig(
        source="qwen_cross_text",
        root=str(tmp_path / "prepared"),
        validation_split="heldout",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        batching=BatchingConfig(enabled=False),
    )
    module = _module(data, tmp_path)

    module.setup("validate")

    assert loaded_splits == ["heldout"]
    assert module.dataset is None
    assert module.val_dataset is not None


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
        codec="longcat",
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
        frame_rate=frame_rate,
        semantic_pad_id=10,
        acoustic_pad_ids=(20,),
    )


def _grid(
    values: list[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    frame_rate: float = 50.0,
    accesses: list[int] | None = None,
    cost_accesses: list[int] | None = None,
) -> SpeakerAudioGrid:
    cells = [
        {
            (Role.DEFAULT, Modality.TEXT): TextItem(
                views={TextView.TEXT: text, TextView.SPEAKERS: "vivian"},
                meta={TextMeta.SOURCE_INDEX: index},
            ),
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={
                    AudioView.LONGCAT: _frame_aligned_codes(semantic, acoustic)
                },
                meta={
                    AudioMeta.DURATION: float(semantic.size(0)) / frame_rate,
                    AudioMeta.SPEAKER_ID: "vivian",
                },
            ),
        }
        for index, (text, semantic, acoustic) in enumerate(values)
    ]
    dataset = _TrackingCells(
        cells,
        accesses=accesses,
        cost_accesses=cost_accesses,
        durations=tuple(float(semantic.size(0)) / frame_rate for _, semantic, _ in values),
    )
    return SpeakerAudioGrid(
        dataset,
        ("vivian",),
        row_specs=tuple(
            SpeakerAudioRow(source_index=index, role=Role.TARGET) for index in range(len(cells))
        ),
    )


def _frame_aligned_codes(semantic: torch.Tensor, acoustic: torch.Tensor) -> torch.Tensor:
    if acoustic.dim() != 2 or acoustic.size(0) < 1:
        raise ValueError("test acoustic codes must have shape [frame, codebook].")
    semantic = semantic.to(dtype=torch.long)
    acoustic = acoustic.to(dtype=torch.long)
    indices = torch.arange(semantic.size(0)).remainder(acoustic.size(0))
    aligned = acoustic.index_select(0, indices)
    return torch.cat((semantic, aligned), dim=-1)


class _TrackingCells(MapStyleABC):
    def __init__(
        self,
        cells,
        *,
        accesses: list[int] | None,
        cost_accesses: list[int] | None,
        durations: tuple[float, ...],
    ) -> None:
        self.cells = cells
        self.accesses = accesses
        self.cost_accesses = cost_accesses
        self.durations = durations

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, index: int):
        if self.accesses is not None:
            self.accesses.append(index)
        return self.cells[index]

    def cost_row(self, index: int):
        if self.cost_accesses is not None:
            self.cost_accesses.append(index)
        return _CostRow(self.durations[index])

    def metadata_cell(self, index: int):
        return self.cells[index]


class _GroupedTrackingCells(_TrackingCells):
    def __init__(self, *args, groups, shuffle_calls, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.shuffle_calls = shuffle_calls

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ):
        self.shuffle_calls.append((shuffle, seed, epoch, num_replicas, rank))
        if num_replicas != 1 or rank != 0:
            raise AssertionError("IndexSelection must request the complete base ordering.")
        yield from self.groups


class _CostRow:
    def __init__(self, duration: float) -> None:
        self.duration = duration

    def item(self, ref: tuple[Role, Modality]):
        if ref != (Role.DEFAULT, Modality.AUDIO):
            return None
        return ref, {AudioMeta.DURATION.value: self.duration}
