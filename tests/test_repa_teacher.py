from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_acoustic_generator.loss.repa import (
    WavLMTeacher,
    _require_prefix_mask,
    _truncate_wavlm_encoder,
    decode_group_metrics,
)


class RecordingCodec:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(codes.shape))
        # Deterministic, non-constant content so normalization keeps signal.
        time = codes.size(1) * 4
        base = torch.arange(time, dtype=torch.float32).view(1, 1, -1)
        scale = codes[:, :, 0].float().mean(dim=1).view(-1, 1, 1) + 1.0
        return (base * scale).expand(codes.size(0), 1, time).contiguous()


class FakeWavLM(nn.Module):
    class config:
        hidden_size = 4
        num_hidden_layers = 2
        conv_kernel = [4, 2]
        conv_stride = [2, 2]

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
            for _ in range(self.config.num_hidden_layers)
        )

    @classmethod
    def from_pretrained(cls, checkpoint: str) -> FakeWavLM:
        del checkpoint
        return cls()

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states
        hidden = inputs.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        return SimpleNamespace(
            hidden_states=tuple(hidden for _ in range(self.config.num_hidden_layers + 1))
        )


def _teacher(codec: RecordingCodec) -> WavLMTeacher:
    # Bypass HuggingFace download; only exercise decode batching + alignment.
    module = WavLMTeacher.__new__(WavLMTeacher)
    nn.Module.__init__(module)
    module.codec = codec
    module.layer = 1
    module.sample_rate = codec.sample_rate
    module.model = FakeWavLM()
    return module


@pytest.fixture
def teacher() -> tuple[WavLMTeacher, RecordingCodec]:
    codec = RecordingCodec()
    return _teacher(codec), codec


def test_teacher_truncates_wavlm_after_requested_layer() -> None:
    model = FakeWavLM()

    _truncate_wavlm_encoder(model, 1)

    assert len(model.encoder.layers) == 1
    assert model.config.num_hidden_layers == 2


def test_require_prefix_mask_rejects_holes() -> None:
    with pytest.raises(ValueError, match="contiguous right padding"):
        _require_prefix_mask(torch.tensor([[True, False, True]]))


def test_decode_group_metrics_reports_singleton_degeneration() -> None:
    mixed = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, False, False, False],
        ]
    )
    metrics = decode_group_metrics(mixed)
    assert metrics["decode_groups"] == 3.0
    assert metrics["decode_group_mean"] == pytest.approx(4 / 3)
    assert metrics["decode_group_max"] == 2.0
    assert metrics["decode_singleton_fraction"] == pytest.approx(2 / 3)

    singles = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
        ]
    )
    assert decode_group_metrics(singles)["decode_singleton_fraction"] == 1.0


def test_waveforms_batches_equal_lengths(teacher: tuple[WavLMTeacher, RecordingCodec]) -> None:
    module, codec = teacher
    semantic = torch.tensor([[[1], [2], [3]], [[4], [5], [6]]], dtype=torch.long)
    acoustic = torch.ones(2, 3, 2, dtype=torch.long)
    mask = torch.ones(2, 3, dtype=torch.bool)

    inputs, lengths = module._waveforms(semantic, acoustic, mask)

    assert codec.calls == [(2, 3, 3)]
    assert tuple(inputs.shape) == (2, 12)
    assert torch.equal(lengths, torch.tensor([12, 12]))


def test_waveforms_groups_by_frame_length(teacher: tuple[WavLMTeacher, RecordingCodec]) -> None:
    module, codec = teacher
    semantic = torch.tensor(
        [[[1], [2], [3], [0]], [[4], [5], [0], [0]], [[6], [7], [8], [0]]],
        dtype=torch.long,
    )
    acoustic = torch.zeros(3, 4, 2, dtype=torch.long)
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    inputs, lengths = module._waveforms(semantic, acoustic, mask)

    assert set(codec.calls) == {(2, 3, 3), (1, 2, 3)}
    assert torch.equal(lengths, torch.tensor([12, 8, 12]))
    assert inputs.shape[0] == 3
    # Equal-length rows share one decode; row 0 and 2 both length 3.
    assert torch.allclose(inputs[0, :12], inputs[0, :12])
    assert inputs[1, 8:].abs().sum() == 0


def test_waveforms_matches_per_row_decode(teacher: tuple[WavLMTeacher, RecordingCodec]) -> None:
    module, codec = teacher
    semantic = torch.tensor([[[1], [2], [0]], [[3], [0], [0]]], dtype=torch.long)
    acoustic = torch.tensor([[[7, 8], [9, 10], [0, 0]], [[11, 12], [0, 0], [0, 0]]])
    mask = torch.tensor([[True, True, False], [True, False, False]])

    batched, batched_lengths = module._waveforms(semantic, acoustic, mask)
    codec.calls.clear()

    expected = []
    for row, valid in enumerate(mask):
        codes = torch.cat((semantic[row, valid], acoustic[row, valid]), dim=-1)
        wave = codec.decode(codes[None]).float().squeeze()
        wave = (wave - wave.mean()) / torch.sqrt(wave.var(unbiased=False) + 1e-7)
        expected.append(wave)

    assert torch.equal(batched_lengths, torch.tensor([w.numel() for w in expected]))
    for row, wave in enumerate(expected):
        assert torch.allclose(batched[row, : wave.numel()], wave)


def test_forward_returns_aligned_features(teacher: tuple[WavLMTeacher, RecordingCodec]) -> None:
    module, _codec = teacher
    semantic = torch.tensor([[[1], [2], [0]], [[3], [4], [5]]], dtype=torch.long)
    acoustic = torch.ones(2, 3, 2, dtype=torch.long)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    features = module(semantic, acoustic, mask)

    assert features.shape == (2, 3, module.feature_dim)
    assert torch.count_nonzero(features[0, 2]) == 0
    assert torch.count_nonzero(features[0, :2]) > 0
