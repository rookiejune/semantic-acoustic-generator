from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_acoustic_generator.config import (
    AnchorTarget,
    DecoderConfig,
    FactorPredictor,
    FMMode,
)
from semantic_acoustic_generator.model import FMFeatureGenerator
from semantic_acoustic_generator.model import rvq as rvq_module
from semantic_acoustic_generator.model.rvq import FactorDepthPredictor


class _CausalDepthCore(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(1, hidden_dim)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del return_dict
        if past_key_values is None:
            hidden = inputs_embeds.cumsum(dim=1)
        else:
            hidden = inputs_embeds + past_key_values[:, -1:]
        cache = hidden[:, -1:] if use_cache else None
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)


def _codebooks(stages: int = 2) -> tuple[torch.Tensor, ...]:
    sizes = (3, 4, 5, 6, 7, 8)[: stages * 2]
    return tuple(
        torch.arange(size * 8, dtype=torch.float32).reshape(size, 8) / (index + 1)
        for index, size in enumerate(sizes)
    )


def _predictor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stages: int = 2,
) -> FactorDepthPredictor:
    monkeypatch.setattr(
        rvq_module,
        "_qwen3_model",
        lambda **options: _CausalDepthCore(options["hidden_dim"]),
    )
    return FactorDepthPredictor(
        6,
        _codebooks(stages),
        hidden_dim=8,
        layers=1,
        heads=1,
        ffn_ratio=2,
    )


def _recurrent_predictor(*, stages: int = 2) -> FactorDepthPredictor:
    return FactorDepthPredictor(
        6,
        _codebooks(stages),
        hidden_dim=8,
        layers=2,
        heads=1,
        ffn_ratio=2,
        recurrent=True,
    )


def test_factor_depth_packs_valid_frames_and_preserves_factor_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _predictor(monkeypatch)
    condition = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    targets = torch.tensor(
        [
            [[0, 1, 2, 3], [2, 3, 4, 5], [-1, -1, -1, -1]],
            [[1, 2, 3, 4], [-1, -1, -1, -1], [-1, -1, -1, -1]],
        ]
    )

    packed = predictor.forward_packed(condition, targets, mask=mask)
    padded = predictor(condition, targets, mask=mask)

    assert packed.labels is not None
    assert torch.equal(
        packed.labels,
        torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5], [1, 2, 3, 4]]),
    )
    assert torch.equal(packed.row_indices, torch.tensor([0, 0, 1]))
    assert [tuple(value.shape) for value in packed.logits] == [
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
    ]
    for packed_value, padded_value in zip(packed.logits, padded):
        assert torch.equal(padded_value[mask], packed_value)
        assert torch.equal(padded_value[~mask], torch.zeros_like(padded_value[~mask]))


def test_factor_depth_teacher_forces_only_the_preceding_stage_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _predictor(monkeypatch)
    condition = torch.randn(1, 2, 6)
    mask = torch.ones(1, 2, dtype=torch.bool)
    targets = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
    baseline = predictor(condition, targets, mask=mask)

    current_changed = targets.clone()
    current_changed[..., 2] = 3
    current_changed[..., 3] = 4
    current_logits = predictor(condition, current_changed, mask=mask)
    assert all(torch.equal(left, right) for left, right in zip(baseline, current_logits))

    prior_a_changed = targets.clone()
    prior_a_changed[..., 0] = 2
    prior_a_logits = predictor(condition, prior_a_changed, mask=mask)
    assert torch.equal(baseline[0], prior_a_logits[0])
    assert torch.equal(baseline[1], prior_a_logits[1])
    assert not torch.equal(baseline[2], prior_a_logits[2])
    assert not torch.equal(baseline[3], prior_a_logits[3])

    prior_b_changed = targets.clone()
    prior_b_changed[..., 1] = 3
    prior_b_logits = predictor(condition, prior_b_changed, mask=mask)
    assert torch.equal(baseline[0], prior_b_logits[0])
    assert torch.equal(baseline[1], prior_b_logits[1])
    assert not torch.equal(baseline[2], prior_b_logits[2])
    assert not torch.equal(baseline[3], prior_b_logits[3])


def test_factor_heads_in_each_stage_share_one_hidden_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _predictor(monkeypatch)
    condition = torch.randn(1, 2, 6)
    targets = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
    captured: list[list[torch.Tensor]] = [[], [], [], []]
    handles = []
    for factor, head in enumerate(predictor.heads):
        handles.append(
            head.register_forward_pre_hook(
                lambda _module, inputs, factor=factor: captured[factor].append(inputs[0].detach())
            )
        )

    predictor(condition, targets)
    for handle in handles:
        handle.remove()

    assert torch.equal(captured[0][0], captured[1][0])
    assert torch.equal(captured[2][0], captured[3][0])


def test_factor_depth_generation_uses_both_greedy_factors_for_the_next_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _predictor(monkeypatch)
    with torch.no_grad():
        for head in predictor.heads:
            head.weight.zero_()
            head.bias.zero_()
        predictor.heads[0].bias[1] = 5
        predictor.heads[1].bias[2] = 5
        predictor.heads[2].bias[3] = 5
        predictor.heads[3].bias[4] = 5

    captured: list[torch.Tensor] = []
    handle = predictor.previous_stage[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    condition = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    generated = predictor.generate(condition, mask=mask)
    handle.remove()

    expected_codes = torch.tensor([1, 2, 3, 4]).expand(int(mask.sum()), -1)
    assert torch.equal(generated[mask], expected_codes)
    assert torch.equal(generated[~mask], torch.zeros_like(generated[~mask]))
    expected_embedding = torch.cat(
        (predictor.factor_codebooks[0][1], predictor.factor_codebooks[1][2])
    ).expand(int(mask.sum()), -1)
    torch.testing.assert_close(captured[0], expected_embedding)


def test_recurrent_factor_depth_preserves_stage_causality_and_generation() -> None:
    predictor = _recurrent_predictor()
    condition = torch.randn(1, 2, 6)
    targets = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
    baseline = predictor(condition, targets)

    changed = targets.clone()
    changed[..., :2] = torch.tensor([2, 3])
    logits = predictor(condition, changed)

    assert torch.equal(baseline[0], logits[0])
    assert torch.equal(baseline[1], logits[1])
    assert not torch.equal(baseline[2], logits[2])
    assert not torch.equal(baseline[3], logits[3])
    generated = predictor.generate(condition)
    assert generated.shape == targets.shape
    assert predictor.decoder is None
    assert len(predictor.recurrent_blocks) == 2
    assert isinstance(predictor.recurrent_output_norm, nn.LayerNorm)


def test_recurrent_factor_depth_retargets_labels_from_generated_prefix() -> None:
    predictor = _recurrent_predictor()
    with torch.no_grad():
        for head in predictor.heads:
            head.weight.zero_()
            head.bias.zero_()
        predictor.heads[0].bias[1] = 5
        predictor.heads[1].bias[2] = 5
    condition = torch.randn(1, 2, 6)
    targets = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
    prefixes: list[torch.Tensor] = []

    def targeter(stage: int, prefix: torch.Tensor) -> torch.Tensor:
        assert stage == 1
        prefixes.append(prefix.detach().clone())
        return torch.tensor([3, 4]).expand(prefix.size(0), -1)

    packed = predictor.forward_packed_retargeted(condition, targets, targeter)

    assert packed.labels is not None
    assert torch.equal(packed.labels, torch.tensor([[0, 0, 3, 4], [1, 1, 3, 4]]))
    assert torch.equal(prefixes[0], torch.tensor([[1, 2], [1, 2]]))


def test_factor_depth_backpropagates_through_all_parallel_heads_and_moves_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _predictor(monkeypatch).double()
    condition = torch.randn(1, 2, 6, dtype=torch.float64)
    targets = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])

    logits = predictor(condition, targets)
    sum(value.mean() for value in logits).backward()

    assert all(value.dtype == torch.float64 for value in logits)
    assert all(value.dtype == torch.float64 for value in predictor.factor_codebooks)
    assert all(head.weight.grad is not None for head in predictor.heads)
    assert predictor.previous_stage[0][0].weight.grad is not None


def test_factor_depth_validates_paired_codebooks_and_flat_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rvq_module,
        "_qwen3_model",
        lambda **options: _CausalDepthCore(options["hidden_dim"]),
    )
    with pytest.raises(ValueError, match="one A/B pair per stage"):
        FactorDepthPredictor(6, _codebooks()[:3], hidden_dim=8, layers=1, heads=1)

    predictor = _predictor(monkeypatch, stages=1)
    with pytest.raises(ValueError, match=r"\[B, F, 2 \* stages\]"):
        predictor(torch.randn(1, 2, 6), torch.zeros(1, 2, 1, dtype=torch.long))

    generated = predictor.generate(torch.randn(1, 2, 6))
    assert generated.shape == (1, 2, 2)


def test_factor_depth_integrates_with_aligned_factor_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rvq_module,
        "_qwen3_model",
        lambda **options: _CausalDepthCore(options["hidden_dim"]),
    )
    codebooks = _codebooks()
    generator = FMFeatureGenerator(
        6,
        32,
        DecoderConfig(
            hidden_dim=8,
            layers=1,
            heads=1,
            ffn_ratio=2,
            fm_mode=FMMode.ANCHOR,
            anchor_target=AnchorTarget.FACTOR,
            factor_predictor=FactorPredictor.DEPTH_AR,
            anchor_hidden_dim=8,
            anchor_layers=1,
        ),
        factor_codebooks=codebooks,
    )
    condition = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    targets = torch.tensor(
        [
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            [[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )

    output = generator.feature_loss_from_condition(
        condition,
        mask,
        target_features=None,
        factor_targets=targets,
    )
    output.loss.backward()
    factors = generator.sample_factor_codes(condition, mask)
    features = generator.sample_features(
        condition,
        mask,
        feature_mean=torch.zeros(1, 1, 32),
        feature_std=torch.ones(1, 1, 32),
        flow_steps=1,
    )

    assert generator.factor_depth is not None
    assert output.primary == "anchor_factor"
    assert output.items["anchor_factor"].details is not None
    assert "codebook_3_top1" in output.items["anchor_factor"].details
    assert factors.shape == (2, 3, 4)
    assert features.shape == (2, 3, 32)
    assert torch.equal(factors[~mask], torch.zeros_like(factors[~mask]))
    assert torch.equal(features[~mask], torch.zeros_like(features[~mask]))
    assert all(head.weight.grad is not None for head in generator.factor_depth.heads)
    assert "factor_codebook_1_b" in generator.state_dict()
    assert "factor_depth.factor_codebook_1_b" in generator.state_dict()


def test_factor_depth_loss_keeps_valid_frames_packed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rvq_module,
        "_qwen3_model",
        lambda **options: _CausalDepthCore(options["hidden_dim"]),
    )
    generator = FMFeatureGenerator(
        6,
        32,
        DecoderConfig(
            hidden_dim=8,
            layers=1,
            heads=1,
            ffn_ratio=2,
            fm_mode=FMMode.ANCHOR,
            anchor_target=AnchorTarget.FACTOR,
            factor_predictor=FactorPredictor.DEPTH_AR,
            anchor_hidden_dim=8,
            anchor_layers=1,
        ),
        factor_codebooks=_codebooks(),
    )
    condition = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    targets = torch.tensor(
        [
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            [[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    monkeypatch.setattr(
        rvq_module,
        "_scatter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected scatter")),
    )

    output = generator.feature_loss_from_condition(
        condition,
        mask,
        target_features=None,
        factor_targets=targets,
        include_details=False,
    )

    assert output.loss.isfinite()
    assert output.items["anchor_factor"].details is None
