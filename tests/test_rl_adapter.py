from __future__ import annotations

import pytest
import torch
from anytrain.codec import AcousticLayout

from semantic_acoustic_codec.rl import SACCandidate, SACRewardBatch, SACRLAdapter, SACRollout
from semantic_acoustic_codec.types import SemanticCodecBatch


def test_sac_reward_batch_and_grpo_contract() -> None:
    adapter = SACRLAdapter()
    rollout = _rollout(batch_size=2, group_size=2)
    rewards = torch.tensor([[1.0, 2.0], [0.5, 1.5]])

    reward_batch = adapter.score(rollout, rewards=rewards)
    policy = torch.zeros(2, 2, 3)
    old = torch.full((2, 2, 3), -0.1)
    mask = torch.ones(2, 2, 3, dtype=torch.bool)

    batch = adapter.to_grpo_batch(
        reward_batch,
        policy_token_logps=policy,
        old_token_logps=old,
        response_mask=mask,
    )

    assert batch["policy_token_logps"] is policy
    assert batch["old_token_logps"] is old
    assert batch["rewards"] is rewards
    assert batch["response_mask"] is mask


def test_sac_continuous_and_neighbor_contracts() -> None:
    adapter = SACRLAdapter()
    reward_batch = SACRewardBatch(rewards=torch.ones(2, 3))
    steps = torch.zeros(2, 3, 4)
    step_mask = torch.ones(2, 3, 4, dtype=torch.bool)

    continuous = adapter.to_continuous_grpo_batch(
        reward_batch,
        policy_step_logps=steps,
        old_step_logps=steps - 0.1,
        step_mask=step_mask,
    )

    assert continuous["policy_step_logps"].shape == (2, 3, 4)
    assert continuous["rewards"].shape == (2, 3)

    neighbor = torch.zeros(2, 5, 4, 3)
    neighbor_mask = torch.ones(2, 5, 4, 3, dtype=torch.bool)
    neighbor_batch = adapter.to_neighbor_grpo_batch(
        reward_batch,
        policy_neighbor_logps=neighbor,
        old_neighbor_logps=neighbor - 0.1,
        neighbor_mask=neighbor_mask,
    )

    assert neighbor_batch["policy_neighbor_logps"].shape == (2, 5, 4, 3)
    assert neighbor_batch["rewards"].shape == (2, 3)


def test_sac_score_requires_external_rewards() -> None:
    adapter = SACRLAdapter()

    with pytest.raises(ValueError, match="task-specific"):
        adapter.score(_rollout(batch_size=1, group_size=1))


def test_sac_rollout_decodes_variable_length_rows_individually() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.decode_shapes: list[tuple[int, int]] = []

        def sample_features(
            self,
            semantic_codes: torch.Tensor,
            *,
            mask: torch.Tensor | None = None,
            reference_features: torch.Tensor | None = None,
            reference_mask: torch.Tensor | None = None,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            del mask, reference_features, reference_mask, generator
            return torch.ones(
                semantic_codes.size(0),
                semantic_codes.size(1),
                2,
                dtype=torch.float32,
            )

        def decode_features(
            self,
            semantic_codes: torch.Tensor,
            features: torch.Tensor,
            *,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if semantic_codes.size(0) != 1:
                raise AssertionError("runtime must decode one row at a time")
            if mask is None:
                raise AssertionError("semantic mask is required")
            self.decode_shapes.append((semantic_codes.size(0), int(mask.sum().item())))
            return features.sum(dim=-1).unsqueeze(1)

    batch = SemanticCodecBatch(
        semantic_codes=torch.tensor([[[1], [2], [3]], [[4], [8], [8]]], dtype=torch.long),
        acoustic_codes=torch.tensor([[[1], [1], [1]], [[2], [5], [5]]], dtype=torch.long),
        mask=torch.tensor([[True, True, True], [True, False, False]]),
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
        acoustic_mask=torch.tensor([[True, True, True], [True, False, False]]),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )
    runtime = Runtime()
    adapter = SACRLAdapter(runtime)  # type: ignore[arg-type]

    rollout = adapter.rollout(batch)

    assert [candidate.sample_id for candidate in rollout.candidates] == [0, 1]
    assert runtime.decode_shapes == [(1, 3), (1, 1)]


def _rollout(*, batch_size: int, group_size: int) -> SACRollout:
    candidates = []
    for sample_id in range(batch_size):
        for candidate_id in range(group_size):
            candidates.append(
                SACCandidate(
                    sample_id=sample_id,
                    group_id=sample_id,
                    candidate_id=candidate_id,
                    semantic_codes=torch.zeros(1, 4, 1, dtype=torch.long),
                    semantic_mask=torch.ones(1, 4, dtype=torch.bool),
                )
            )
    return SACRollout(tuple(candidates), batch_size=batch_size, group_size=group_size)
