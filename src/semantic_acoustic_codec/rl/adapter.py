from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import torch
from anytrain.codec import masked_acoustic_features
from anytrain.framework.rl import ContinuousRolloutBatch, NeighborRolloutBatch, RolloutBatch
from torch import Tensor

from semantic_acoustic_codec.types import SemanticCodecBatch

from .types import SACCandidate, SACRewardBatch, SACRollout

if TYPE_CHECKING:
    from semantic_acoustic_codec.runtime import SemanticCodecRuntime


class SACRLAdapter:
    """SAC-owned adapter from semantic/acoustic candidates to anytrain RL tensors.

    The adapter owns codec rollout records and reward plumbing, while anytrain
    continues to own only tensor-level RL objectives.
    """

    def __init__(self, runtime: SemanticCodecRuntime | None = None) -> None:
        self.runtime = runtime

    @torch.no_grad()
    def rollout(
        self,
        batch: SemanticCodecBatch,
        *,
        group_size: int = 1,
        generator: torch.Generator | None = None,
    ) -> SACRollout:
        if self.runtime is None:
            raise ValueError("runtime is required for SAC codec rollout.")
        if group_size <= 0:
            raise ValueError("group_size must be positive.")
        reference_features = _reference_features(self.runtime, batch)
        candidates: list[SACCandidate] = []
        for candidate_id in range(group_size):
            features = self.runtime.sample_features(
                batch.semantic_codes,
                mask=batch.semantic_mask,
                reference_features=reference_features,
                reference_mask=batch.reference_acoustic_mask,
                generator=generator,
            )
            waveform = self.runtime.decode_features(
                batch.semantic_codes,
                features,
                mask=batch.semantic_mask,
            )
            for sample_id in range(batch.semantic_codes.size(0)):
                metadata = _metadata(batch, sample_id)
                acoustic_mask = batch.target_acoustic_mask[sample_id : sample_id + 1]
                candidates.append(
                    SACCandidate(
                        sample_id=sample_id,
                        group_id=sample_id,
                        candidate_id=candidate_id,
                        semantic_codes=batch.semantic_codes[sample_id : sample_id + 1].detach(),
                        semantic_mask=batch.semantic_mask[sample_id : sample_id + 1].detach(),
                        acoustic_features=features[sample_id : sample_id + 1].detach(),
                        acoustic_mask=acoustic_mask.detach(),
                        waveform=waveform[sample_id : sample_id + 1].detach(),
                        metadata=metadata,
                    )
                )
        ordered = tuple(sorted(candidates, key=lambda item: (item.sample_id, item.candidate_id)))
        return SACRollout(
            candidates=ordered,
            batch_size=batch.semantic_codes.size(0),
            group_size=group_size,
        )

    def score(
        self,
        candidates: SACRollout,
        *,
        rewards: Tensor | None = None,
        group_mask: Tensor | None = None,
        components: Mapping[str, Tensor] | None = None,
    ) -> SACRewardBatch:
        if rewards is None:
            raise ValueError("SAC reward scoring is task-specific; pass rewards explicitly.")
        _validate_reward_shape(rewards, candidates)
        return SACRewardBatch(
            rewards=rewards,
            group_mask=group_mask,
            components={} if components is None else components,
        )

    def to_grpo_batch(
        self,
        reward_batch: SACRewardBatch,
        *,
        policy_token_logps: Tensor,
        old_token_logps: Tensor,
        response_mask: Tensor,
        ref_token_logps: Tensor | None = None,
        group_mask: Tensor | None = None,
    ) -> RolloutBatch:
        _same_shape(policy_token_logps, old_token_logps, name="token logps")
        _same_shape(policy_token_logps, response_mask, name="response_mask")
        _candidate_shape(policy_token_logps, reward_batch.rewards)
        return {
            "policy_token_logps": policy_token_logps,
            "old_token_logps": old_token_logps,
            "rewards": reward_batch.rewards,
            "response_mask": response_mask,
            "ref_token_logps": ref_token_logps,
            "group_mask": reward_batch.group_mask if group_mask is None else group_mask,
        }

    def to_continuous_grpo_batch(
        self,
        reward_batch: SACRewardBatch,
        *,
        policy_step_logps: Tensor,
        old_step_logps: Tensor,
        step_mask: Tensor,
        kl_values: Tensor | None = None,
        group_mask: Tensor | None = None,
    ) -> ContinuousRolloutBatch:
        _same_shape(policy_step_logps, old_step_logps, name="step logps")
        _same_shape(policy_step_logps, step_mask, name="step_mask")
        _candidate_shape(policy_step_logps, reward_batch.rewards)
        return {
            "policy_step_logps": policy_step_logps,
            "old_step_logps": old_step_logps,
            "rewards": reward_batch.rewards,
            "step_mask": step_mask,
            "kl_values": kl_values,
            "group_mask": reward_batch.group_mask if group_mask is None else group_mask,
        }

    def to_neighbor_grpo_batch(
        self,
        reward_batch: SACRewardBatch,
        *,
        policy_neighbor_logps: Tensor,
        old_neighbor_logps: Tensor,
        neighbor_mask: Tensor,
        kl_values: Tensor | None = None,
        group_mask: Tensor | None = None,
        anchor_mask: Tensor | None = None,
    ) -> NeighborRolloutBatch:
        _same_shape(policy_neighbor_logps, old_neighbor_logps, name="neighbor logps")
        _same_shape(policy_neighbor_logps, neighbor_mask, name="neighbor_mask")
        if policy_neighbor_logps.shape[0] != reward_batch.rewards.shape[0]:
            raise ValueError("neighbor logps must align with reward batch axis.")
        if policy_neighbor_logps.shape[-1] != reward_batch.rewards.shape[1]:
            raise ValueError("neighbor logps candidate axis must align with rewards.")
        return {
            "policy_neighbor_logps": policy_neighbor_logps,
            "old_neighbor_logps": old_neighbor_logps,
            "rewards": reward_batch.rewards,
            "neighbor_mask": neighbor_mask,
            "kl_values": kl_values,
            "group_mask": reward_batch.group_mask if group_mask is None else group_mask,
            "anchor_mask": anchor_mask,
        }


def _reference_features(runtime: SemanticCodecRuntime, batch: SemanticCodecBatch) -> Tensor | None:
    if batch.reference_acoustic_codes is None:
        return None
    if batch.reference_acoustic_mask is None:
        raise RuntimeError("reference_acoustic_mask is required with reference_acoustic_codes.")
    return masked_acoustic_features(
        runtime.backend,
        batch.reference_acoustic_codes,
        batch.reference_acoustic_mask,
    )


def _metadata(batch: SemanticCodecBatch, sample_id: int) -> Mapping[str, object]:
    if not batch.metadata:
        return {}
    return {"pair": batch.metadata[sample_id]}


def _validate_reward_shape(rewards: Tensor, candidates: SACRollout) -> None:
    if rewards.shape != (candidates.batch_size, candidates.group_size):
        raise ValueError("rewards must have shape [batch, group].")


def _same_shape(left: Tensor, right: Tensor, *, name: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{name} tensors must have the same shape.")


def _candidate_shape(values: Tensor, rewards: Tensor) -> None:
    if values.dim() != 3:
        raise ValueError("GRPO tensors must have shape [batch, group, steps].")
    if values.shape[:2] != rewards.shape:
        raise ValueError("GRPO tensors must align with rewards on [batch, group].")


__all__ = ["SACRLAdapter"]
