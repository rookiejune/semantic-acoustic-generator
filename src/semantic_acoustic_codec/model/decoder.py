from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from semantic_acoustic_codec.config import DecoderConfig, Route, RVQPredictor
from semantic_acoustic_codec.data import SemanticCodecBatch
from semantic_acoustic_codec.loss import (
    FlowLoss,
    LossItem,
    RectifiedFlowRuntime,
    RepaLoss,
    RVQLoss,
    Teacher,
)
from semantic_acoustic_codec.model.dit import DiTDecoder
from semantic_acoustic_codec.model.rvq import AcousticRVQDecoder, AcousticRVQMTPDecoder
from semantic_acoustic_codec.runtime.protocol import CodecBackend


@dataclass(frozen=True)
class DecoderLoss:
    loss: Tensor
    item: LossItem
    log_name: str
    logs: dict[str, Tensor | float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class CodecUnitGenerator(ABC, nn.Module):
    route: Route

    @abstractmethod
    def sample_features(
        self,
        backend: CodecBackend,
        condition: Tensor,
        mask: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        flow_steps: int,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...

    def sample_acoustic_codes(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        raise RuntimeError("sample_acoustic_codes is only available for acoustic-code decoders.")

    @abstractmethod
    def loss(
        self,
        backend: CodecBackend,
        batch: SemanticCodecBatch,
        condition: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
    ) -> DecoderLoss: ...


class FMFeatureGenerator(CodecUnitGenerator):
    route = Route.FM

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        config: DecoderConfig,
    ) -> None:
        super().__init__()
        self.core = DiTDecoder(
            condition_dim,
            feature_dim,
            hidden_dim=config.hidden_dim,
            layers=config.layers,
            heads=config.heads,
            ffn_ratio=config.ffn_ratio,
            repa_feature_dim=config.repa_feature_dim,
            repa_student_layer=config.repa_student_layer,
        )
        self.repa_loss_weight = config.repa_loss_weight
        self.flow_loss = FlowLoss()
        self.repa_loss = RepaLoss()
        self.flow_runtime = RectifiedFlowRuntime()

    @torch.no_grad()
    def sample_features(
        self,
        backend: CodecBackend,
        condition: Tensor,
        mask: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        flow_steps: int,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del backend, temperature, top_p
        features = self.core.sample(
            condition,
            mask=mask,
            steps=flow_steps,
            generator=generator,
        )
        return features * feature_std + feature_mean

    def loss(
        self,
        backend: CodecBackend,
        batch: SemanticCodecBatch,
        condition: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
    ) -> DecoderLoss:
        target = _normalized_features(backend, batch, feature_mean=feature_mean, feature_std=feature_std)
        if self.repa_loss_weight <= 0:
            item = self.flow_loss(self.core, condition, target, batch.mask, self.flow_runtime)
            return DecoderLoss(
                loss=item.loss.mean(),
                item=item,
                log_name="train/flow_loss",
            )

        if repa_teacher is None:
            raise RuntimeError("REPA requires a teacher.")
        item, representation = self.flow_loss.forward_with_features(
            self.core,
            condition,
            target,
            batch.mask,
            self.flow_runtime,
        )
        repa_features = repa_teacher(batch.semantic_codes, batch.safe_acoustic_codes, batch.mask)
        repa = self.repa_loss(representation, repa_features, batch.mask)
        item_loss = item.loss.mean()
        repa_loss = repa.loss.mean()
        loss = item_loss + self.repa_loss_weight * repa_loss
        return DecoderLoss(
            loss=loss,
            item=item,
            log_name="train/flow_loss",
            logs={
                "train/flow_loss": item_loss,
                "train/repa_loss": repa_loss,
                "train/repa_weight": self.repa_loss_weight,
            },
            extras={"repa": repa},
        )


class RVQCodeGenerator(CodecUnitGenerator):
    route = Route.RVQ

    def __init__(
        self,
        condition_dim: int,
        codebook_sizes: tuple[int, ...],
        config: DecoderConfig,
    ) -> None:
        super().__init__()
        if not codebook_sizes:
            raise ValueError("RVQ route requires acoustic codebooks.")
        if config.rvq_predictor is RVQPredictor.CODEBOOK_AR:
            self.core: AcousticRVQDecoder | AcousticRVQMTPDecoder = AcousticRVQDecoder(
                condition_dim,
                len(codebook_sizes),
                codebook_sizes,
                hidden_dim=config.hidden_dim,
                layers=config.layers,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
            )
        elif config.rvq_predictor is RVQPredictor.MTP:
            self.core = AcousticRVQMTPDecoder(
                condition_dim,
                len(codebook_sizes),
                codebook_sizes,
                hidden_dim=config.hidden_dim,
                layers=config.layers,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
                mtp_layers=config.mtp_layers,
                mtp_heads=config.mtp_heads,
            )
        else:
            raise AssertionError(f"unsupported RVQ predictor: {config.rvq_predictor}")
        self.rvq_loss = RVQLoss()

    @torch.no_grad()
    def sample_features(
        self,
        backend: CodecBackend,
        condition: Tensor,
        mask: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        flow_steps: int,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del feature_mean, feature_std, flow_steps
        codes = self.sample_acoustic_codes(
            condition,
            mask,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        return backend.acoustic_codes_to_features(codes).to(device=condition.device, dtype=condition.dtype)

    @torch.no_grad()
    def sample_acoustic_codes(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.core.generate(
            condition,
            mask=mask,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )

    def loss(
        self,
        backend: CodecBackend,
        batch: SemanticCodecBatch,
        condition: Tensor,
        *,
        feature_mean: Tensor,
        feature_std: Tensor,
        repa_teacher: Teacher | None = None,
    ) -> DecoderLoss:
        del backend, feature_mean, feature_std, repa_teacher
        labels = batch.safe_acoustic_codes
        item = self.rvq_loss(self.core(condition, labels, mask=batch.mask), labels, batch.mask)
        return DecoderLoss(
            loss=item.loss.mean(),
            item=item,
            log_name="train/rvq_loss",
        )


@torch.no_grad()
def backend_features(backend: CodecBackend, acoustic_codes: Tensor, mask: Tensor) -> Tensor:
    if acoustic_codes.dim() != 3 or mask.shape != acoustic_codes.shape[:2]:
        raise ValueError("acoustic_codes and mask must have shapes [B, F, K] and [B, F].")
    features = backend.acoustic_codes_to_features(acoustic_codes.masked_fill(~mask[..., None], 0))
    return features.masked_fill(~mask[..., None], 0)


def _normalized_features(
    backend: CodecBackend,
    batch: SemanticCodecBatch,
    *,
    feature_mean: Tensor,
    feature_std: Tensor,
) -> Tensor:
    features = backend_features(backend, batch.safe_acoustic_codes, batch.mask)
    features = features.to(device=feature_mean.device, dtype=feature_mean.dtype)
    return (features - feature_mean) / feature_std
