from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from lightning import LightningModule

from semantic_acoustic_codec.config import Route
from semantic_acoustic_codec.data.longcat import SemanticCodecBatch
from semantic_acoustic_codec.runtime.semantic import (
    SemanticCodecSupport,
    SemanticSupportConfig,
    build_support,
    save_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor

    from semantic_acoustic_codec.loss.repa import Teacher
    from semantic_acoustic_codec.runtime.protocol import CodecBackend


class SemanticCodecModule(LightningModule):
    """Train the semantic-only unit generator owned by this package."""

    def __init__(
        self,
        support: SemanticCodecSupport,
        config: SemanticSupportConfig,
        *,
        backend: CodecBackend,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        repa_teacher: Teacher | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if support.route is not config.route:
            raise ValueError("module config route must match support route.")
        repa_weight = config.decoder.repa_loss_weight
        if repa_weight > 0 and support.route is not Route.FM:
            raise ValueError("REPA is only supported by the FM route.")
        if repa_weight > 0 and repa_teacher is None:
            raise ValueError("REPA requires a teacher when repa_loss_weight is positive.")
        if (
            repa_weight > 0
            and repa_teacher is not None
            and repa_teacher.feature_dim != config.decoder.repa_feature_dim
        ):
            raise ValueError("REPA teacher feature_dim must match repa_feature_dim.")
        self.support = support
        self.backend = backend
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.repa_teacher = repa_teacher

    def training_step(self, batch: SemanticCodecBatch, batch_idx: int) -> dict[str, Any]:
        del batch_idx
        batch = _move(batch, self.device)
        mask = batch.mask
        target_features = self._target_features(batch)
        condition = self.support.condition(
            batch.semantic_codes,
            mask=mask,
            reference_features=target_features,
            reference_mask=mask,
        )
        output = self.support.generator.loss(
            batch,
            condition,
            target_features,
            feature_mean=self.support.feature_mean,
            feature_std=self.support.feature_std,
            repa_teacher=self.repa_teacher,
        )

        if output.logs:
            for name, value in output.logs.items():
                self.log(name, value, on_step=True, prog_bar=name == output.log_name, sync_dist=True)
        else:
            self.log(output.log_name, output.loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/batch_size", float(mask.size(0)), on_step=True, sync_dist=True)
        self.log("train/valid_frames", mask.sum().float(), on_step=True, sync_dist=True)
        if output.item.details is not None:
            for name, value in output.item.details.items():
                if name == "frames":
                    continue
                self.log(f"{output.log_name}/{name}", value.mean(), on_step=True, sync_dist=True)
        result: dict[str, Any] = {"loss": output.loss, "item": output.item}
        result.update(output.extras)
        return result

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [parameter for parameter in self.support.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def export_artifact(self, path: str | Path) -> None:
        save_artifact(path, self.support, self.config)

    def _normalized_features(self, batch: SemanticCodecBatch) -> Tensor:
        features = self._target_features(batch)
        reference = self.support.feature_mean
        features = features.to(device=reference.device, dtype=reference.dtype)
        return (features - self.support.feature_mean) / self.support.feature_std

    @torch.no_grad()
    def _target_features(self, batch: SemanticCodecBatch) -> Tensor:
        features = self.backend.acoustic_codes_to_features(batch.safe_acoustic_codes)
        return features.masked_fill(~batch.mask[..., None], 0)


@torch.no_grad()
def build_module(
    backend: CodecBackend,
    config: SemanticSupportConfig,
    sample: SemanticCodecBatch | None = None,
    *,
    normalize_features: bool = True,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    repa_teacher: Teacher | None = None,
) -> SemanticCodecModule:
    if normalize_features and config.route is not Route.RVQ:
        if sample is None:
            raise ValueError("feature normalization requires a representative sample batch.")
        mean, std = feature_stats(backend, sample)
        config = replace(config, feature_mean=mean, feature_std=std)
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    )
    return SemanticCodecModule(
        support,
        config,
        backend=backend,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        repa_teacher=repa_teacher,
    )


@torch.no_grad()
def feature_stats(backend: CodecBackend, batch: SemanticCodecBatch) -> tuple[tuple[float, ...], tuple[float, ...]]:
    target = backend.acoustic_codes_to_features(batch.safe_acoustic_codes).float()
    target = target.masked_fill(~batch.mask[..., None], 0)
    valid = target[batch.mask]
    if valid.numel() == 0:
        raise ValueError("feature stats require at least one valid frame.")
    mean = valid.mean(dim=0)
    std = valid.std(dim=0, correction=0).clamp_min(1e-5)
    return _tuple(mean), _tuple(std)


def _move(batch: SemanticCodecBatch, device: torch.device) -> SemanticCodecBatch:
    return SemanticCodecBatch(
        semantic_codes=batch.semantic_codes.to(device=device),
        acoustic_codes=batch.acoustic_codes.to(device=device),
        mask=batch.mask.to(device=device),
    )


def _tuple(value: Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu())


__all__ = [
    "SemanticCodecModule",
    "build_module",
    "feature_stats",
]
