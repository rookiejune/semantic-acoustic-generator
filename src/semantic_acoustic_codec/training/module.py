from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from semantic_acoustic_codec.config import Route
from semantic_acoustic_codec.data import SemanticCodecBatch
from semantic_acoustic_codec.loss import FlowLoss, RepaLoss, RVQLoss, Teacher
from semantic_acoustic_codec.model import DiTDecoder, RectifiedFlowRuntime, teacher_features
from semantic_acoustic_codec.model.rvq import AcousticRVQDecoder
from semantic_acoustic_codec.runtime import (
    SemanticAcousticCodec,
    SemanticCodecConfig,
    build_codec,
    save_artifact,
)
from semantic_acoustic_codec.runtime.protocol import TeacherCodec


class SemanticCodecModule(LightningModule):
    """Train the semantic-only acoustic decoder owned by this package."""

    def __init__(
        self,
        codec: SemanticAcousticCodec,
        config: SemanticCodecConfig,
        *,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        repa_teacher: Teacher | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if codec.route is not config.route:
            raise ValueError("module config route must match codec route.")
        repa_weight = config.decoder.repa_loss_weight
        if repa_weight > 0 and codec.route is not Route.FM:
            raise ValueError("REPA is only supported by the FM route.")
        if repa_weight > 0 and repa_teacher is None:
            raise ValueError("REPA requires a teacher when repa_loss_weight is positive.")
        if (
            repa_weight > 0
            and repa_teacher is not None
            and repa_teacher.feature_dim != config.decoder.repa_feature_dim
        ):
            raise ValueError("REPA teacher feature_dim must match repa_feature_dim.")
        self.codec = codec
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.flow_loss = FlowLoss()
        self.repa_loss = RepaLoss()
        self.repa_teacher = repa_teacher
        self.rvq_loss = RVQLoss()
        self.flow_runtime = RectifiedFlowRuntime()

    def training_step(self, batch: SemanticCodecBatch, batch_idx: int) -> dict[str, Any]:
        del batch_idx
        batch = _move(batch, self.device)
        mask = batch.mask
        condition = self.codec.condition(
            batch.semantic_codes,
            mask=mask,
            reference_acoustic_codes=batch.safe_acoustic_codes,
            reference_mask=mask,
        )
        if self.codec.route is Route.FM:
            decoder = cast(DiTDecoder, self.codec.decoder)
            target = self._normalized_features(batch)
            if self.config.decoder.repa_loss_weight > 0:
                if self.repa_teacher is None:
                    raise RuntimeError("REPA requires a teacher.")
                item, representation = self.flow_loss.forward_with_features(
                    decoder,
                    condition,
                    target,
                    mask,
                    self.flow_runtime,
                )
                teacher = self.repa_teacher(
                    batch.semantic_codes,
                    batch.safe_acoustic_codes,
                    mask,
                )
                repa = self.repa_loss(representation, teacher, mask)
                repa_loss = repa.loss.mean()
                item_loss = item.loss.mean()
                loss = item_loss + self.config.decoder.repa_loss_weight * repa_loss
                self.log("train/flow_loss", item_loss, on_step=True, prog_bar=True, sync_dist=True)
                self.log("train/repa_loss", repa_loss, on_step=True, prog_bar=True, sync_dist=True)
                self.log(
                    "train/repa_weight",
                    self.config.decoder.repa_loss_weight,
                    on_step=True,
                    sync_dist=True,
                )
                log_name = "train/flow_loss"
            else:
                item = self.flow_loss(decoder, condition, target, mask, self.flow_runtime)
                loss = item.loss.mean()
                log_name = "train/flow_loss"
        elif self.codec.route is Route.RVQ:
            decoder = cast(AcousticRVQDecoder, self.codec.decoder)
            labels = batch.safe_acoustic_codes
            item = self.rvq_loss(decoder(condition, labels, mask=mask), labels, mask)
            loss = item.loss.mean()
            log_name = "train/rvq_loss"
        else:
            raise AssertionError(f"unsupported route: {self.codec.route}")

        if self.codec.route is not Route.FM or self.config.decoder.repa_loss_weight <= 0:
            self.log(log_name, loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/batch_size", float(mask.size(0)), on_step=True, sync_dist=True)
        self.log("train/valid_frames", mask.sum().float(), on_step=True, sync_dist=True)
        if item.details is not None:
            for name, value in item.details.items():
                if name == "frames":
                    continue
                self.log(f"{log_name}/{name}", value.mean(), on_step=True, sync_dist=True)
        result: dict[str, Any] = {"loss": loss, "item": item}
        if self.codec.route is Route.FM and self.config.decoder.repa_loss_weight > 0:
            result["repa"] = repa
        return result

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [parameter for parameter in self.codec.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def export_artifact(self, path: str | Path) -> None:
        save_artifact(path, self.codec, self.config)

    def _normalized_features(self, batch: SemanticCodecBatch) -> Tensor:
        features = teacher_features(self.codec.teacher, batch.safe_acoustic_codes, batch.mask)
        reference = self.codec.feature_mean
        features = features.to(device=reference.device, dtype=reference.dtype)
        return (features - self.codec.feature_mean) / self.codec.feature_std


class ArtifactExport(Callback):
    def __init__(self, output_dir: str | Path) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        module = _module(pl_module)
        module.export_artifact(self.output_dir / "artifact")


@torch.no_grad()
def build_module(
    teacher: TeacherCodec,
    config: SemanticCodecConfig,
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
        mean, std = feature_stats(teacher, sample)
        config = replace(config, feature_mean=mean, feature_std=std)
    codec = build_codec(teacher, config)
    return SemanticCodecModule(
        codec,
        config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        repa_teacher=repa_teacher,
    )


@torch.no_grad()
def feature_stats(teacher: TeacherCodec, batch: SemanticCodecBatch) -> tuple[tuple[float, ...], tuple[float, ...]]:
    target = teacher_features(teacher, batch.safe_acoustic_codes, batch.mask).float()
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


def _module(module: LightningModule) -> SemanticCodecModule:
    if not isinstance(module, SemanticCodecModule):
        raise TypeError("ArtifactExport requires a SemanticCodecModule.")
    return module


def _tuple(value: Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu())


__all__ = [
    "ArtifactExport",
    "SemanticCodecModule",
    "build_module",
    "feature_stats",
]
