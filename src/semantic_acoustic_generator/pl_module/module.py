"""Lightning lifecycle for semantic-to-acoustic generator training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from anytrain import observation
from anytrain.codec import SemanticAcousticCodec
from anytrain.lightning import LightningLogMixin
from lightning import LightningModule

from semantic_acoustic_generator.backend import LongCatCodebookAdapter
from semantic_acoustic_generator.config import FactorPredictor, Route
from semantic_acoustic_generator.loss.repa import decode_group_metrics
from semantic_acoustic_generator.pl_module.objective import (
    reference_features,
    training_loss,
    validation_error,
    validation_metric,
)
from semantic_acoustic_generator.runtime.artifact import save_artifact
from semantic_acoustic_generator.runtime.semantic import (
    GeneratorConfig,
    GeneratorSupport,
)
from semantic_acoustic_generator.types import GeneratorBatch

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor

    from semantic_acoustic_generator.loss.repa import Teacher

_DependencyT = TypeVar("_DependencyT")


class _ExternalDependency(Generic[_DependencyT]):
    """Keep frozen runtime dependencies out of Lightning module registration."""

    def __init__(self, value: _DependencyT) -> None:
        self.value = value


class GeneratorModule(LightningLogMixin, LightningModule):
    """Train the semantic-only unit generator owned by this package."""

    def __init__(
        self,
        support: GeneratorSupport,
        config: GeneratorConfig,
        *,
        backend: SemanticAcousticCodec,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        reference_dropout: float = 0.5,
        validation_seed: int = 0,
        residual_retarget: bool = False,
        repa_teacher: Teacher | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if isinstance(reference_dropout, bool) or not isinstance(reference_dropout, (int, float)):
            raise TypeError("reference_dropout must be a number.")
        if not math.isfinite(reference_dropout) or not 0 <= reference_dropout <= 1:
            raise ValueError("reference_dropout must be between 0 and 1.")
        if isinstance(validation_seed, bool) or not isinstance(validation_seed, int):
            raise TypeError("validation_seed must be an integer.")
        if validation_seed < 0:
            raise ValueError("validation_seed must be non-negative.")
        if not isinstance(residual_retarget, bool):
            raise TypeError("residual_retarget must be a boolean.")
        if support.route is not config.route:
            raise ValueError("module config route must match support route.")
        repa_weight = config.head.repa_loss_weight
        if repa_weight > 0 and support.route is not Route.FM:
            raise ValueError("REPA is only supported by the FM route.")
        if repa_weight > 0 and repa_teacher is None:
            raise ValueError("REPA requires a teacher when repa_loss_weight is positive.")
        if (
            repa_weight > 0
            and repa_teacher is not None
            and repa_teacher.feature_dim != config.head.repa_feature_dim
        ):
            raise ValueError("REPA teacher feature_dim must match repa_feature_dim.")
        if residual_retarget and (
            not isinstance(backend, LongCatCodebookAdapter)
            or config.head.factor_predictor is not FactorPredictor.DEPTH_RECURRENT
        ):
            raise ValueError(
                "residual retargeting requires a LongCat recurrent factor predictor."
            )
        self.support = support
        self._backend = _ExternalDependency(backend)
        _freeze_backend(backend)
        self._repa_teacher = _ExternalDependency(repa_teacher)
        _freeze_external(repa_teacher)
        self.strict_loading = True
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reference_dropout = float(reference_dropout)
        self.validation_seed = validation_seed
        self.residual_retarget = residual_retarget

    @property
    def backend(self) -> SemanticAcousticCodec:
        return self._backend.value

    @property
    def repa_teacher(self) -> Teacher | None:
        return self._repa_teacher.value

    def train(self, mode: bool = True):
        result = super().train(mode)
        _eval_external(self.backend)
        _eval_external(self.repa_teacher)
        return result

    def on_fit_start(self) -> None:
        _move_external(self.backend, device=self.device)
        _move_external(self.repa_teacher, device=self.device)

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if isinstance(batch, GeneratorBatch):
            return batch.to(device, non_blocking=True)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def training_step(self, batch: GeneratorBatch, batch_idx: int) -> dict[str, Tensor]:
        del batch_idx
        mask = batch.mask
        acoustic_mask = batch.acoustic_mask
        condition, reference_rows = self._condition(batch)
        output = training_loss(
            self.support,
            self.backend,
            batch,
            condition,
            repa_teacher=self.repa_teacher,
            residual_retarget=self.residual_retarget,
        )
        self.log("loss", output.loss, on_step=True, prog_bar=True, sync_dist=False)
        diagnostics = {
            "batch_size": observation.Curve(output.loss.new_tensor(mask.size(0))),
            "valid_frames": observation.Curve(mask.sum().to(dtype=output.loss.dtype)),
            "reference_fraction": observation.Curve(
                reference_rows.to(dtype=output.loss.dtype).mean()
            ),
            "valid_acoustic_units": observation.Curve(
                acoustic_mask.sum().to(dtype=output.loss.dtype)
            ),
        }
        if self.repa_teacher is not None:
            diagnostics.update(
                {
                    f"repa/{name}": observation.Curve(output.loss.new_tensor(value))
                    for name, value in decode_group_metrics(mask).items()
                }
            )
        observation.emit(self, "batch", diagnostics)
        return {"loss": output.loss, **output.losses}

    @torch.no_grad()
    def validation_step(self, batch: GeneratorBatch, batch_idx: int) -> dict[str, Tensor]:
        paired_features = reference_features(self.backend, batch)
        reference_mask = (
            None
            if paired_features is None
            else batch.reference.acoustic_mask.to(device=batch.semantic_codes.device)
        )
        without = self._validation_error(
            batch,
            reference_features=None,
            reference_mask=None,
            generator=self._validation_generator(batch_idx),
        )
        suffix = validation_metric(self.support)
        metrics = {f"val/without_reference_{suffix}": without}
        if paired_features is not None:
            with_reference = self._validation_error(
                batch,
                reference_features=paired_features,
                reference_mask=reference_mask,
                generator=self._validation_generator(batch_idx),
            )
            metrics[f"val/with_reference_{suffix}"] = with_reference
            metrics[f"val/reference_gain_{suffix}"] = without - with_reference
        # Lightning uses batch_size as the denominator weight for its epoch mean.
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch.acoustic_valid_units,
            )
        return metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [parameter for parameter in self.support.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def export_artifact(self, path: str | Path) -> None:
        save_artifact(path, self.support, backend=self.backend)

    def _condition(self, batch: GeneratorBatch) -> tuple[Tensor, Tensor]:
        if not batch.has_reference:
            condition = self.support.condition(
                batch.semantic_codes,
                mask=batch.mask,
                validate=False,
            )
            rows = torch.zeros(batch.semantic_codes.size(0), dtype=torch.bool, device=self.device)
            return condition, rows

        rows, indices = self._reference_rows(batch.semantic_codes.size(0))
        if indices.numel() == 0:
            condition = self.support.condition(
                batch.semantic_codes,
                mask=batch.mask,
                validate=False,
            )
            return condition, rows

        reference = batch.reference
        reference_indices = indices.to(device=reference.acoustic_mask.device)
        reference_mask = reference.acoustic_mask.index_select(0, reference_indices)
        reference_mask = reference_mask.to(device=batch.semantic_codes.device)
        paired_features = reference_features(self.backend, batch, indices=indices)
        if paired_features is None:
            raise RuntimeError("paired generator batch is missing reference features.")
        condition = self.support.condition(
            batch.semantic_codes,
            mask=batch.mask,
            reference_features=paired_features,
            reference_mask=reference_mask,
            reference_indices=indices,
            validate=False,
        )
        return condition, rows

    def _reference_rows(self, batch_size: int) -> tuple[Tensor, Tensor]:
        if self.reference_dropout <= 0:
            rows = torch.ones(batch_size, dtype=torch.bool)
        elif self.reference_dropout >= 1:
            rows = torch.zeros(batch_size, dtype=torch.bool)
        else:
            rows = torch.rand(batch_size, device="cpu") >= self.reference_dropout
        indices = rows.nonzero(as_tuple=False).flatten()
        return rows.to(device=self.device), indices

    def _validation_generator(self, batch_idx: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.validation_seed + batch_idx)
        return generator

    def _validation_error(
        self,
        batch: GeneratorBatch,
        *,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
        generator: torch.Generator,
    ) -> Tensor:
        return validation_error(
            self.support,
            self.backend,
            batch,
            reference_features=reference_features,
            reference_mask=reference_mask,
            generator=generator,
        )


def _freeze_backend(backend: SemanticAcousticCodec) -> None:
    _freeze_external(backend)


def _freeze_external(value: object) -> None:
    if isinstance(value, torch.nn.Module):
        value.requires_grad_(False)
        value.eval()


def _eval_external(value: object) -> None:
    if isinstance(value, torch.nn.Module):
        value.eval()


def _move_external(value: object, *, device: torch.device) -> None:
    if isinstance(value, torch.nn.Module):
        value.to(device=device)
        _freeze_external(value)


observation.registry.register(
    GeneratorModule,
    (
        observation.ForwardEvent(
            "batch",
            reduction=observation.Reduction.Mean,
            recommended=True,
        ),
    ),
)


__all__ = ["GeneratorModule"]
