from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from anytrain.codec import (
    SemanticAcousticCodec,
    masked_acoustic_features,
    semantic_acoustic_spec,
)
from anytrain.lightning import LightningLogMixin
from lightning import LightningModule

from semantic_acoustic_generator.backend import LongCatCodebookAdapter, adapt_backend
from semantic_acoustic_generator.config import AnchorTarget, FactorPredictor, Route
from semantic_acoustic_generator.loss.repa import decode_group_metrics
from semantic_acoustic_generator.model.decoder import FMFeatureGenerator, RVQCodeGenerator
from semantic_acoustic_generator.runtime.artifact import save_artifact
from semantic_acoustic_generator.runtime.semantic import (
    GeneratorConfig,
    GeneratorSupport,
    build_support,
)
from semantic_acoustic_generator.types import GeneratorBatch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from torch import Tensor

    from semantic_acoustic_generator.loss.repa import Teacher

CHECKPOINT_SCHEMA_VERSION = 3
CHECKPOINT_METADATA_KEY = "semantic_acoustic_generator"
LEGACY_CHECKPOINT_SCHEMA_VERSION = 2
LEGACY_CHECKPOINT_METADATA_KEY = "semantic_acoustic_codec"

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
        finite_loss_check_interval: int = 100,
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
        if isinstance(finite_loss_check_interval, bool) or not isinstance(
            finite_loss_check_interval,
            int,
        ):
            raise TypeError("finite_loss_check_interval must be an integer.")
        if finite_loss_check_interval <= 0:
            raise ValueError("finite_loss_check_interval must be positive.")
        if not isinstance(residual_retarget, bool):
            raise TypeError("residual_retarget must be a boolean.")
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
        if residual_retarget and (
            not isinstance(backend, LongCatCodebookAdapter)
            or config.decoder.factor_predictor is not FactorPredictor.DEPTH_RECURRENT
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
        self.finite_loss_check_interval = finite_loss_check_interval
        self.residual_retarget = residual_retarget
        self._finite_training_loss_ok: Tensor | None = None
        self._finite_training_loss_name: str | None = None

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

    def training_step(self, batch: GeneratorBatch, batch_idx: int) -> dict[str, Any]:
        del batch_idx
        mask = batch.mask
        acoustic_mask = batch.acoustic_mask
        feature_generator: FMFeatureGenerator | None = None
        target_features: Tensor | None = None
        if self.support.route is Route.FM:
            generator = self.support.generator
            if not isinstance(generator, FMFeatureGenerator):
                raise TypeError("FM support requires an FMFeatureGenerator.")
            feature_generator = generator
            if generator.anchor_target is AnchorTarget.FEATURE:
                target_features = self._target_features(batch)
        condition, reference_rows = self._condition(batch)
        if feature_generator is not None:
            factor_targets = self._factor_targets(batch)
            output = feature_generator.loss(
                batch,
                condition,
                target_features,
                feature_mean=self.support.feature_mean,
                feature_std=self.support.feature_std,
                repa_teacher=self.repa_teacher,
                factor_targets=factor_targets,
                factor_codebooks=self._factor_codebooks(),
                factor_targeter=self._factor_targeter(batch),
                include_details=self._include_factor_details(feature_generator),
            )
        else:
            generator = self.support.generator
            if not isinstance(generator, RVQCodeGenerator):
                raise TypeError("RVQ support requires an RVQCodeGenerator.")
            output = generator.loss(batch, condition)
        self._track_finite_training_loss(output.loss, name=output.primary)

        live: dict[str, Any] = {
            f"{name}_loss": item.loss.mean() for name, item in output.items.items()
        }
        live.update({name: value for name, value in output.scalars.items()})
        primary_key = f"{output.primary}_loss"
        self.log_prefixed_dict(
            "train",
            {primary_key: live[primary_key]},
            on_step=True,
            prog_bar=True,
            sync_dist=False,
        )
        secondary = {name: value for name, value in live.items() if name != primary_key}
        if secondary:
            self.log_prefixed_dict(
                "train",
                secondary,
                on_step=True,
                prog_bar=False,
                sync_dist=False,
            )
        self.log_prefixed_dict(
            "train",
            {
                "batch_size": float(mask.size(0)),
                "valid_frames": mask.sum().float(),
                "reference_fraction": reference_rows.float().mean(),
                "valid_acoustic_units": acoustic_mask.sum().float(),
            },
            on_step=True,
            sync_dist=False,
        )
        if self.repa_teacher is not None:
            self.log_prefixed_dict(
                "train/repa",
                decode_group_metrics(mask),
                on_step=True,
                sync_dist=False,
            )
        primary_item = output.items[output.primary]
        if primary_item.details is not None:
            details = {
                name: value.mean()
                for name, value in primary_item.details.items()
                if name != "frames"
            }
            if details:
                self.log_prefixed_dict(
                    f"train/{primary_key}",
                    details,
                    on_step=True,
                    sync_dist=False,
                )
        result: dict[str, Any] = {"loss": output.loss, **output.items}
        return result

    def _include_factor_details(self, generator: FMFeatureGenerator) -> bool:
        if generator.anchor_target is not AnchorTarget.FACTOR or self._trainer is None:
            return True
        interval = int(getattr(self.trainer, "log_every_n_steps", 1))
        return (self.global_step + 1) % interval == 0

    @torch.no_grad()
    def validation_step(self, batch: GeneratorBatch, batch_idx: int) -> dict[str, Tensor]:
        reference_features = self._reference_features(batch)
        reference_mask = (
            None
            if reference_features is None
            else batch.reference.acoustic_mask.to(device=batch.semantic_codes.device)
        )
        without = self._validation_error(
            batch,
            reference_features=None,
            reference_mask=None,
            generator=self._validation_generator(batch_idx),
        )
        suffix = "feature_mse"
        if self.support.route is Route.RVQ:
            suffix = "code_error"
        elif (
            isinstance(self.support.generator, FMFeatureGenerator)
            and self.support.generator.anchor_target is AnchorTarget.FACTOR
        ):
            suffix = "factor_code_error"
        metrics = {f"val/without_reference_{suffix}": without}
        if reference_features is not None:
            with_reference = self._validation_error(
                batch,
                reference_features=reference_features,
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

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        if (self.global_step + 1) % self.finite_loss_check_interval == 0:
            self._flush_finite_training_loss()

    def on_train_epoch_end(self) -> None:
        self._flush_finite_training_loss()

    def on_train_end(self) -> None:
        self._flush_finite_training_loss()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._flush_finite_training_loss()
        state = checkpoint.get("state_dict")
        if isinstance(state, dict):
            _strip_external_state(state)
        checkpoint.pop(LEGACY_CHECKPOINT_METADATA_KEY, None)
        checkpoint[CHECKPOINT_METADATA_KEY] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "backend_state": "external",
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        metadata, schema_version = _checkpoint_metadata(checkpoint)
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("generator checkpoint metadata must be a mapping.")
            if metadata.get("schema_version") != schema_version:
                raise ValueError(
                    "unsupported semantic-acoustic generator checkpoint schema: "
                    f"{metadata.get('schema_version')!r}"
                )
            if metadata.get("backend_state") != "external":
                raise ValueError("generator checkpoint backend_state must be 'external'.")
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise TypeError("generator checkpoint state_dict must be a mapping.")
        if isinstance(state, dict):
            _strip_external_state(state)
        optimizer_states = checkpoint.get("optimizer_states")
        if optimizer_states is not None:
            if not isinstance(optimizer_states, list):
                raise TypeError("checkpoint optimizer_states must be a list.")
            for optimizer_state in optimizer_states:
                if not isinstance(optimizer_state, dict):
                    raise TypeError("checkpoint optimizer state must be a mapping.")
                param_groups = optimizer_state.get("param_groups")
                if not isinstance(param_groups, list):
                    raise TypeError("checkpoint optimizer param_groups must be a list.")
                for param_group in param_groups:
                    if not isinstance(param_group, dict):
                        raise TypeError("checkpoint optimizer param group must be a mapping.")
                    param_group["lr"] = self.learning_rate
        required = {f"support.{key}" for key in self.support.state_dict()}
        missing = sorted(key for key in required if key not in state)
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
            raise RuntimeError(
                f"generator checkpoint is missing support state: {preview}{suffix}"
            )

    def export_artifact(self, path: str | Path) -> None:
        save_artifact(path, self.support, backend=self.backend)

    def _normalized_features(self, batch: GeneratorBatch) -> Tensor:
        features = self._target_features(batch)
        reference = self.support.feature_mean
        features = features.to(device=reference.device, dtype=reference.dtype)
        return (features - self.support.feature_mean) / self.support.feature_std

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
        reference_features = self._reference_features(batch, indices=indices)
        if reference_features is None:
            raise RuntimeError("paired semantic codec batch is missing reference features.")
        condition = self.support.condition(
            batch.semantic_codes,
            mask=batch.mask,
            reference_features=reference_features,
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

    def _track_finite_training_loss(self, loss: Tensor, *, name: str) -> None:
        finite = torch.isfinite(loss.detach()).all()
        if finite.device.type == "cuda":
            self._defer_finite_training_loss(finite, name=name)
            return
        if not bool(finite):
            raise FloatingPointError(f"{name} training loss is non-finite.")

    def _defer_finite_training_loss(self, finite: Tensor, *, name: str) -> None:
        pending = self._finite_training_loss_ok
        if pending is None:
            self._finite_training_loss_ok = finite
            self._finite_training_loss_name = name
            return
        pending.logical_and_(finite)

    def _flush_finite_training_loss(self) -> None:
        pending = self._finite_training_loss_ok
        if pending is None:
            return
        reduced = pending.to(dtype=torch.uint8)
        if self._trainer is not None:
            reduced = self.trainer.strategy.reduce(reduced, reduce_op="min")
        if not bool(reduced):
            name = self._finite_training_loss_name
            if name is None:
                raise RuntimeError("finite loss guard is missing its loss name.")
            raise FloatingPointError(f"{name} training loss is non-finite.")
        self._finite_training_loss_ok = None
        self._finite_training_loss_name = None

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
        if self.support.route is Route.FM:
            feature_generator = self.support.generator
            if (
                isinstance(feature_generator, FMFeatureGenerator)
                and feature_generator.anchor_target is AnchorTarget.FACTOR
            ):
                condition = self.support.condition(
                    batch.semantic_codes,
                    mask=batch.mask,
                    reference_features=reference_features,
                    reference_mask=reference_mask,
                )
                predicted = feature_generator.sample_factor_codes(condition, batch.mask)
                target_factors = self._factor_targets(batch)
                if target_factors is None:
                    raise RuntimeError("factor validation requires factor targets.")
                error = predicted.ne(target_factors.to(device=predicted.device)).float().mean(-1)
                return _masked_mean(error, batch.acoustic_mask)
            prediction = self.support.sample_features(
                batch.semantic_codes,
                mask=batch.mask,
                reference_features=reference_features,
                reference_mask=reference_mask,
                generator=generator,
            )
            target = self._target_features(batch).to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            return _masked_mean((prediction - target).square().mean(dim=-1), batch.acoustic_mask)
        prediction = self.support.sample_acoustic_codes(
            batch.semantic_codes,
            mask=batch.mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            generator=generator,
        )
        target_codes = batch.acoustic_codes.to(device=prediction.device)
        error = (prediction != target_codes).float().mean(dim=-1)
        return _masked_mean(error, batch.acoustic_mask)

    @torch.no_grad()
    def _target_features(self, batch: GeneratorBatch) -> Tensor:
        return masked_acoustic_features(
            self.backend,
            batch.acoustic_codes,
            batch.acoustic_mask,
            validate=False,
        )

    @torch.no_grad()
    def _reference_features(
        self,
        batch: GeneratorBatch,
        *,
        indices: Tensor | None = None,
    ) -> Tensor | None:
        if not batch.has_reference:
            return None
        reference = batch.reference
        codes = reference.acoustic_codes
        mask = reference.acoustic_mask
        if indices is not None:
            selected = indices.to(device=codes.device)
            codes = codes.index_select(0, selected)
            mask = mask.index_select(0, selected)
        mask = mask.to(device=batch.semantic_codes.device)
        return masked_acoustic_features(
            self.backend, codes, mask, validate=False
        )

    @torch.no_grad()
    def _factor_targets(self, batch: GeneratorBatch) -> Tensor | None:
        if not isinstance(self.backend, LongCatCodebookAdapter):
            return None
        codes = batch.acoustic_codes.masked_fill(
            ~batch.acoustic_mask[..., None],
            0,
        )
        return self.backend.factor_codes(codes)

    def _factor_codebooks(self) -> tuple[Tensor, ...] | None:
        if not isinstance(self.backend, LongCatCodebookAdapter):
            return None
        return self.backend.factor_codebooks

    def _factor_targeter(
        self,
        batch: GeneratorBatch,
    ) -> Callable[[int, Tensor], Tensor] | None:
        if not self.residual_retarget:
            return None
        if not isinstance(self.backend, LongCatCodebookAdapter):
            raise RuntimeError("residual retargeting requires a LongCat codebook adapter.")
        return self.backend.residual_factor_targeter(
            batch.acoustic_codes,
            batch.acoustic_mask,
        )


@torch.no_grad()
def build_module(
    backend: SemanticAcousticCodec,
    config: GeneratorConfig,
    sample: GeneratorBatch | None = None,
    *,
    normalize_features: bool = True,
    feature_mean: tuple[float, ...] | None = None,
    feature_std: tuple[float, ...] | None = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    reference_dropout: float = 0.5,
    validation_seed: int = 0,
    finite_loss_check_interval: int = 100,
    residual_retarget: bool = False,
    repa_teacher: Teacher | None = None,
) -> GeneratorModule:
    backend = adapt_backend(
        backend,
        config.feature_adapter,
        codebooks=config.feature_codebooks,
    )
    normalize_features = (
        normalize_features
        and config.route is not Route.RVQ
        and config.decoder.anchor_target is AnchorTarget.FEATURE
    )
    if normalize_features:
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("feature_mean and feature_std must be provided together.")
        if feature_mean is None or feature_std is None:
            if sample is None:
                raise ValueError("feature normalization requires dataset feature statistics.")
            feature_mean, feature_std = feature_stats(backend, sample)
        mean, std = feature_mean, feature_std
        config = replace(config, feature_mean=mean, feature_std=std)
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
        factor_codebooks=(
            backend.factor_codebooks
            if isinstance(backend, LongCatCodebookAdapter)
            and config.decoder.anchor_target is AnchorTarget.FACTOR
            else None
        ),
    )
    return GeneratorModule(
        support,
        config,
        backend=backend,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        reference_dropout=reference_dropout,
        validation_seed=validation_seed,
        finite_loss_check_interval=finite_loss_check_interval,
        residual_retarget=residual_retarget,
        repa_teacher=repa_teacher,
    )


@torch.no_grad()
def feature_stats(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    acoustic_mask = batch.acoustic_mask
    target = masked_acoustic_features(backend, batch.acoustic_codes, acoustic_mask).float()
    acoustic_mask = acoustic_mask.to(device=target.device)
    valid = target[acoustic_mask]
    if valid.numel() == 0:
        raise ValueError("feature stats require at least one valid frame.")
    mean = valid.mean(dim=0)
    std = valid.std(dim=0, correction=0).clamp_min(1e-5)
    return _tuple(mean), _tuple(std)


@torch.no_grad()
def dataset_feature_stats(
    backend: SemanticAcousticCodec,
    batches: Iterable[GeneratorBatch],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mean: Tensor | None = None
    m2: Tensor | None = None
    count = 0
    for batch in batches:
        mask = batch.acoustic_mask
        target = masked_acoustic_features(backend, batch.acoustic_codes, mask).float()
        valid = target[mask.to(device=target.device)]
        if valid.numel() == 0:
            continue
        batch_variance, batch_mean = torch.var_mean(valid, dim=0, correction=0)
        batch_count = valid.size(0)
        batch_mean = batch_mean.double()
        batch_m2 = batch_variance.double() * batch_count
        if mean is None or m2 is None:
            mean = batch_mean
            m2 = batch_m2
            count = batch_count
            continue
        combined_count = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * (batch_count / combined_count)
        m2 = m2 + batch_m2 + delta.square() * (count * batch_count / combined_count)
        count = combined_count
    if count == 0 or mean is None or m2 is None:
        raise ValueError("feature stats require at least one valid acoustic unit.")
    std = (m2 / count).clamp_min(0).sqrt().clamp_min(1e-5)
    stats = torch.stack((mean, std)).cpu()
    return (
        tuple(float(item) for item in stats[0]),
        tuple(float(item) for item in stats[1]),
    )


def _tuple(value: Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu())


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    aligned = mask.to(device=value.device)
    return value[aligned].mean()


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


def _strip_external_state(state: dict[str, object]) -> None:
    for key in list(state):
        if key.startswith(("backend.", "repa_teacher.")):
            del state[key]


def _checkpoint_metadata(
    checkpoint: Mapping[str, Any],
) -> tuple[object | None, int]:
    metadata = checkpoint.get(CHECKPOINT_METADATA_KEY)
    legacy = checkpoint.get(LEGACY_CHECKPOINT_METADATA_KEY)
    if metadata is not None and legacy is not None:
        raise ValueError(
            "checkpoint must not contain both generator and legacy semantic codec metadata."
        )
    if metadata is not None:
        return metadata, CHECKPOINT_SCHEMA_VERSION
    return legacy, LEGACY_CHECKPOINT_SCHEMA_VERSION


__all__ = [
    "CHECKPOINT_METADATA_KEY",
    "CHECKPOINT_SCHEMA_VERSION",
    "LEGACY_CHECKPOINT_METADATA_KEY",
    "LEGACY_CHECKPOINT_SCHEMA_VERSION",
    "GeneratorModule",
    "build_module",
    "dataset_feature_stats",
    "feature_stats",
]
