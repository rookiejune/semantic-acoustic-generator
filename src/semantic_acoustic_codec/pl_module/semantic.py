from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from anytrain.codec import SemanticAcousticCodec, masked_acoustic_features
from lightning import LightningModule

from semantic_acoustic_codec.config import Route
from semantic_acoustic_codec.runtime.semantic import (
    SemanticCodecSupport,
    SemanticSupportConfig,
    build_support,
    save_artifact,
)
from semantic_acoustic_codec.types import SemanticCodecBatch

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor

    from semantic_acoustic_codec.loss.repa import Teacher

CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_METADATA_KEY = "semantic_acoustic_codec"

_DependencyT = TypeVar("_DependencyT")


class _ExternalDependency(Generic[_DependencyT]):
    """Keep frozen runtime dependencies out of Lightning module registration."""

    def __init__(self, value: _DependencyT) -> None:
        self.value = value


class SemanticCodecModule(LightningModule):
    """Train the semantic-only unit generator owned by this package."""

    def __init__(
        self,
        support: SemanticCodecSupport,
        config: SemanticSupportConfig,
        *,
        backend: SemanticAcousticCodec,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        reference_dropout: float = 0.5,
        validation_seed: int = 0,
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

    def training_step(self, batch: SemanticCodecBatch, batch_idx: int) -> dict[str, Any]:
        del batch_idx
        mask = batch.mask
        target_features = None
        if self.support.route is Route.FM:
            target_features = self._target_features(batch)
        acoustic_mask = _acoustic_mask(batch)
        condition, reference_rows = self._condition(batch)
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
                self.log(
                    name, value, on_step=True, prog_bar=name == output.log_name, sync_dist=True
                )
        else:
            self.log(output.log_name, output.loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/batch_size", float(mask.size(0)), on_step=True, sync_dist=False)
        self.log("train/valid_frames", mask.sum().float(), on_step=True, sync_dist=False)
        self.log(
            "train/reference_fraction",
            reference_rows.float().mean(),
            on_step=True,
            sync_dist=False,
        )
        self.log(
            "train/valid_acoustic_units",
            acoustic_mask.sum().float(),
            on_step=True,
            sync_dist=False,
        )
        if output.item.details is not None:
            for name, value in output.item.details.items():
                if name == "frames":
                    continue
                self.log(f"{output.log_name}/{name}", value.mean(), on_step=True, sync_dist=False)
        result: dict[str, Any] = {"loss": output.loss, "item": output.item}
        result.update(output.extras)
        return result

    @torch.no_grad()
    def validation_step(self, batch: SemanticCodecBatch, batch_idx: int) -> dict[str, Tensor]:
        reference_features = self._reference_features(batch)
        reference_mask = None if reference_features is None else _reference_acoustic_mask(batch)
        without = self._validation_error(
            batch,
            reference_features=None,
            reference_mask=None,
            generator=self._validation_generator(batch_idx),
        )
        suffix = "feature_mse" if self.support.route is Route.FM else "code_error"
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
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch.semantic_codes.size(0),
            )
        return metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [parameter for parameter in self.support.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("state_dict")
        if isinstance(state, dict):
            _strip_external_state(state)
        checkpoint[CHECKPOINT_METADATA_KEY] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "backend_state": "external",
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        metadata = checkpoint.get(CHECKPOINT_METADATA_KEY)
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("semantic-acoustic checkpoint metadata must be a mapping.")
            if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(
                    "unsupported semantic-acoustic checkpoint schema: "
                    f"{metadata.get('schema_version')!r}"
                )
            if metadata.get("backend_state") != "external":
                raise ValueError("semantic-acoustic checkpoint backend_state must be 'external'.")
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise TypeError("semantic-acoustic checkpoint state_dict must be a mapping.")
        if isinstance(state, dict):
            _strip_external_state(state)
        required = {f"support.{key}" for key in self.support.state_dict()}
        missing = sorted(key for key in required if key not in state)
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
            raise RuntimeError(
                f"semantic-acoustic checkpoint is missing support state: {preview}{suffix}"
            )

    def export_artifact(self, path: str | Path) -> None:
        save_artifact(path, self.support, self.config, backend=self.backend)

    def _normalized_features(self, batch: SemanticCodecBatch) -> Tensor:
        features = self._target_features(batch)
        reference = self.support.feature_mean
        features = features.to(device=reference.device, dtype=reference.dtype)
        return (features - self.support.feature_mean) / self.support.feature_std

    def _condition(self, batch: SemanticCodecBatch) -> tuple[Tensor, Tensor]:
        reference_features = self._reference_features(batch)
        if reference_features is None:
            condition = self.support.condition(
                batch.semantic_codes,
                mask=batch.mask,
                validate=False,
            )
            rows = torch.zeros(batch.semantic_codes.size(0), dtype=torch.bool, device=self.device)
            return condition, rows

        reference_mask = _reference_acoustic_mask(batch)
        rows = self._reference_rows(batch.semantic_codes.size(0))
        condition = self.support.condition(
            batch.semantic_codes,
            mask=batch.mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            use_reference=rows,
            validate=False,
        )
        return condition, rows

    def _reference_rows(self, batch_size: int) -> Tensor:
        if self.reference_dropout <= 0:
            return torch.ones(batch_size, dtype=torch.bool, device=self.device)
        if self.reference_dropout >= 1:
            return torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        return torch.rand(batch_size, device=self.device) >= self.reference_dropout

    def _validation_generator(self, batch_idx: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.validation_seed + batch_idx)
        return generator

    def _validation_error(
        self,
        batch: SemanticCodecBatch,
        *,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
        generator: torch.Generator,
    ) -> Tensor:
        output_length = self.backend.acoustic_unit_length
        if self.support.route is Route.FM:
            prediction = self.support.sample_features(
                batch.semantic_codes,
                mask=batch.mask,
                reference_features=reference_features,
                reference_mask=reference_mask,
                output_length=output_length,
                generator=generator,
            )
            target = self._target_features(batch).to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            return _masked_mean((prediction - target).square().mean(dim=-1), _acoustic_mask(batch))
        prediction = self.support.sample_acoustic_codes(
            batch.semantic_codes,
            mask=batch.mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            output_length=output_length,
            generator=generator,
        )
        target_codes = batch.acoustic_codes.to(device=prediction.device)
        error = (prediction != target_codes).float().mean(dim=-1)
        return _masked_mean(error, _acoustic_mask(batch))

    @torch.no_grad()
    def _target_features(self, batch: SemanticCodecBatch) -> Tensor:
        return masked_acoustic_features(
            self.backend,
            batch.acoustic_codes,
            _acoustic_mask(batch),
            validate=False,
        )

    @torch.no_grad()
    def _reference_features(self, batch: SemanticCodecBatch) -> Tensor | None:
        codes = batch.reference_acoustic_codes
        if codes is None:
            return None
        mask = _reference_acoustic_mask(batch)
        return masked_acoustic_features(self.backend, codes, mask, validate=False)


@torch.no_grad()
def build_module(
    backend: SemanticAcousticCodec,
    config: SemanticSupportConfig,
    sample: SemanticCodecBatch | None = None,
    *,
    normalize_features: bool = True,
    feature_mean: tuple[float, ...] | None = None,
    feature_std: tuple[float, ...] | None = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    reference_dropout: float = 0.5,
    validation_seed: int = 0,
    repa_teacher: Teacher | None = None,
) -> SemanticCodecModule:
    if normalize_features and config.route is not Route.RVQ:
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
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
        acoustic_layout=backend.acoustic_layout,
        acoustic_unit_length=backend.acoustic_unit_length,
    )
    return SemanticCodecModule(
        support,
        config,
        backend=backend,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        reference_dropout=reference_dropout,
        validation_seed=validation_seed,
        repa_teacher=repa_teacher,
    )


@torch.no_grad()
def feature_stats(
    backend: SemanticAcousticCodec,
    batch: SemanticCodecBatch,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    acoustic_mask = _acoustic_mask(batch)
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
    batches: Iterable[SemanticCodecBatch],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    total = torch.zeros(backend.acoustic_feature_dim, dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for batch in batches:
        mask = _acoustic_mask(batch)
        target = masked_acoustic_features(backend, batch.acoustic_codes, mask).double()
        valid = target[mask.to(device=target.device)]
        if valid.numel() == 0:
            continue
        total += valid.sum(dim=0).cpu()
        squares += valid.square().sum(dim=0).cpu()
        count += valid.size(0)
    if count == 0:
        raise ValueError("feature stats require at least one valid acoustic unit.")
    mean = total / count
    variance = (squares / count - mean.square()).clamp_min(0)
    std = variance.sqrt().clamp_min(1e-5)
    return _tuple(mean), _tuple(std)


def _acoustic_mask(batch: SemanticCodecBatch) -> Tensor:
    mask = batch.acoustic_mask
    if mask is None:
        raise RuntimeError("SemanticCodecBatch must expose acoustic_mask after validation.")
    return mask


def _reference_acoustic_mask(batch: SemanticCodecBatch) -> Tensor:
    mask = batch.reference_acoustic_mask
    if mask is None:
        raise RuntimeError("reference_acoustic_mask is required when reference codes are present.")
    return mask.to(device=batch.semantic_codes.device)


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


__all__ = [
    "CHECKPOINT_METADATA_KEY",
    "CHECKPOINT_SCHEMA_VERSION",
    "SemanticCodecModule",
    "build_module",
    "dataset_feature_stats",
    "feature_stats",
]
