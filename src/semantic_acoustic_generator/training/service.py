from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast

import torch
from anytrain.codec import AcousticLayout
from anytrain.lightning import (
    EMACallback,
    LossSummaryCallback,
    LossTimeBucketLoggerCallback,
    ModelCheckpoint,
    PerformanceCallback,
)
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, LearningRateMonitor

from semantic_acoustic_generator.backend import adapt_backend, load_backend
from semantic_acoustic_generator.callback import (
    ArtifactExport,
    CodebookUsageLogger,
    SampleLogConfig,
    SampleLogger,
    SemanticFrameUnits,
    UnitThroughputCallback,
)
from semantic_acoustic_generator.config import (
    AnchorTarget,
    DecoderConfig,
    Route,
)
from semantic_acoustic_generator.datamodule import (
    DataModule,
    load_batch,
    single_batch_loader,
)
from semantic_acoustic_generator.loss import WavLMTeacher
from semantic_acoustic_generator.pl_module import build_module, dataset_feature_stats
from semantic_acoustic_generator.runtime import GeneratorConfig, SamplingConfig
from semantic_acoustic_generator.training.config import (
    CallbackConfig,
    CheckpointCallbackConfig,
    CodebookUsageCallbackConfig,
    DataThroughputCallbackConfig,
    EMACallbackConfig,
    PerformanceCallbackConfig,
    PLModuleConfig,
    SampleCallbackConfig,
    TrainConfig,
    parse_train_config,
    resolve_output_dir,
)

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec
    from omegaconf import DictConfig

    from semantic_acoustic_generator.loss.repa import Teacher
    from semantic_acoustic_generator.pl_module import GeneratorModule
    from semantic_acoustic_generator.types import GeneratorBatch


@dataclass(frozen=True)
class TrainingSession:
    """Constructed training components ready for a single ``fit`` call."""

    config: TrainConfig
    output_dir: Path
    backend: SemanticAcousticCodec
    data_module: DataModule
    fixed_batch: GeneratorBatch
    module: GeneratorModule
    trainer: pl.Trainer

    def fit(self) -> None:
        checkpoint = self.config.trainer.ckpt_path
        if not self.config.datamodule.fixed_batch:
            self.trainer.fit(
                self.module,
                datamodule=self.data_module,
                ckpt_path=checkpoint,
            )
            return
        validation = None
        if self.config.datamodule.validation_split is not None:
            self.data_module.setup("fit")
            validation = self.data_module.val_dataloader()
        self.trainer.fit(
            self.module,
            train_dataloaders=single_batch_loader(self.fixed_batch),
            val_dataloaders=validation,
            ckpt_path=checkpoint,
        )


def run(config: DictConfig | TrainConfig) -> None:
    build_session(config).fit()


def build_session(config: DictConfig | TrainConfig) -> TrainingSession:
    config = parse_train_config(config)
    seed = config.seed
    pl.seed_everything(seed, workers=True)
    device = resolve_device(config.runtime.device)
    run_dir = resolve_output_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    codec = config.backend.name
    data = config.datamodule
    backend = load_backend(config.backend, device=device)
    semantic_pad_id = int(backend.semantic_codebook.size(0))
    acoustic_pad_ids = backend.acoustic_codebook_sizes
    data_module = DataModule(
        data,
        codec=codec,
        acoustic_layout=backend.acoustic_layout,
        frame_rate=backend.frame_rate,
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
    )
    if config.datamodule.fixed_batch:
        fixed_batch = load_batch(
            data,
            codec=codec,
            frame_rate=backend.frame_rate,
            acoustic_layout=backend.acoustic_layout,
            semantic_pad_id=semantic_pad_id,
            acoustic_pad_ids=acoustic_pad_ids,
        )
    else:
        data_module.setup("fit")
        fixed_batch = data_module.sample_batch()
    route = config.model.route
    repa_teacher = build_repa_teacher(config, backend, device=device, route=route)
    support_config = build_support_config(config, seed=seed, repa_teacher=repa_teacher)
    backend = adapt_backend(
        backend,
        support_config.feature_adapter,
        codebooks=support_config.feature_codebooks,
    )
    ckpt_path = config.trainer.ckpt_path
    normalize_features = (
        config.pl_module.normalize_features
        and support_config.decoder.anchor_target is AnchorTarget.FEATURE
    )
    feature_mean: tuple[float, ...] | None = None
    feature_std: tuple[float, ...] | None = None
    if normalize_features and support_config.route is Route.FM:
        checkpoint_stats = load_checkpoint_feature_stats(
            ckpt_path,
            feature_dim=int(backend.acoustic_feature_dim),
        )
        if checkpoint_stats is not None:
            feature_mean, feature_std = checkpoint_stats
        elif config.datamodule.fixed_batch:
            feature_mean, feature_std = dataset_feature_stats(backend, (fixed_batch,))
        else:
            feature_mean, feature_std = dataset_feature_stats(
                backend,
                data_module.feature_stats_dataloader(),
            )
    module = build_module(
        backend,
        support_config,
        fixed_batch,
        normalize_features=normalize_features,
        feature_mean=feature_mean,
        feature_std=feature_std,
        learning_rate=config.pl_module.learning_rate,
        weight_decay=config.pl_module.weight_decay,
        reference_dropout=config.pl_module.reference_dropout,
        validation_seed=config.pl_module.validation_seed,
        finite_loss_check_interval=config.pl_module.finite_loss_check_interval,
        repa_teacher=repa_teacher,
    )

    callbacks = build_callbacks(config, output_dir=run_dir, fixed_batch=fixed_batch)

    trainer = build_trainer(config, output_dir=run_dir, callbacks=callbacks)
    return TrainingSession(
        config=config,
        output_dir=run_dir,
        backend=backend,
        data_module=data_module,
        fixed_batch=fixed_batch,
        module=module,
        trainer=trainer,
    )


def build_trainer(
    config: TrainConfig,
    *,
    output_dir: Path,
    callbacks: list[Callback],
) -> pl.Trainer:
    checkpoint = config.callback.checkpoint
    return pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=cast(Union[str, int], config.trainer.devices),
        strategy=config.trainer.strategy,
        precision=cast(Any, config.trainer.precision),
        max_steps=config.trainer.max_steps,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=checkpoint.enabled,
        gradient_clip_val=config.trainer.gradient_clip_val,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        use_distributed_sampler=config.trainer.use_distributed_sampler,
        val_check_interval=config.trainer.val_check_interval,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
    )


def build_support_config(
    config: TrainConfig,
    *,
    seed: int,
    repa_teacher: Teacher | None,
) -> GeneratorConfig:
    decoder = config.model.decoder
    loss = config.loss
    sampling = config.runtime.sampling
    repa_feature_dim = loss.repa_feature_dim
    if repa_teacher is not None:
        if repa_feature_dim is None:
            repa_feature_dim = int(repa_teacher.feature_dim)
        elif int(repa_feature_dim) != int(repa_teacher.feature_dim):
            raise ValueError(
                "loss.repa_feature_dim must match the REPA teacher feature_dim: "
                f"{repa_feature_dim} != {repa_teacher.feature_dim}."
            )
    return GeneratorConfig(
        route=config.model.route,
        condition_dim=config.model.condition_dim,
        feature_adapter=config.model.feature_adapter,
        feature_codebooks=config.model.feature_codebooks,
        decoder=DecoderConfig(
            hidden_dim=decoder.hidden_dim,
            layers=decoder.layers,
            heads=decoder.heads,
            ffn_ratio=decoder.ffn_ratio,
            rvq_predictor=decoder.rvq_predictor,
            mtp_layers=decoder.mtp_layers,
            mtp_heads=decoder.mtp_heads,
            repa_feature_dim=repa_feature_dim,
            repa_student_layer=loss.repa_student_layer,
            repa_loss_weight=loss.repa_loss_weight,
            fm_mode=decoder.fm_mode,
            anchor_context=decoder.anchor_context,
            anchor_target=decoder.anchor_target,
            anchor_hidden_dim=decoder.anchor_hidden_dim,
            anchor_layers=decoder.anchor_layers,
            anchor_kernel_size=decoder.anchor_kernel_size,
            anchor_cosine_weight=decoder.anchor_cosine_weight,
            anchor_factor_weight=decoder.anchor_factor_weight,
            anchor_factor_temperature=decoder.anchor_factor_temperature,
        ),
        initialization=config.runtime.initialization,
        seed=seed,
        sampling=SamplingConfig(
            flow_steps=sampling.flow_steps,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            cfg_scale=sampling.cfg_scale,
        ),
    )


def build_repa_teacher(
    config: TrainConfig,
    backend: SemanticAcousticCodec,
    *,
    device: torch.device,
    route: Route,
) -> Teacher | None:
    weight = config.loss.repa_loss_weight
    if weight <= 0:
        return None
    if route is not Route.FM:
        raise ValueError("REPA requires the FM route.")
    if backend.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
        raise ValueError("REPA currently requires frame-aligned acoustic units.")
    decode = getattr(backend, "decode", None)
    if not callable(decode):
        raise TypeError("REPA teacher requires a backend that implements decode(codes).")
    teacher = config.loss.repa_teacher
    sample_rate = teacher.sample_rate
    return WavLMTeacher(
        cast(Any, backend),
        checkpoint=teacher.checkpoint,
        layer=teacher.layer,
        sample_rate=int(backend.sample_rate if sample_rate is None else sample_rate),
        device=device,
    )


def build_callbacks(
    config: TrainConfig,
    *,
    output_dir: Path,
    fixed_batch: GeneratorBatch,
) -> list[Callback]:
    callback_config = config.callback
    performance = callback_config.performance
    callbacks: list[Callback] = [
        ArtifactExport(output_dir),
        LearningRateMonitor(logging_interval="step"),
    ]
    callbacks.extend(_loss_callbacks(callback_config))

    codebook_usage_callback = _codebook_usage_callback(callback_config.codebook_usage)
    if codebook_usage_callback is not None:
        callbacks.append(codebook_usage_callback)

    performance_callback = _performance_callback(performance)
    if performance_callback is not None:
        callbacks.append(performance_callback)

    throughput_callback = _data_throughput_callback(
        callback_config.data_throughput,
    )
    if throughput_callback is not None:
        callbacks.append(throughput_callback)

    ema_callback = _ema_callback(config.pl_module, callback_config.ema)
    if ema_callback is not None:
        callbacks.append(ema_callback)

    sample_callback = _sample_callback(callback_config.sample, output_dir, fixed_batch)
    if sample_callback is not None:
        callbacks.append(sample_callback)

    checkpoint_callback = _checkpoint_callback(callback_config.checkpoint, output_dir)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)
    return callbacks


def _codebook_usage_callback(
    config: CodebookUsageCallbackConfig,
) -> Callback | None:
    if not config.enabled:
        return None
    return CodebookUsageLogger(every_n_steps=config.every_n_steps)


def _loss_callbacks(config: CallbackConfig) -> list[Callback]:
    callbacks: list[Callback] = []
    loss_summary = config.loss_summary
    if loss_summary.enabled:
        callbacks.append(LossSummaryCallback(window_capacity=loss_summary.window_capacity))

    loss_time_bucket = config.loss_time_bucket
    if loss_time_bucket.enabled:
        callbacks.append(
            LossTimeBucketLoggerCallback(
                item_name=loss_time_bucket.item_name,
                detail_key=loss_time_bucket.detail_key,
                histogram_tag=loss_time_bucket.histogram_tag,
                scalar_template=loss_time_bucket.scalar_template,
                every_n_steps=loss_time_bucket.every_n_steps,
                bucket_count=loss_time_bucket.bucket_count,
            )
        )
    return callbacks


def _performance_callback(
    config: PerformanceCallbackConfig,
) -> Callback | None:
    if not config.enabled:
        return None
    return PerformanceCallback(
        model_flops_per_step=config.model_flops_per_step,
        profile_flops=config.profile_flops,
        hardware_peak_flops=config.hardware_peak_flops,
        log_every_n_steps=config.log_every_n_steps,
        warmup_steps=config.warmup_steps,
        measure_window_steps=config.measure_window_steps,
    )


def _data_throughput_callback(
    config: DataThroughputCallbackConfig,
) -> Callback | None:
    if not config.enabled:
        return None
    return UnitThroughputCallback(
        unit_provider=SemanticFrameUnits(),
        log_every_n_steps=config.log_every_n_steps,
        warmup_steps=config.warmup_steps,
        measure_window_steps=config.measure_window_steps,
    )


def _ema_callback(config: PLModuleConfig, ema_config: EMACallbackConfig) -> Callback | None:
    ema_decay = config.ema_decay
    if ema_decay is None:
        return None
    return EMACallback(
        decay=ema_decay,
        update_after_step=config.ema_update_after_step,
        use_ema_weights=ema_config.use_ema_weights,
    )


def _sample_callback(
    config: SampleCallbackConfig,
    output_dir: Path,
    fixed_batch: GeneratorBatch,
) -> Callback | None:
    if not config.enabled:
        return None
    if not fixed_batch.has_reference:
        warnings.warn(
            "sample callback disabled because the fixed sample has no reference pair.",
            stacklevel=2,
        )
        return None
    return SampleLogger(
        output_dir,
        fixed_batch,
        SampleLogConfig(
            every_n_train_steps=config.every_n_train_steps,
            seed=config.seed,
        ),
    )


def _checkpoint_callback(
    config: CheckpointCallbackConfig,
    output_dir: Path,
) -> Callback | None:
    if not config.enabled:
        return None
    return ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=config.filename,
        save_last=config.save_last,
        save_top_k=config.save_top_k,
        every_n_train_steps=config.every_n_train_steps,
        auto_insert_metric_name=False,
    )


def load_checkpoint_feature_stats(
    path: str | None,
    *,
    feature_dim: int,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if path is None or not Path(path).is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("training checkpoint must contain a mapping.")
    state = checkpoint.get("state_dict")
    if state is None:
        return None
    if not isinstance(state, Mapping):
        raise TypeError("training checkpoint state_dict must be a mapping.")
    mean = state.get("support.feature_mean")
    std = state.get("support.feature_std")
    if mean is None or std is None:
        return None
    return (
        _checkpoint_feature_stat(mean, feature_dim=feature_dim, name="feature_mean"),
        _checkpoint_feature_stat(std, feature_dim=feature_dim, name="feature_std", positive=True),
    )


def _checkpoint_feature_stat(
    value: object,
    *,
    feature_dim: int,
    name: str,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"checkpoint {name} must be a floating-point Tensor.")
    flattened = value.detach().reshape(-1)
    if flattened.numel() != feature_dim:
        raise ValueError(
            f"checkpoint {name} has {flattened.numel()} values; expected {feature_dim}."
        )
    if not bool(torch.isfinite(flattened).all()):
        raise ValueError(f"checkpoint {name} must contain only finite values.")
    if positive and not bool((flattened > 0).all()):
        raise ValueError(f"checkpoint {name} must contain only positive values.")
    return tuple(float(item) for item in flattened)


def resolve_device(value: str | None) -> torch.device:
    requested = torch.device(
        "cuda" if value is None and torch.cuda.is_available() else (value or "cpu")
    )
    if requested.type == "cuda" and requested.index is None:
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return requested


__all__ = [
    "TrainingSession",
    "build_callbacks",
    "build_repa_teacher",
    "build_session",
    "build_support_config",
    "build_trainer",
    "load_checkpoint_feature_stats",
    "resolve_device",
    "run",
]
