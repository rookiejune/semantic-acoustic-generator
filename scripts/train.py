from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import hydra
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

from semantic_acoustic_codec.backend import load_backend
from semantic_acoustic_codec.callback import (
    ArtifactExport,
    CodebookUsageLogger,
    SampleLogConfig,
    SampleLogger,
    SemanticFrameUnits,
    UnitThroughputCallback,
)
from semantic_acoustic_codec.config import (
    DecoderConfig,
    Route,
)
from semantic_acoustic_codec.datamodule import (
    BatchingConfig,
    DataConfig,
    DataModule,
    load_batch,
    single_batch_loader,
)
from semantic_acoustic_codec.loss import WavLMTeacher
from semantic_acoustic_codec.pl_module import build_module, dataset_feature_stats
from semantic_acoustic_codec.runtime import SamplingConfig, SemanticSupportConfig

if __package__:
    from ._train_config import TrainConfig, output_dir, parse_train_config
else:
    from _train_config import TrainConfig, output_dir, parse_train_config

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec
    from omegaconf import DictConfig

    from semantic_acoustic_codec.loss.repa import Teacher
    from semantic_acoustic_codec.types import SemanticCodecBatch


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig | TrainConfig) -> None:
    config = parse_train_config(config)
    seed = config.seed
    pl.seed_everything(seed, workers=True)
    device = _device(config.runtime.device)
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    repa_teacher = _repa_teacher(config, backend, device=device, route=route)
    support_config = _support_config(config, seed=seed, repa_teacher=repa_teacher)
    normalize_features = config.pl_module.normalize_features
    feature_mean: tuple[float, ...] | None = None
    feature_std: tuple[float, ...] | None = None
    if normalize_features and support_config.route is Route.FM:
        if config.datamodule.fixed_batch:
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
        repa_teacher=repa_teacher,
    )

    checkpoint = config.callback.checkpoint
    callbacks = _build_callbacks(config, output_dir=output_dir, fixed_batch=fixed_batch)

    ckpt_path = _ckpt_path(config)
    trainer = pl.Trainer(
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
    if not config.datamodule.fixed_batch:
        trainer.fit(
            module,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
    else:
        val_dataloaders = None
        if data.validation_split is not None:
            data_module.setup("fit")
            val_dataloaders = data_module.val_dataloader()
        trainer.fit(
            module,
            train_dataloaders=single_batch_loader(fixed_batch),
            val_dataloaders=val_dataloaders,
            ckpt_path=ckpt_path,
        )


def _support_config(
    config: TrainConfig,
    *,
    seed: int,
    repa_teacher: Teacher | None,
) -> SemanticSupportConfig:
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
    return SemanticSupportConfig(
        route=config.model.route,
        condition_dim=config.model.condition_dim,
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


def _repa_teacher(
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


def _build_callbacks(
    config: Any,
    *,
    output_dir: Path,
    fixed_batch: SemanticCodecBatch,
) -> list[Callback]:
    callback_config = config.callback
    performance = callback_config.performance
    callbacks: list[Callback] = [
        ArtifactExport(output_dir),
        LearningRateMonitor(logging_interval="step"),
    ]
    callbacks.extend(_loss_callbacks(callback_config))

    codebook_usage_callback = _codebook_usage_callback(
        callback_config.get("codebook_usage"),
        trainer_log_every_n_steps=config.trainer.log_every_n_steps,
    )
    if codebook_usage_callback is not None:
        callbacks.append(codebook_usage_callback)

    performance_callback = _performance_callback(
        performance,
        trainer_log_every_n_steps=config.trainer.log_every_n_steps,
    )
    if performance_callback is not None:
        callbacks.append(performance_callback)

    throughput_callback = _data_throughput_callback(
        callback_config.get("data_throughput"),
        performance=performance,
        trainer_log_every_n_steps=config.trainer.log_every_n_steps,
    )
    if throughput_callback is not None:
        callbacks.append(throughput_callback)

    ema_callback = _ema_callback(config.pl_module, callback_config.get("ema"))
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
    config: Any,
    *,
    trainer_log_every_n_steps: Any,
) -> Callback | None:
    if config is not None and not bool(config.get("enabled", True)):
        return None
    usage = {} if config is None else config
    return CodebookUsageLogger(
        every_n_steps=int(usage.get("every_n_steps", trainer_log_every_n_steps))
    )


def _loss_callbacks(config: Any) -> list[Callback]:
    callbacks: list[Callback] = []
    loss_summary = config.get("loss_summary")
    if loss_summary is None or bool(loss_summary.get("enabled", True)):
        summary = {} if loss_summary is None else loss_summary
        callbacks.append(LossSummaryCallback(window_capacity=int(summary.get("window_capacity", 20))))

    loss_time_bucket = config.get("loss_time_bucket")
    if loss_time_bucket is None or bool(loss_time_bucket.get("enabled", True)):
        bucket = {} if loss_time_bucket is None else loss_time_bucket
        every_n_steps = bucket.get("every_n_steps", 100)
        callbacks.append(
            LossTimeBucketLoggerCallback(
                item_name=str(bucket.get("item_name", "flow")),
                detail_key=str(bucket.get("detail_key", "t")),
                histogram_tag=cast(
                    Optional[str],
                    bucket.get("histogram_tag", "train/flow_loss/t_hist"),
                ),
                scalar_template=str(
                    bucket.get(
                        "scalar_template",
                        "train/flow_loss/t/{lower:.1f}_{upper:.1f}",
                    )
                ),
                every_n_steps=None if every_n_steps is None else int(every_n_steps),
                bucket_count=int(bucket.get("bucket_count", 10)),
            )
        )
    return callbacks


def _performance_callback(
    config: Any,
    *,
    trainer_log_every_n_steps: Any,
) -> Callback | None:
    if not bool(config.get("enabled", True)):
        return None
    return PerformanceCallback(
        model_flops_per_step=_optional_float(config.get("model_flops_per_step")),
        profile_flops=bool(config.get("profile_flops", False)),
        hardware_peak_flops=_optional_float(config.get("hardware_peak_flops")),
        log_every_n_steps=int(config.get("log_every_n_steps", trainer_log_every_n_steps)),
        warmup_steps=int(config.get("warmup_steps", 20)),
        measure_window_steps=int(config.get("measure_window_steps", 100)),
    )


def _data_throughput_callback(
    config: Any,
    *,
    performance: Any,
    trainer_log_every_n_steps: Any,
) -> Callback | None:
    if config is not None and not bool(config.get("enabled", True)):
        return None
    throughput = {} if config is None else config
    return UnitThroughputCallback(
        unit_provider=SemanticFrameUnits(),
        log_every_n_steps=int(
            throughput.get(
                "log_every_n_steps",
                performance.get("log_every_n_steps", trainer_log_every_n_steps),
            )
        ),
        warmup_steps=int(throughput.get("warmup_steps", performance.get("warmup_steps", 20))),
        measure_window_steps=int(
            throughput.get(
                "measure_window_steps",
                performance.get("measure_window_steps", 100),
            )
        ),
    )


def _ema_callback(config: Any, ema_config: Any) -> Callback | None:
    ema_decay = config.get("ema_decay")
    if ema_decay is None:
        return None
    return EMACallback(
        decay=float(ema_decay),
        update_after_step=int(config.get("ema_update_after_step", 0)),
        use_ema_weights=bool(True if ema_config is None else ema_config.get("use_ema_weights", True)),
    )


def _sample_callback(
    config: Any,
    output_dir: Path,
    fixed_batch: SemanticCodecBatch,
) -> Callback | None:
    if not bool(config.enabled):
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
            every_n_train_steps=int(config.every_n_train_steps),
            seed=int(config.seed),
        ),
    )


def _checkpoint_callback(config: Any, output_dir: Path) -> Callback | None:
    if not bool(config.enabled):
        return None
    return ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=str(config.filename),
        save_last=bool(config.save_last),
        save_top_k=int(config.save_top_k),
        every_n_train_steps=int(config.every_n_train_steps),
        auto_insert_metric_name=False,
    )


def _data_config(config: Any) -> DataConfig:
    if isinstance(config, DataConfig):
        return config
    batching = config.batching
    return DataConfig(
        source=str(config.get("source", "qwen_cross_text")),
        root=cast(Optional[str], config.get("root")),
        split=str(config.split),
        validation_split=cast(Optional[str], config.get("validation_split")),
        validation_sample_limit=cast(Optional[int], config.get("validation_sample_limit")),
        role=str(config.get("role", "target")),
        speaker_id=str(config.get("speaker_id", "vivian")),
        sample_index=int(config.sample_index),
        max_seconds=cast(Optional[float], config.get("max_seconds")),
        overlong=str(config.overlong),
        sample_limit=cast(Optional[int], config.get("sample_limit")),
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers),
        batching=BatchingConfig(
            enabled=bool(batching.enabled),
            max_batch_seconds=float(batching.max_batch_seconds),
            planning_window=int(batching.planning_window),
            prefetch_factor=int(batching.prefetch_factor),
            drop_distributed_tail=bool(batching.drop_distributed_tail),
            seed=int(batching.seed),
        ),
    )


def _output_dir(config: TrainConfig) -> Path:
    return output_dir(config)


def _ckpt_path(config: TrainConfig) -> str | None:
    return config.trainer.ckpt_path


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _device(value: str | None) -> torch.device:
    requested = torch.device("cuda" if value is None and torch.cuda.is_available() else (value or "cpu"))
    if requested.type == "cuda" and requested.index is None:
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return requested


if __name__ == "__main__":
    main()
