from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import hydra
import torch
from anytrain.codec import load_semantic_acoustic
from anytrain.lightning import ModelCheckpoint, PerformanceCallback
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback

from semantic_acoustic_codec.callback import ArtifactExport, SampleLogConfig, SampleLogger
from semantic_acoustic_codec.config import (
    DecoderConfig,
    Initialization,
    Route,
    RVQPredictor,
)
from semantic_acoustic_codec.datamodule import (
    BatchingConfig,
    DataConfig,
    DataModule,
    load_batch,
    single_batch_loader,
)
from semantic_acoustic_codec.pl_module import build_module, dataset_feature_stats
from semantic_acoustic_codec.runtime import SamplingConfig, SemanticSupportConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    seed = int(config.seed)
    pl.seed_everything(seed, workers=True)
    device = _device(cast(Optional[str], config.runtime.get("device")))
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    codec = str(config.backend.name)
    data = _data_config(config.datamodule)
    backend = load_semantic_acoustic(codec, device=device)
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
    if bool(config.datamodule.fixed_batch):
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
    support_config = _support_config(config, seed=seed)
    normalize_features = bool(config.pl_module.normalize_features)
    feature_mean: tuple[float, ...] | None = None
    feature_std: tuple[float, ...] | None = None
    if normalize_features and support_config.route is Route.FM:
        if bool(config.datamodule.fixed_batch):
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
        learning_rate=float(config.pl_module.learning_rate),
        weight_decay=float(config.pl_module.weight_decay),
        reference_dropout=float(config.pl_module.reference_dropout),
        validation_seed=int(config.pl_module.get("validation_seed", seed)),
    )

    callbacks: list[Callback] = [ArtifactExport(output_dir)]
    performance = config.callback.performance
    if bool(performance.get("enabled", True)):
        callbacks.append(
            PerformanceCallback(
                model_flops_per_step=_optional_float(performance.get("model_flops_per_step")),
                profile_flops=bool(performance.get("profile_flops", False)),
                hardware_peak_flops=_optional_float(performance.get("hardware_peak_flops")),
                log_every_n_steps=int(
                    performance.get("log_every_n_steps", config.trainer.log_every_n_steps)
                ),
                warmup_steps=int(performance.get("warmup_steps", 20)),
                measure_window_steps=int(performance.get("measure_window_steps", 100)),
            )
        )
    sample = config.callback.sample
    if bool(sample.enabled) and fixed_batch.has_reference:
        callbacks.append(
            SampleLogger(
                output_dir,
                fixed_batch,
                SampleLogConfig(
                    every_n_train_steps=int(sample.every_n_train_steps),
                    seed=int(sample.seed),
                ),
            )
        )
    elif bool(sample.enabled):
        warnings.warn(
            "sample callback disabled because the fixed sample has no reference pair.",
            stacklevel=2,
        )
    checkpoint = config.callback.checkpoint
    if bool(checkpoint.enabled):
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename=str(checkpoint.filename),
                save_last=bool(checkpoint.save_last),
                save_top_k=int(checkpoint.save_top_k),
                every_n_train_steps=int(checkpoint.every_n_train_steps),
                auto_insert_metric_name=False,
            )
        )

    ckpt_path = _ckpt_path(config)
    trainer = pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=cast(Union[str, int], config.trainer.devices),
        strategy=str(config.trainer.strategy),
        precision=cast(Any, config.trainer.precision),
        max_steps=int(config.trainer.max_steps),
        max_epochs=int(config.trainer.max_epochs),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(checkpoint.enabled),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        use_distributed_sampler=bool(config.trainer.use_distributed_sampler),
        val_check_interval=cast(
            Union[int, float],
            config.trainer.get("val_check_interval", 1.0),
        ),
        check_val_every_n_epoch=cast(
            Optional[int],
            config.trainer.get("check_val_every_n_epoch"),
        ),
    )
    if not bool(config.datamodule.fixed_batch):
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


def _support_config(config: DictConfig, *, seed: int) -> SemanticSupportConfig:
    decoder = config.model.decoder
    loss = config.loss
    sampling = config.runtime.sampling
    return SemanticSupportConfig(
        route=_route(config.model.route),
        condition_dim=int(config.model.condition_dim),
        decoder=DecoderConfig(
            hidden_dim=cast(Optional[int], decoder.get("hidden_dim")),
            layers=int(decoder.layers),
            heads=int(decoder.heads),
            ffn_ratio=int(decoder.ffn_ratio),
            rvq_predictor=_rvq_predictor(decoder.get("rvq_predictor", "mtp")),
            mtp_layers=int(decoder.get("mtp_layers", 2)),
            mtp_heads=int(decoder.get("mtp_heads", 4)),
            repa_feature_dim=cast(Optional[int], loss.get("repa_feature_dim")),
            repa_student_layer=cast(Optional[int], loss.get("repa_student_layer")),
            repa_loss_weight=float(loss.repa_loss_weight),
        ),
        initialization=_initialization(config.runtime.initialization),
        seed=seed,
        sampling=SamplingConfig(
            flow_steps=int(sampling.flow_steps),
            temperature=float(sampling.temperature),
            top_p=float(sampling.top_p),
        ),
    )


def _data_config(config: DictConfig) -> DataConfig:
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


def _output_dir(config: DictConfig) -> Path:
    value = config.get("output_dir")
    if value is not None:
        return Path(str(value)).expanduser()
    root = os.environ.get("SEMANTIC_ACOUSTIC_CODEC_TRAIN_ROOT")
    if root is None:
        dynamic = os.environ.get("DYNAMIC_HOME")
        root = "/tmp/semantic-acoustic-codec" if dynamic is None else f"{dynamic}/train/semantic-acoustic-codec"
    return Path(root).expanduser() / str(config.output_subdir)


def _ckpt_path(config: DictConfig) -> str | None:
    value = config.trainer.get("ckpt_path")
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _device(value: str | None) -> torch.device:
    requested = torch.device("cuda" if value is None and torch.cuda.is_available() else (value or "cpu"))
    if requested.type == "cuda" and requested.index is None:
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return requested


def _route(value: Any) -> Route:
    raw = str(value)
    return Route[raw] if raw in Route.__members__ else Route(raw)


def _initialization(value: Any) -> Initialization:
    raw = str(value)
    return Initialization[raw] if raw in Initialization.__members__ else Initialization(raw)


def _rvq_predictor(value: Any) -> RVQPredictor:
    raw = str(value)
    return RVQPredictor[raw] if raw in RVQPredictor.__members__ else RVQPredictor(raw)


if __name__ == "__main__":
    main()
