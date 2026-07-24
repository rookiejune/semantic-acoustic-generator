from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from semantic_acoustic_codec.backend import LongCatBackend
from semantic_acoustic_codec.callback import ArtifactExport
from semantic_acoustic_codec.config import AdapterType, DecoderConfig, Initialization, Route
from semantic_acoustic_codec.datamodule import (
    DataConfig,
    DataModule,
    LBAConfig,
    collate_codes,
    load_codes,
    single_batch_loader,
)
from semantic_acoustic_codec.pl_module import build_module
from semantic_acoustic_codec.runtime import SamplingConfig, SemanticSupportConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    seed = int(config.train.seed)
    pl.seed_everything(seed, workers=True)
    device = _device(cast(str | None, config.get("device")))
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = LongCatBackend.from_pretrained(device=str(device))
    data = _data_config(config.data)
    fixed_codes = load_codes(data, frame_rate=backend.frame_rate)
    fixed_batch = collate_codes([fixed_codes])
    support_config = _support_config(config, seed=seed)
    module = build_module(
        backend,
        support_config,
        fixed_batch,
        normalize_features=bool(config.get("normalize_features", True)),
        learning_rate=float(config.optimizer.learning_rate),
        weight_decay=float(config.optimizer.weight_decay),
    )

    callbacks = [ArtifactExport(output_dir)]
    if bool(config.trainer.enable_checkpointing):
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename=str(config.checkpoint.filename),
                save_last=bool(config.checkpoint.save_last),
                save_top_k=int(config.checkpoint.save_top_k),
                every_n_train_steps=int(config.checkpoint.every_n_train_steps),
                auto_insert_metric_name=False,
            )
        )

    trainer = pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=cast(str | int, config.trainer.devices),
        strategy=str(config.trainer.strategy),
        precision=cast(str, config.trainer.precision),
        max_steps=int(config.train.max_steps),
        max_epochs=int(config.trainer.max_epochs),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(config.trainer.enable_checkpointing),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        use_distributed_sampler=bool(config.trainer.use_distributed_sampler),
    )
    if data.lba.enabled:
        trainer.fit(module, datamodule=DataModule(data, frame_rate=backend.frame_rate, output_dir=output_dir))
    else:
        trainer.fit(module, train_dataloaders=single_batch_loader(fixed_codes))


def _support_config(config: DictConfig, *, seed: int) -> SemanticSupportConfig:
    decoder = config.decoder
    sampling = config.sampling
    adapter = config.get("adapter")
    return SemanticSupportConfig(
        route=_route(config.route),
        condition_dim=int(config.condition_dim),
        decoder=DecoderConfig(
            hidden_dim=cast(int | None, decoder.get("hidden_dim")),
            layers=int(decoder.layers),
            heads=int(decoder.heads),
            ffn_ratio=int(decoder.ffn_ratio),
            repa_feature_dim=cast(int | None, decoder.get("repa_feature_dim")),
            repa_student_layer=cast(int | None, decoder.get("repa_student_layer")),
            repa_loss_weight=float(decoder.get("repa_loss_weight", 0.0)),
        ),
        adapter=None if adapter is None else _adapter(adapter),
        initialization=_initialization(config.initialization),
        seed=seed,
        sampling=SamplingConfig(
            flow_steps=int(sampling.flow_steps),
            temperature=float(sampling.temperature),
            top_p=float(sampling.top_p),
        ),
    )


def _data_config(config: DictConfig) -> DataConfig:
    lba = config.lba
    return DataConfig(
        root=cast(str | None, config.get("root")),
        split=str(config.split),
        sample_index=int(config.sample_index),
        max_seconds=cast(float | None, config.get("max_seconds")),
        overlong=str(config.overlong),
        sample_limit=cast(int | None, config.get("sample_limit")),
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers),
        lba=LBAConfig(
            enabled=bool(lba.enabled),
            max_batch_seconds=float(lba.max_batch_seconds),
            max_padding_ratio=float(lba.max_padding_ratio),
            prefetch_batches=int(lba.prefetch_batches),
            planner_mode=str(lba.planner_mode),
            drop_last_flush=bool(lba.drop_last_flush),
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


def _device(value: str | None) -> torch.device:
    requested = torch.device("cuda" if value is None and torch.cuda.is_available() else (value or "cpu"))
    if requested.type == "cuda" and requested.index is None:
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return requested


def _route(value: Any) -> Route:
    raw = str(value)
    return Route[raw] if raw in Route.__members__ else Route(raw)


def _adapter(value: Any) -> AdapterType:
    raw = str(value)
    return AdapterType[raw] if raw in AdapterType.__members__ else AdapterType(raw)


def _initialization(value: Any) -> Initialization:
    raw = str(value)
    return Initialization[raw] if raw in Initialization.__members__ else Initialization(raw)


if __name__ == "__main__":
    main()
