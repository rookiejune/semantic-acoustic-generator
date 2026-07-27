from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

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
    DataConfig,
    DataModule,
    LBAConfig,
    load_batch,
    single_batch_loader,
)
from semantic_acoustic_codec.pl_module import build_module
from semantic_acoustic_codec.runtime import SamplingConfig, SemanticSupportConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    seed = int(config.train.seed)
    pl.seed_everything(seed, workers=True)
    device = _device(cast(Optional[str], config.get("device")))
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    codec = str(config.get("codec", "longcat"))
    data = _data_config(config.data)
    _validate_data_backend(data, codec=codec)
    backend = load_semantic_acoustic(codec, device=device)
    semantic_pad_id = int(backend.semantic_codebook.size(0))
    acoustic_pad_ids = backend.acoustic_codebook_sizes
    fixed_batch = load_batch(
        data,
        codec=codec,
        frame_rate=backend.frame_rate,
        acoustic_layout=backend.acoustic_layout,
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=acoustic_pad_ids,
    )
    support_config = _support_config(config, seed=seed)
    module = build_module(
        backend,
        support_config,
        fixed_batch,
        normalize_features=bool(config.get("normalize_features", True)),
        learning_rate=float(config.optimizer.learning_rate),
        weight_decay=float(config.optimizer.weight_decay),
        reference_dropout=float(config.train.reference_dropout),
    )

    callbacks: list[Callback] = [ArtifactExport(output_dir)]
    performance = config.get("performance", {})
    if bool(performance.get("enabled", True)):
        callbacks.append(
            PerformanceCallback(
                model_flops_per_step=_optional_float(performance.get("model_flops_per_step")),
                hardware_peak_flops=_optional_float(performance.get("hardware_peak_flops")),
                log_every_n_steps=int(
                    performance.get("log_every_n_steps", config.trainer.log_every_n_steps)
                ),
                warmup_steps=int(performance.get("warmup_steps", 20)),
                measure_window_steps=int(performance.get("measure_window_steps", 100)),
            )
        )
    if bool(config.sample.enabled):
        callbacks.append(
            SampleLogger(
                output_dir,
                fixed_batch,
                SampleLogConfig(
                    every_n_train_steps=int(config.sample.every_n_train_steps),
                    seed=int(config.sample.seed),
                ),
            )
        )
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

    ckpt_path = _ckpt_path(config)
    trainer = pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=cast(Union[str, int], config.trainer.devices),
        strategy=str(config.trainer.strategy),
        precision=cast(Any, config.trainer.precision),
        max_steps=int(config.train.max_steps),
        max_epochs=int(config.trainer.max_epochs),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(config.trainer.enable_checkpointing),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        use_distributed_sampler=bool(config.trainer.use_distributed_sampler),
    )
    if not bool(config.train.fixed_batch):
        trainer.fit(
            module,
            datamodule=DataModule(
                data,
                codec=codec,
                acoustic_layout=backend.acoustic_layout,
                frame_rate=backend.frame_rate,
                output_dir=output_dir,
                semantic_pad_id=semantic_pad_id,
                acoustic_pad_ids=acoustic_pad_ids,
            ),
            ckpt_path=ckpt_path,
        )
    else:
        trainer.fit(
            module,
            train_dataloaders=single_batch_loader(fixed_batch),
            ckpt_path=ckpt_path,
        )


def _support_config(config: DictConfig, *, seed: int) -> SemanticSupportConfig:
    decoder = config.decoder
    sampling = config.sampling
    return SemanticSupportConfig(
        route=_route(config.route),
        condition_dim=int(config.condition_dim),
        decoder=DecoderConfig(
            hidden_dim=cast(Optional[int], decoder.get("hidden_dim")),
            layers=int(decoder.layers),
            heads=int(decoder.heads),
            ffn_ratio=int(decoder.ffn_ratio),
            rvq_predictor=_rvq_predictor(decoder.get("rvq_predictor", "mtp")),
            mtp_layers=int(decoder.get("mtp_layers", 2)),
            mtp_heads=int(decoder.get("mtp_heads", 4)),
            repa_feature_dim=cast(Optional[int], decoder.get("repa_feature_dim")),
            repa_student_layer=cast(Optional[int], decoder.get("repa_student_layer")),
            repa_loss_weight=float(decoder.get("repa_loss_weight", 0.0)),
        ),
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
        source=str(config.get("source", "qwen_cross_text")),
        root=cast(Optional[str], config.get("root")),
        split=str(config.split),
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
        lba=LBAConfig(
            enabled=bool(lba.enabled),
            max_batch_seconds=float(lba.max_batch_seconds),
            max_padding_ratio=float(lba.max_padding_ratio),
            prefetch_batches=int(lba.prefetch_batches),
            planner_mode=cast(Literal["quality", "throughput"], str(lba.planner_mode)),
            drop_last_flush=bool(lba.drop_last_flush),
        ),
    )


def _validate_data_backend(data: DataConfig, *, codec: str) -> None:
    if data.source == "wmt19_tts_codec" and codec != "longcat":
        raise NotImplementedError(
            "wmt19_tts_codec training currently parses the prepared LongCat [frame, codebook] "
            "view only. Use codec='longcat' or add a backend-native structured data source."
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
    value = config.checkpoint.get("resume_from")
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
