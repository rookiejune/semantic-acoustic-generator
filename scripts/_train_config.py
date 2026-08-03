from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from semantic_acoustic_codec.config import (
    DecoderConfig,
    Initialization,
    Route,
    RVQPredictor,
)
from semantic_acoustic_codec.datamodule import DataConfig
from semantic_acoustic_codec.runtime import SamplingConfig

EnumT = TypeVar("EnumT", bound=Enum)


class _ConfigNode:
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass(frozen=True)
class BackendConfig(_ConfigNode):
    name: str = "longcat"
    model_dir: str | None = None
    revision: str | None = None
    local_files_only: bool = True
    allow_unpinned_revision: bool = False


@dataclass(frozen=True)
class DataModuleConfig(_ConfigNode, DataConfig):
    fixed_batch: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.fixed_batch, bool):
            raise TypeError("datamodule.fixed_batch must be a boolean.")


@dataclass(frozen=True)
class ModelConfig(_ConfigNode):
    route: Route = Route.FM
    condition_dim: int = 1024
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self) -> None:
        if isinstance(self.condition_dim, bool) or not isinstance(self.condition_dim, int):
            raise TypeError("model.condition_dim must be an integer.")
        if self.condition_dim <= 0:
            raise ValueError("model.condition_dim must be positive.")


@dataclass(frozen=True)
class RepaTeacherConfig(_ConfigNode):
    checkpoint: str = "microsoft/wavlm-base"
    layer: int = 9
    sample_rate: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError("loss.repa_teacher.checkpoint must be a non-empty string.")
        _positive_integer(self.layer, "loss.repa_teacher.layer")
        if self.sample_rate is not None:
            _positive_integer(self.sample_rate, "loss.repa_teacher.sample_rate")


@dataclass(frozen=True)
class LossConfig(_ConfigNode):
    repa_feature_dim: int | None = None
    repa_student_layer: int | None = None
    repa_loss_weight: float = 0.0
    repa_teacher: RepaTeacherConfig = field(default_factory=RepaTeacherConfig)

    def __post_init__(self) -> None:
        _optional_positive_integer(self.repa_feature_dim, "loss.repa_feature_dim")
        _optional_non_negative_integer(self.repa_student_layer, "loss.repa_student_layer")
        _non_negative_number(self.repa_loss_weight, "loss.repa_loss_weight")


@dataclass(frozen=True)
class PLModuleConfig(_ConfigNode):
    normalize_features: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    reference_dropout: float = 0.5
    validation_seed: int = 0
    ema_decay: float | None = None
    ema_update_after_step: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.normalize_features, bool):
            raise TypeError("pl_module.normalize_features must be a boolean.")
        _positive_number(self.learning_rate, "pl_module.learning_rate")
        _non_negative_number(self.weight_decay, "pl_module.weight_decay")
        _ratio(self.reference_dropout, "pl_module.reference_dropout")
        _non_negative_integer(self.validation_seed, "pl_module.validation_seed")
        _optional_positive_number(self.ema_decay, "pl_module.ema_decay")
        _non_negative_integer(self.ema_update_after_step, "pl_module.ema_update_after_step")


@dataclass(frozen=True)
class RuntimeConfig(_ConfigNode):
    device: str | None = None
    initialization: Initialization = Initialization.CODEC
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass(frozen=True)
class SampleCallbackConfig(_ConfigNode):
    enabled: bool = True
    every_n_train_steps: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        _boolean(self.enabled, "callback.sample.enabled")
        _positive_integer(self.every_n_train_steps, "callback.sample.every_n_train_steps")
        _non_negative_integer(self.seed, "callback.sample.seed")


@dataclass(frozen=True)
class PerformanceCallbackConfig(_ConfigNode):
    enabled: bool = True
    model_flops_per_step: float | None = None
    profile_flops: bool = False
    hardware_peak_flops: float | None = None
    log_every_n_steps: int = 100
    warmup_steps: int = 20
    measure_window_steps: int = 100

    def __post_init__(self) -> None:
        _boolean(self.enabled, "callback.performance.enabled")
        _optional_positive_number(
            self.model_flops_per_step,
            "callback.performance.model_flops_per_step",
        )
        _boolean(self.profile_flops, "callback.performance.profile_flops")
        _optional_positive_number(
            self.hardware_peak_flops,
            "callback.performance.hardware_peak_flops",
        )
        _positive_integer(self.log_every_n_steps, "callback.performance.log_every_n_steps")
        _non_negative_integer(self.warmup_steps, "callback.performance.warmup_steps")
        _positive_integer(self.measure_window_steps, "callback.performance.measure_window_steps")


@dataclass(frozen=True)
class DataThroughputCallbackConfig(_ConfigNode):
    enabled: bool = True
    log_every_n_steps: int = 100
    warmup_steps: int = 20
    measure_window_steps: int = 100

    def __post_init__(self) -> None:
        _boolean(self.enabled, "callback.data_throughput.enabled")
        _positive_integer(self.log_every_n_steps, "callback.data_throughput.log_every_n_steps")
        _non_negative_integer(self.warmup_steps, "callback.data_throughput.warmup_steps")
        _positive_integer(
            self.measure_window_steps,
            "callback.data_throughput.measure_window_steps",
        )


@dataclass(frozen=True)
class CodebookUsageCallbackConfig(_ConfigNode):
    enabled: bool = True
    every_n_steps: int = 100

    def __post_init__(self) -> None:
        _boolean(self.enabled, "callback.codebook_usage.enabled")
        _positive_integer(self.every_n_steps, "callback.codebook_usage.every_n_steps")


@dataclass(frozen=True)
class LossSummaryCallbackConfig(_ConfigNode):
    enabled: bool = True
    window_capacity: int = 20

    def __post_init__(self) -> None:
        _boolean(self.enabled, "callback.loss_summary.enabled")
        _positive_integer(self.window_capacity, "callback.loss_summary.window_capacity")


@dataclass(frozen=True)
class LossTimeBucketCallbackConfig(_ConfigNode):
    enabled: bool = True
    item_name: str = "flow"
    detail_key: str = "t"
    histogram_tag: str | None = "train/flow_loss/t_hist"
    scalar_template: str = "train/flow_loss/t/{lower:.1f}_{upper:.1f}"
    every_n_steps: int | None = 100
    bucket_count: int = 10

    def __post_init__(self) -> None:
        _boolean(self.enabled, "callback.loss_time_bucket.enabled")
        _non_empty_string(self.item_name, "callback.loss_time_bucket.item_name")
        _non_empty_string(self.detail_key, "callback.loss_time_bucket.detail_key")
        if self.histogram_tag is not None:
            _non_empty_string(self.histogram_tag, "callback.loss_time_bucket.histogram_tag")
        _non_empty_string(self.scalar_template, "callback.loss_time_bucket.scalar_template")
        _optional_positive_integer(self.every_n_steps, "callback.loss_time_bucket.every_n_steps")
        _positive_integer(self.bucket_count, "callback.loss_time_bucket.bucket_count")


@dataclass(frozen=True)
class CheckpointCallbackConfig(_ConfigNode):
    enabled: bool = True
    filename: str = "step-{step:08d}"
    save_last: bool = True
    save_top_k: int = -1
    every_n_train_steps: int = 10_000

    def __post_init__(self) -> None:
        _boolean(self.enabled, "callback.checkpoint.enabled")
        _non_empty_string(self.filename, "callback.checkpoint.filename")
        _boolean(self.save_last, "callback.checkpoint.save_last")
        _integer(self.save_top_k, "callback.checkpoint.save_top_k")
        _positive_integer(
            self.every_n_train_steps,
            "callback.checkpoint.every_n_train_steps",
        )


@dataclass(frozen=True)
class EMACallbackConfig(_ConfigNode):
    use_ema_weights: bool = True

    def __post_init__(self) -> None:
        _boolean(self.use_ema_weights, "callback.ema.use_ema_weights")


@dataclass(frozen=True)
class CallbackConfig(_ConfigNode):
    sample: SampleCallbackConfig = field(default_factory=SampleCallbackConfig)
    performance: PerformanceCallbackConfig = field(default_factory=PerformanceCallbackConfig)
    data_throughput: DataThroughputCallbackConfig = field(
        default_factory=DataThroughputCallbackConfig
    )
    codebook_usage: CodebookUsageCallbackConfig = field(default_factory=CodebookUsageCallbackConfig)
    loss_summary: LossSummaryCallbackConfig = field(default_factory=LossSummaryCallbackConfig)
    loss_time_bucket: LossTimeBucketCallbackConfig = field(
        default_factory=LossTimeBucketCallbackConfig
    )
    checkpoint: CheckpointCallbackConfig = field(default_factory=CheckpointCallbackConfig)
    ema: EMACallbackConfig = field(default_factory=EMACallbackConfig)


@dataclass(frozen=True)
class TrainerConfig(_ConfigNode):
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    use_distributed_sampler: bool = False
    precision: int | str = "bf16-mixed"
    max_steps: int = 1_000_000
    max_epochs: int = -1
    log_every_n_steps: int = 10
    gradient_clip_val: float = 1.0
    ckpt_path: str | None = None
    val_check_interval: int | float = 10_000
    check_val_every_n_epoch: int | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.accelerator, "trainer.accelerator")
        if isinstance(self.devices, bool) or not isinstance(self.devices, (int, str)):
            raise TypeError("trainer.devices must be an integer or string.")
        if isinstance(self.devices, int) and self.devices <= 0:
            raise ValueError("trainer.devices must be positive.")
        if isinstance(self.devices, str) and not self.devices:
            raise ValueError("trainer.devices must be a non-empty string.")
        _non_empty_string(self.strategy, "trainer.strategy")
        _boolean(self.use_distributed_sampler, "trainer.use_distributed_sampler")
        if isinstance(self.precision, bool) or not isinstance(self.precision, (int, str)):
            raise TypeError("trainer.precision must be an integer or string.")
        _integer(self.max_steps, "trainer.max_steps")
        _integer(self.max_epochs, "trainer.max_epochs")
        _positive_integer(self.log_every_n_steps, "trainer.log_every_n_steps")
        _non_negative_number(self.gradient_clip_val, "trainer.gradient_clip_val")
        if self.ckpt_path is not None:
            _non_empty_string(self.ckpt_path, "trainer.ckpt_path")
        _positive_number(self.val_check_interval, "trainer.val_check_interval")
        _optional_non_negative_integer(
            self.check_val_every_n_epoch,
            "trainer.check_val_every_n_epoch",
        )


@dataclass(frozen=True)
class TrainConfig(_ConfigNode):
    seed: int = 0
    output_dir: str | None = None
    output_subdir: str = "${backend.name}/${model.route}-${model.decoder.layers}l/${runtime.initialization}"
    backend: BackendConfig = field(default_factory=BackendConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    pl_module: PLModuleConfig = field(default_factory=PLModuleConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    callback: CallbackConfig = field(default_factory=CallbackConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self) -> None:
        _non_negative_integer(self.seed, "seed")
        if self.output_dir is not None:
            _non_empty_string(self.output_dir, "output_dir")
        _validate_output_subdir(self.output_subdir)
        if self.loss.repa_loss_weight > 0 and self.model.route is not Route.FM:
            raise ValueError("REPA requires model.route=fm.")


def parse_train_config(config: DictConfig | TrainConfig) -> TrainConfig:
    if isinstance(config, TrainConfig):
        return config
    prepared = _prepare(config)
    structured = OmegaConf.structured(TrainConfig)
    _writable(structured)
    merged = OmegaConf.merge(structured, prepared)
    OmegaConf.resolve(merged)
    return cast(TrainConfig, OmegaConf.to_object(merged))


def output_dir(config: TrainConfig) -> Path:
    if config.output_dir is not None:
        return Path(config.output_dir).expanduser()
    root = os.environ.get("SEMANTIC_ACOUSTIC_CODEC_TRAIN_ROOT")
    if root is None:
        dynamic = os.environ.get("DYNAMIC_HOME")
        root = (
            "/tmp/semantic-acoustic-codec"
            if dynamic is None
            else f"{dynamic}/train/semantic-acoustic-codec"
        )
    return Path(root).expanduser() / config.output_subdir


def _prepare(config: DictConfig) -> DictConfig:
    result = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))
    OmegaConf.resolve(result)
    model = result.get("model")
    if isinstance(model, DictConfig):
        route = model.get("route")
        if route is not None:
            model.route = _enum_name(Route, route)
        decoder = model.get("decoder")
        if isinstance(decoder, DictConfig):
            predictor = decoder.get("rvq_predictor")
            if predictor is not None:
                decoder.rvq_predictor = _enum_name(RVQPredictor, predictor)
    runtime = result.get("runtime")
    if isinstance(runtime, DictConfig):
        initialization = runtime.get("initialization")
        if initialization is not None:
            runtime.initialization = _enum_name(Initialization, initialization)
    return result


def _writable(config: DictConfig | ListConfig) -> None:
    OmegaConf.set_readonly(config, False)
    values = config.values() if isinstance(config, DictConfig) else config
    for value in values:
        if isinstance(value, (DictConfig, ListConfig)):
            _writable(value)


def _enum_name(enum: type[EnumT], value: object) -> str:
    raw = str(value)
    if raw in enum.__members__:
        return raw
    return enum(raw).name


def _validate_output_subdir(value: str) -> None:
    _non_empty_string(value, "output_subdir")
    path = Path(value)
    if path.is_absolute():
        raise ValueError("output_subdir must be relative.")
    if any(part == ".." for part in path.parts):
        raise ValueError("output_subdir must not contain '..'.")


def _boolean(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def _integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


def _positive_integer(value: object, name: str) -> None:
    _integer(value, name)
    number = cast(int, value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")


def _non_negative_integer(value: object, name: str) -> None:
    _integer(value, name)
    number = cast(int, value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative.")


def _optional_positive_integer(value: object, name: str) -> None:
    if value is None:
        return
    _positive_integer(value, name)


def _optional_non_negative_integer(value: object, name: str) -> None:
    if value is None:
        return
    _non_negative_integer(value, name)


def _positive_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _non_negative_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _optional_positive_number(value: object, name: str) -> None:
    if value is None:
        return
    _positive_number(value, name)


def _ratio(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1].")
