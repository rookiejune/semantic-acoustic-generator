from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from semantic_acoustic_generator.backend import BackendConfig
from semantic_acoustic_generator.config import (
    AnchorContext,
    DecoderConfig,
    FeatureAdapter,
    FMMode,
    Route,
    RVQPredictor,
    decoder_options,
)
from semantic_acoustic_generator.training import TrainConfig, parse_train_config

CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    "field",
    [
        "hidden_dim",
        "layers",
        "heads",
        "ffn_ratio",
        "mtp_layers",
        "mtp_heads",
        "repa_feature_dim",
        "repa_student_layer",
    ],
)
@pytest.mark.parametrize("value", [True, "4"])
def test_decoder_config_rejects_non_integer_fields(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        DecoderConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, "0.1"])
def test_decoder_config_rejects_non_numeric_repa_weight(value: object) -> None:
    with pytest.raises(TypeError, match="repa_loss_weight"):
        DecoderConfig(repa_loss_weight=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_decoder_config_rejects_non_finite_repa_weight(value: float) -> None:
    with pytest.raises(ValueError, match="repa_loss_weight must be finite"):
        DecoderConfig(repa_loss_weight=value)


def test_decoder_config_requires_declared_predictor() -> None:
    with pytest.raises(TypeError, match="rvq_predictor must be an RVQPredictor"):
        DecoderConfig(rvq_predictor="mtp")  # type: ignore[arg-type]


def test_decoder_options_rejects_non_finite_repa_weight() -> None:
    with pytest.raises(ValueError, match="repa_loss_weight must be finite"):
        decoder_options(
            {
                "layers": 8,
                "heads": 8,
                "ffn_ratio": 4,
                "repa_loss_weight": float("nan"),
            }
        )


def test_transformer_anchor_requires_heads_to_divide_hidden_dim() -> None:
    with pytest.raises(ValueError, match="divisible by heads"):
        DecoderConfig(
            heads=3,
            anchor_context=AnchorContext.TRANSFORMER,
            anchor_hidden_dim=8,
        )


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="train", overrides=list(overrides))


@pytest.mark.parametrize(
    ("experiment", "backend", "route", "max_steps"),
    [
        ("001_longcat_fm", "longcat", "fm", 1_000_000),
        ("001_longcat_rvq", "longcat", "rvq", 1_000_000),
        ("smoke", "longcat", "fm", 2),
        ("smoke32", "longcat", "fm", 4),
        ("overfit", "longcat", "fm", 20),
        ("screening", "longcat", "fm", -1),
    ],
)
def test_experiment_composes_src_aligned_groups(
    experiment: str,
    backend: str,
    route: str,
    max_steps: int,
) -> None:
    config = _compose(f"experiment={experiment}")

    assert config.backend.name == backend
    assert config.datamodule.source == "qwen_cross_text"
    assert config.model.route == route
    assert config.trainer.max_steps == max_steps


def test_root_has_no_legacy_flat_training_groups() -> None:
    config = _compose()

    for key in ("codec", "data", "decoder", "optimizer", "sampling", "train"):
        assert key not in config
    assert config.seed == 0
    assert config.datamodule.fixed_batch is False
    assert config.pl_module.reference_dropout == 0.5
    assert config.pl_module.finite_loss_check_interval == 100
    assert config.runtime.sampling.cfg_scale == 1.0
    assert config.callback.checkpoint.enabled is True
    assert config.callback.performance.profile_flops is False
    assert config.callback.data_throughput.enabled is True


def test_longcat_fm_uses_24gb_single_gpu_batching_defaults() -> None:
    config = _compose("experiment=001_longcat_fm")

    assert config.datamodule.batch_size == 48
    assert config.datamodule.batching.max_batch_seconds == 576.0


def test_train_config_parses_to_typed_entry_schema() -> None:
    config = parse_train_config(_compose())

    assert isinstance(config, TrainConfig)
    assert isinstance(config.backend, BackendConfig)
    assert config.model.route is Route.FM
    assert config.model.feature_adapter is FeatureAdapter.NONE
    assert config.model.decoder.rvq_predictor is RVQPredictor.MTP
    assert config.datamodule.fixed_batch is False
    assert config.callback.performance.enabled is False
    assert config.runtime.sampling.cfg_scale == 1.0
    assert config.output_subdir == "longcat/fm-8l/codec"


@pytest.mark.parametrize("experiment", ["smoke", "smoke32", "overfit"])
def test_debug_experiments_check_finite_loss_every_step(experiment: str) -> None:
    config = _compose(f"experiment={experiment}")

    assert config.pl_module.finite_loss_check_interval == 1


def test_train_config_rejects_non_positive_finite_loss_interval() -> None:
    raw = _compose("pl_module.finite_loss_check_interval=0")

    with pytest.raises(ValueError, match="finite_loss_check_interval"):
        parse_train_config(raw)


def test_train_config_preserves_false_backend_boolean() -> None:
    config = parse_train_config(_compose("+backend.local_files_only=false"))

    assert config.backend.local_files_only is False


def test_train_config_rejects_unknown_fields_before_training() -> None:
    raw = _compose("+callback.unused=1")

    with pytest.raises(Exception) as raised:
        parse_train_config(raw)

    assert "callback.unused" in str(raised.value)


def test_train_config_rejects_unsafe_output_subdir() -> None:
    raw = _compose("output_subdir=../bad")

    with pytest.raises(ValueError, match="output_subdir"):
        parse_train_config(raw)


def test_train_config_rejects_repa_on_rvq_route() -> None:
    raw = _compose(
        "model/route=rvq",
        "loss.repa_loss_weight=0.1",
        "loss.repa_feature_dim=768",
    )

    with pytest.raises(ValueError, match="REPA requires model.route=fm"):
        parse_train_config(raw)


def test_train_config_parses_longcat_first_codebook_adapter() -> None:
    config = parse_train_config(
        _compose("model.feature_adapter=longcat_first_codebook")
    )

    assert config.model.feature_adapter is FeatureAdapter.LONGCAT_FIRST_CODEBOOK


def test_train_config_parses_first_codebook_anchor_mode() -> None:
    config = parse_train_config(
        _compose(
            "model.feature_adapter=longcat_first_codebook",
            "model.decoder.fm_mode=anchor",
            "model.decoder.anchor_hidden_dim=256",
            "model.decoder.anchor_layers=2",
            "model.decoder.anchor_context=transformer",
        )
    )

    assert config.model.decoder.fm_mode is FMMode.ANCHOR
    assert config.model.decoder.anchor_hidden_dim == 256
    assert config.model.decoder.anchor_layers == 2
    assert config.model.decoder.anchor_context is AnchorContext.TRANSFORMER


def test_longcat_transformer_anchor_experiment_is_frame_aligned() -> None:
    config = parse_train_config(_compose("experiment=011_longcat_transformer_anchor"))

    assert config.model.feature_adapter is FeatureAdapter.LONGCAT_FIRST_CODEBOOK
    assert config.model.decoder.fm_mode is FMMode.ANCHOR
    assert config.model.decoder.anchor_context is AnchorContext.TRANSFORMER
    assert config.datamodule.sample_limit == 10000
    assert config.trainer.max_steps == 2000


def test_train_config_rejects_anchor_without_first_codebook_adapter() -> None:
    with pytest.raises(ValueError, match="requires model.feature_adapter"):
        parse_train_config(_compose("model.decoder.fm_mode=anchor"))


def test_train_config_rejects_feature_adapter_on_rvq_route() -> None:
    raw = _compose(
        "model/route=rvq",
        "model.feature_adapter=longcat_first_codebook",
    )

    with pytest.raises(ValueError, match="feature_adapter requires model.route=fm"):
        parse_train_config(raw)


def test_train_config_rejects_longcat_adapter_on_other_backend() -> None:
    raw = _compose(
        "backend.name=other",
        "model.feature_adapter=longcat_first_codebook",
    )

    with pytest.raises(ValueError, match="requires backend=longcat"):
        parse_train_config(raw)


def test_legacy_bicodec_config_is_rejected_as_semantic_global() -> None:
    with pytest.raises(ValueError, match="semantic-global"):
        parse_train_config(
            _compose("experiment=overfit", "backend=bicodec", "model/route=rvq")
        )


def test_screening_uses_disjoint_training_partition() -> None:
    config = _compose("experiment=screening")

    assert config.datamodule.sample_limit == 984
    assert config.datamodule.max_seconds == 32.0
    assert config.datamodule.overlong == "filter"
    assert "lba" not in config.datamodule
    assert config.datamodule.batching.max_batch_seconds == 32.0
    assert config.trainer.use_distributed_sampler is False
    assert config.trainer.max_steps == -1
    assert config.trainer.max_epochs == 1
    assert config.callback.performance.profile_flops is False


def test_fm_ablation_experiments_compose_repa_and_ema_cells() -> None:
    baseline = _compose("experiment=ablation_fm_baseline")
    repa = _compose("experiment=ablation_fm_repa")
    ema = _compose("experiment=ablation_fm_ema")

    assert baseline.model.route == "fm"
    assert baseline.backend.name == "longcat"
    assert baseline.loss.repa_loss_weight == 0.0
    assert baseline.pl_module.ema_decay is None

    assert repa.loss.repa_loss_weight == 0.25
    assert repa.loss.repa_feature_dim == 768
    assert repa.pl_module.ema_decay is None
    assert repa.output_subdir == "ablation/longcat-fm/repa"

    assert ema.pl_module.ema_decay == 0.999
    assert ema.loss.repa_loss_weight == 0.0
    assert ema.output_subdir == "ablation/longcat-fm/ema"
