from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="train", overrides=list(overrides))


@pytest.mark.parametrize(
    ("experiment", "backend", "route", "max_steps"),
    [
        ("001_longcat_fm", "longcat", "fm", 1_000_000),
        ("001_longcat_rvq", "longcat", "rvq", 1_000_000),
        ("001_bicodec_fm", "bicodec", "fm", 1_000_000),
        ("001_bicodec_rvq", "bicodec", "rvq", 1_000_000),
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
    assert config.callback.checkpoint.enabled is True
    assert config.callback.performance.profile_flops is False
    assert config.callback.data_throughput.enabled is True


def test_experiment_composition_can_be_overridden_explicitly() -> None:
    config = _compose("experiment=overfit", "backend=bicodec", "model/route=rvq")

    assert config.backend.name == "bicodec"
    assert config.model.route == "rvq"
    assert config.datamodule.fixed_batch is True
    assert config.output_subdir == "overfit/bicodec/rvq-8l"


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
