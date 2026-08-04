from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from anydataset.dataset import MapStyleABC
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from lightning import pytorch as pl
from torch import Tensor

pytest.importorskip("lightning")

try:
    import semantic_acoustic_generator.training.service as training_service
    from semantic_acoustic_generator.callback import (
        CodebookUsageLogger,
        SampleLogConfig,
        SampleLogger,
    )
    from semantic_acoustic_generator.config import DecoderConfig, Route
    from semantic_acoustic_generator.datamodule import collate_codes, single_batch_loader
    from semantic_acoustic_generator.pl_module import (
        CHECKPOINT_METADATA_KEY,
        CHECKPOINT_SCHEMA_VERSION,
        build_module,
        dataset_feature_stats,
        feature_stats,
    )
    from semantic_acoustic_generator.pl_module.semantic import (
        LEGACY_CHECKPOINT_METADATA_KEY,
        LEGACY_CHECKPOINT_SCHEMA_VERSION,
    )
    from semantic_acoustic_generator.runtime import (
        GeneratorConfig,
        GeneratorRuntime,
        SamplingConfig,
    )
    from semantic_acoustic_generator.runtime.artifact import load_artifact
    from semantic_acoustic_generator.training import build_callbacks, parse_train_config
    from semantic_acoustic_generator.types import GeneratorBatch, PairMetadata
except TypeError as exc:
    if "SPEAKER_ID" not in str(exc):
        raise
    pytest.skip(
        "anydataset TextMeta currently defines SPEAKER_ID twice; training tests require that third_party fix.",
        allow_module_level=True,
    )


class FakeCodec:
    name = "fake"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None
    semantic_codebook = torch.randn(8, 6)
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (5, 7)
    acoustic_feature_dim = 4

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del sample_rate
        return audio.new_zeros((audio.size(0), 2, 3), dtype=torch.long)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.new_zeros((codes.size(0), 1, codes.size(1) * 320), dtype=torch.float32)

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> SemanticAcousticCodes:
        codes = self.encode(audio, sample_rate)
        return SemanticAcousticCodes(codes[..., :1], codes[..., 1:])

    def detokenize(self, codes: SemanticAcousticCodes) -> torch.Tensor:
        return self.decode(torch.cat((codes.semantic, codes.acoustic), dim=-1))

    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(acoustic_codes.float(), (0, 2))[:, :, :4]

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        return acoustic_features.new_zeros(
            (semantic_codes.size(0), 1, semantic_codes.size(1) * 320)
        )


class FakeModuleCodec(torch.nn.Module):
    name = "fake_module"
    sample_rate = FakeCodec.sample_rate
    frame_rate = FakeCodec.frame_rate
    semantic_frame_rate = FakeCodec.semantic_frame_rate
    acoustic_layout = FakeCodec.acoustic_layout
    acoustic_unit_length = FakeCodec.acoustic_unit_length
    semantic_codebook_sizes = FakeCodec.semantic_codebook_sizes
    acoustic_codebook_sizes = FakeCodec.acoustic_codebook_sizes
    acoustic_feature_dim = FakeCodec.acoustic_feature_dim

    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Linear(1, 1)
        self.semantic_codebook = torch.randn(8, 6)

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return FakeCodec().encode(audio, sample_rate)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return FakeCodec().decode(codes)

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> SemanticAcousticCodes:
        return FakeCodec().tokenize(audio, sample_rate)

    def detokenize(self, codes: SemanticAcousticCodes) -> torch.Tensor:
        return FakeCodec().detokenize(codes)

    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        return FakeCodec().acoustic_codes_to_features(acoustic_codes)

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        return FakeCodec().decode_features(semantic_codes, acoustic_features)


class FakeRepaTeacher:
    feature_dim = 3

    def __call__(
        self,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if (
            semantic_codes.shape[:2] != acoustic_codes.shape[:2]
            or mask.shape != semantic_codes.shape[:2]
        ):
            raise ValueError("fake teacher inputs must align")
        return torch.ones(mask.shape + (self.feature_dim,), device=semantic_codes.device)


def _paired_batch(backend: FakeCodec) -> GeneratorBatch:
    target = collate_codes(
        [
            torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long),
            torch.tensor([[5, 3, 4]], dtype=torch.long),
        ],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    reference = collate_codes(
        [
            torch.tensor([[6, 4, 6]], dtype=torch.long),
            torch.tensor([[2, 0, 1], [3, 2, 5]], dtype=torch.long),
        ],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    metadata = (
        PairMetadata(
            target_index=0,
            reference_index=2,
            target_text_index=0,
            reference_text_index=2,
            target_source_index=0,
            reference_source_index=2,
            target_role="target",
            reference_role="target",
            target_utterance_id="target-0",
            reference_utterance_id="reference-0",
            target_speaker_id="vivian",
            reference_speaker_id="vivian",
            target_text="target text zero",
            reference_text="reference text zero",
        ),
        PairMetadata(
            target_index=1,
            reference_index=3,
            target_text_index=1,
            reference_text_index=3,
            target_source_index=1,
            reference_source_index=3,
            target_role="target",
            reference_role="target",
            target_utterance_id="target-1",
            reference_utterance_id="reference-1",
            target_speaker_id="vivian",
            reference_speaker_id="vivian",
            target_text="target text one",
            reference_text="reference text one",
        ),
    )
    return GeneratorBatch(
        semantic_codes=target.semantic_codes,
        acoustic_codes=target.acoustic_codes,
        mask=target.mask,
        semantic_pad_id=target.semantic_pad_id,
        acoustic_pad_ids=target.acoustic_pad_ids,
        acoustic_mask=target.acoustic_mask,
        acoustic_layout=target.acoustic_layout,
        reference_semantic_codes=reference.semantic_codes,
        reference_acoustic_codes=reference.acoustic_codes,
        reference_mask=reference.mask,
        reference_acoustic_mask=reference.acoustic_mask,
        metadata=metadata,
    )


def test_lightning_runs_anydataset_dynamic_loader_across_epochs() -> None:
    class Dataset(MapStyleABC):
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> Tensor:
            return torch.tensor(float(index + 1))

        def _shuffle(
            self,
            *,
            shuffle: bool,
            seed: int,
            epoch: int,
            num_replicas: int,
            rank: int,
        ):
            self.epochs.append(epoch)
            yield from super()._shuffle(
                shuffle=shuffle,
                seed=seed,
                epoch=epoch,
                num_replicas=num_replicas,
                rank=rank,
            )

    class Module(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch: Tensor, batch_index: int) -> Tensor:
            del batch_index
            return (self.weight * batch).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    dataset = Dataset()
    loader = dataset.dataloader(
        costs=[1] * len(dataset),
        max_batch_memory=2,
        max_batch_samples=2,
        shuffle=True,
        epoch=0,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        use_distributed_sampler=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(Module(), train_dataloaders=loader)

    assert trainer.train_dataloader is loader
    assert type(loader.batch_sampler.sampler).__module__ == "anydataset.dataset.batching"
    assert dataset.epochs == [0, 1]
    assert trainer.current_epoch == 2


def test_training_module_trains_and_exports_artifact(tmp_path) -> None:
    backend = FakeCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)

    output = module.training_step(batch, 0)
    loss = output["loss"]
    loss.backward()
    module.export_artifact(tmp_path)
    loaded = load_artifact(tmp_path)

    assert torch.isfinite(loss)
    assert module.config.feature_mean is not None
    assert module.support.reference_conditioner.null_condition.grad is not None
    assert loaded.feature_mean.shape == (1, 1, backend.acoustic_feature_dim)
    assert loaded.sample_features(batch.semantic_codes, mask=batch.mask).shape == (
        1,
        2,
        backend.acoustic_feature_dim,
    )


def test_training_module_rejects_non_finite_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)
    loss = module.support.generator.loss

    def non_finite_loss(*args: Any, **kwargs: Any):
        output = loss(*args, **kwargs)
        return replace(output, loss=output.loss.new_full((), float("nan")))

    monkeypatch.setattr(module.support.generator, "loss", non_finite_loss)

    with pytest.raises(FloatingPointError, match="flow training loss is non-finite"):
        module.training_step(batch, 0)


def test_deferred_finite_loss_guard_is_sticky_and_checkpoint_safe() -> None:
    backend = FakeCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)
    module._defer_finite_training_loss(torch.tensor(True), name="flow")
    module._defer_finite_training_loss(torch.tensor(False), name="flow")
    module._defer_finite_training_loss(torch.tensor(True), name="flow")

    with pytest.raises(FloatingPointError, match="flow training loss is non-finite"):
        module.on_save_checkpoint({"state_dict": {}})


def test_deferred_finite_loss_guard_resets_after_successful_flush() -> None:
    backend = FakeCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)
    module._defer_finite_training_loss(torch.tensor(True), name="flow")

    module.on_train_end()

    assert module._finite_training_loss_ok is None
    assert module._finite_training_loss_name is None


def test_training_resume_uses_configured_learning_rate(tmp_path: Path) -> None:
    backend = FakeCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    original = build_module(
        backend,
        config,
        batch,
        normalize_features=True,
        learning_rate=3e-4,
    )
    first = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    first.fit(original, train_dataloaders=single_batch_loader(batch))
    checkpoint_path = tmp_path / "resume.ckpt"
    first.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(3e-4)

    observed: list[float] = []

    class CaptureLearningRate(pl.Callback):
        def on_train_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
        ) -> None:
            del pl_module
            observed.append(float(trainer.optimizers[0].param_groups[0]["lr"]))

    resumed = build_module(
        backend,
        config,
        batch,
        normalize_features=True,
        learning_rate=1e-4,
    )
    second = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[CaptureLearningRate()],
    )
    second.fit(
        resumed,
        train_dataloaders=single_batch_loader(batch),
        ckpt_path=checkpoint_path,
    )

    assert second.global_step == 2
    assert observed == pytest.approx([1e-4])
    assert second.optimizers[0].param_groups[0]["lr"] == pytest.approx(1e-4)


def test_training_checkpoint_drops_backend_and_requires_support_state() -> None:
    backend = FakeModuleCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)
    checkpoint = {"state_dict": dict(module.state_dict())}

    assert not any(key.startswith("backend.") for key in checkpoint["state_dict"])
    checkpoint["state_dict"]["backend.probe.weight"] = backend.probe.weight.detach().clone()
    checkpoint[LEGACY_CHECKPOINT_METADATA_KEY] = {
        "schema_version": LEGACY_CHECKPOINT_SCHEMA_VERSION,
        "backend_state": "external",
    }
    module.on_save_checkpoint(checkpoint)

    assert LEGACY_CHECKPOINT_METADATA_KEY not in checkpoint
    assert checkpoint[CHECKPOINT_METADATA_KEY] == {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "backend_state": "external",
    }
    assert not any(key.startswith("backend.") for key in checkpoint["state_dict"])
    module.on_load_checkpoint(checkpoint)

    broken = {
        CHECKPOINT_METADATA_KEY: checkpoint[CHECKPOINT_METADATA_KEY],
        "state_dict": dict(checkpoint["state_dict"]),
    }
    support_key = next(key for key in broken["state_dict"] if key.startswith("support."))
    del broken["state_dict"][support_key]

    with pytest.raises(RuntimeError, match="missing support state"):
        module.on_load_checkpoint(broken)


def test_training_checkpoint_loads_legacy_metadata_key() -> None:
    backend = FakeModuleCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)
    checkpoint = {
        LEGACY_CHECKPOINT_METADATA_KEY: {
            "schema_version": LEGACY_CHECKPOINT_SCHEMA_VERSION,
            "backend_state": "external",
        },
        "state_dict": dict(module.state_dict()),
    }

    module.on_load_checkpoint(checkpoint)


def test_paired_reference_dropout_is_sampled_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeCodec()
    batch = _paired_batch(backend)
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(
        backend,
        config,
        batch,
        normalize_features=True,
        reference_dropout=0.5,
    )
    convert = backend.acoustic_codes_to_features
    converted_batches: list[int] = []

    def acoustic_codes_to_features(codes: Tensor) -> Tensor:
        converted_batches.append(codes.size(0))
        return convert(codes)

    monkeypatch.setattr(backend, "acoustic_codes_to_features", acoustic_codes_to_features)
    original_rand = torch.rand
    sampled = False

    def rand(*size: int, **kwargs: Any) -> Tensor:
        nonlocal sampled
        if not sampled:
            sampled = True
            assert size == (2,)
            return torch.tensor([0.1, 0.9], device=kwargs.get("device"))
        return original_rand(*size, **kwargs)

    captured: dict[str, Any] = {}

    def capture(
        _conditioner: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        captured["features"] = args[0]
        captured["mask"] = kwargs["mask"]
        captured["use_reference"] = kwargs["use_reference"]
        captured["row_indices"] = kwargs["row_indices"]

    logs: dict[str, Any] = {}

    def log_dict(metrics: dict[str, Any], **_: Any) -> None:
        logs.update(metrics)

    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(module, "log_dict", log_dict)
    handle = module.support.reference_conditioner.register_forward_pre_hook(
        capture,
        with_kwargs=True,
    )
    output = module.training_step(batch, 0)
    handle.remove()
    output["loss"].backward()

    features = cast(Tensor, captured["features"])
    reference_codes = batch.reference_acoustic_codes
    reference_mask = batch.reference_acoustic_mask
    assert reference_codes is not None
    assert reference_mask is not None
    expected_codes = reference_codes[1:].masked_fill(~reference_mask[1:, ..., None], 0)
    expected = convert(expected_codes).masked_fill(~reference_mask[1:, ..., None], 0)
    null = module.support.reference_conditioner.null_condition

    assert sampled
    assert converted_batches == [2, 1]
    torch.testing.assert_close(features, expected)
    assert torch.equal(cast(Tensor, captured["mask"]), reference_mask[1:])
    assert captured["use_reference"] is None
    assert cast(Tensor, captured["row_indices"]).tolist() == [1]
    assert float(cast(Tensor, logs["train/reference_fraction"])) == pytest.approx(0.5)
    assert null.grad is not None


def test_sample_logger_uses_fixed_pair_and_writes_paired_metrics(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeCodec()
    batch = _paired_batch(backend)
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)
    experiment = SimpleNamespace(audio=[], scalars=[])

    def add_audio(tag, value, *, global_step, sample_rate):
        experiment.audio.append((tag, value, global_step, sample_rate))

    def add_scalar(tag, value, *, global_step):
        experiment.scalars.append((tag, value, global_step))

    experiment.add_audio = add_audio
    experiment.add_scalar = add_scalar
    trainer = SimpleNamespace(
        is_global_zero=True,
        global_step=2,
        loggers=[SimpleNamespace(experiment=experiment)],
    )
    generators: list[torch.Generator] = []
    generator_states: list[Tensor] = []
    sample_features = GeneratorRuntime.sample_features

    def capture_generators(
        runtime: GeneratorRuntime,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        cfg_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        assert generator is not None
        generators.append(generator)
        generator_states.append(generator.get_state().clone())
        return sample_features(
            runtime,
            semantic_codes,
            mask=mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            cfg_scale=cfg_scale,
            generator=generator,
        )

    monkeypatch.setattr(GeneratorRuntime, "sample_features", capture_generators)
    callback = SampleLogger(
        tmp_path,
        batch,
        SampleLogConfig(every_n_train_steps=2, seed=7),
    )

    callback.on_train_batch_end(cast(Any, trainer), module, None, batch, 0)

    events = json.loads((tmp_path / "sample_metrics.json").read_text(encoding="utf-8"))
    assert len(events) == 1
    event = events[0]
    assert event["step"] == 2
    assert event["metadata"]["target_index"] == 0
    assert "target_utterance_id" not in event["metadata"]
    assert "reference_utterance_id" not in event["metadata"]
    assert "target_speaker_id" not in event["metadata"]
    assert "reference_speaker_id" not in event["metadata"]
    assert "target_text" not in event["metadata"]
    assert "reference_text" not in event["metadata"]
    assert event["generated_with_reference"]["finite"] is True
    assert event["target_codec_reconstruction"]["finite"] is True
    assert event["reference_codec_reconstruction"]["finite"] is True
    assert event["reference_gain"] == pytest.approx(
        event["feature_mse_without_reference"] - event["feature_mse_with_reference"]
    )
    assert event["feature_mse_without_reference"] == event["feature_mse_with_reference"]
    assert len(generators) == 2
    assert generators[0] is not generators[1]
    assert torch.equal(generator_states[0], generator_states[1])
    assert {item[0] for item in experiment.audio} == {
        "sample/generated_without_reference",
        "sample/generated_with_reference",
        "sample/target_codec_reconstruction",
        "sample/reference_codec_reconstruction",
    }
    assert {item[0] for item in experiment.scalars} == {
        "sample/feature_mse_without_reference",
        "sample/feature_mse_with_reference",
        "sample/reference_gain",
    }
    assert all(item[2] == 2 and item[3] == backend.sample_rate for item in experiment.audio)
    state = callback.state_dict()
    assert state == {"last_logged_step": 2, "logged_static_audio": True}

    restored = SampleLogger(
        tmp_path,
        batch,
        SampleLogConfig(every_n_train_steps=2, seed=7),
    )
    restored.load_state_dict(state)
    trainer.global_step = 4
    replacement = collate_codes(
        [torch.tensor([[7, 1, 1]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    restored.on_train_batch_end(cast(Any, trainer), module, None, replacement, 9)

    events = json.loads((tmp_path / "sample_metrics.json").read_text(encoding="utf-8"))
    assert len(events) == 2
    assert events[1]["metadata"] == event["metadata"]
    assert events[1]["target_codec_reconstruction"] is None
    assert events[1]["reference_codec_reconstruction"] is None
    assert events[1]["feature_mse_without_reference"] == pytest.approx(
        event["feature_mse_without_reference"]
    )
    assert events[1]["feature_mse_with_reference"] == pytest.approx(
        event["feature_mse_with_reference"]
    )
    assert {item[0] for item in experiment.audio if item[2] == 4} == {
        "sample/generated_without_reference",
        "sample/generated_with_reference",
    }
    assert len(generators) == 4
    assert generators[2] is not generators[3]
    assert all(torch.equal(state, generator_states[0]) for state in generator_states[1:])


def test_training_session_uses_datamodule_when_fixed_batch_is_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    omegaconf = pytest.importorskip("omegaconf")

    fits: list[dict[str, Any]] = []
    built: list[dict[str, Any]] = []
    fixed_batch = collate_codes(
        [torch.tensor([[1, 1, 2], [2, 3, 4]], dtype=torch.long)],
        semantic_pad_id=FakeCodec.semantic_codebook.size(0),
        acoustic_pad_ids=FakeCodec.acoustic_codebook_sizes,
    )

    class DataModuleStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            self.setup_stages: list[str] = []

        def setup(self, stage: str) -> None:
            self.setup_stages.append(stage)

        def sample_batch(self) -> GeneratorBatch:
            return fixed_batch

        def feature_stats_dataloader(self) -> object:
            raise AssertionError("resume must reuse feature stats from the checkpoint")

    trainer_configs: list[dict[str, Any]] = []

    class TrainerStub:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            trainer_configs.append(kwargs)

        def fit(self, module: object, **kwargs: Any) -> None:
            fits.append({"module": module, **kwargs})

    def fail_single_batch(*args: Any, **kwargs: Any) -> object:
        raise AssertionError("fixed_batch=false must not use single_batch_loader")

    def build(*args: Any, **kwargs: Any) -> object:
        del args
        built.append(kwargs)
        return object()

    checkpoint_path = tmp_path / "last.ckpt"
    torch.save(
        {
            "state_dict": {
                "support.feature_mean": torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
                "support.feature_std": torch.tensor([[[0.5, 1.0, 1.5, 2.0]]]),
            }
        },
        checkpoint_path,
    )

    monkeypatch.setattr(training_service.pl, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_service.pl, "Trainer", TrainerStub)
    monkeypatch.setattr(training_service, "load_backend", lambda *args, **kwargs: FakeCodec())
    monkeypatch.setattr(training_service, "load_batch", fail_single_batch)
    monkeypatch.setattr(training_service, "build_module", build)
    monkeypatch.setattr(training_service, "DataModule", DataModuleStub)
    monkeypatch.setattr(training_service, "single_batch_loader", fail_single_batch)

    config = omegaconf.OmegaConf.create(
        {
            "seed": 0,
            "output_dir": str(tmp_path),
            "output_subdir": "unused",
            "backend": {"name": "longcat"},
            "model": {
                "route": "fm",
                "condition_dim": 10,
                "decoder": {
                    "hidden_dim": None,
                    "layers": 1,
                    "heads": 2,
                    "ffn_ratio": 2,
                    "rvq_predictor": "mtp",
                    "mtp_layers": 1,
                    "mtp_heads": 2,
                },
            },
            "loss": {
                "repa_feature_dim": None,
                "repa_student_layer": None,
                "repa_loss_weight": 0.0,
            },
            "pl_module": {
                "normalize_features": True,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "reference_dropout": 0.5,
            },
            "runtime": {
                "device": "cpu",
                "initialization": "codec",
                "sampling": {"flow_steps": 2, "temperature": 1.0, "top_p": 1.0},
            },
            "datamodule": {
                "source": "qwen_fixed_speaker",
                "root": None,
                "split": "train",
                "sample_index": 0,
                "max_seconds": None,
                "overlong": "truncate",
                "sample_limit": 32,
                "batch_size": 8,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "fixed_batch": False,
                "batching": {
                    "enabled": False,
                    "max_batch_seconds": 8.0,
                    "planning_window": 256,
                    "prefetch_factor": 4,
                    "drop_distributed_tail": True,
                    "seed": 0,
                },
            },
            "callback": {
                "sample": {"enabled": False, "every_n_train_steps": 2, "seed": 0},
                "performance": {
                    "enabled": False,
                    "model_flops_per_step": None,
                    "hardware_peak_flops": None,
                    "log_every_n_steps": 1,
                    "warmup_steps": 0,
                    "measure_window_steps": 1,
                },
                "checkpoint": {
                    "enabled": False,
                    "filename": "step-{step:08d}",
                    "save_last": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 2,
                },
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": 1,
                "strategy": "auto",
                "use_distributed_sampler": False,
                "precision": "32-true",
                "max_steps": 4,
                "max_epochs": -1,
                "log_every_n_steps": 1,
                "gradient_clip_val": 0.0,
                "ckpt_path": str(checkpoint_path),
            },
        }
    )

    session = training_service.build_session(config)
    assert isinstance(session, training_service.TrainingSession)
    assert isinstance(session.data_module, DataModuleStub)
    session.fit()

    assert len(fits) == 1
    assert trainer_configs[0]["max_steps"] == 4
    assert isinstance(fits[0]["datamodule"], DataModuleStub)
    assert fits[0]["datamodule"].setup_stages == ["fit"]
    assert "train_dataloaders" not in fits[0]
    assert fits[0]["ckpt_path"] == str(checkpoint_path)
    assert built[0]["feature_mean"] == (1.0, 2.0, 3.0, 4.0)
    assert built[0]["feature_std"] == (0.5, 1.0, 1.5, 2.0)


def test_training_script_delegates_to_public_service() -> None:
    from scripts import train

    assert train.run is training_service.run


def test_training_callback_builder_respects_switches(tmp_path: Path) -> None:
    omegaconf = pytest.importorskip("omegaconf")

    config = omegaconf.OmegaConf.create(
        {
            "callback": {
                "sample": {"enabled": True, "every_n_train_steps": 2, "seed": 7},
                "performance": {
                    "enabled": False,
                    "log_every_n_steps": 5,
                    "warmup_steps": 1,
                    "measure_window_steps": 3,
                },
                "data_throughput": {"enabled": False},
                "codebook_usage": {"enabled": False},
                "loss_summary": {"enabled": False},
                "loss_time_bucket": {"enabled": False},
                "checkpoint": {
                    "enabled": True,
                    "filename": "step-{step:08d}",
                    "save_last": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 2,
                },
            },
            "pl_module": {"ema_decay": 0.999, "ema_update_after_step": 4},
            "trainer": {"log_every_n_steps": 11},
        }
    )

    callbacks = build_callbacks(
        parse_train_config(config),
        output_dir=tmp_path,
        fixed_batch=_paired_batch(FakeCodec()),
    )

    assert [type(callback).__name__ for callback in callbacks] == [
        "ArtifactExport",
        "LearningRateMonitor",
        "EMACallback",
        "SampleLogger",
        "ModelCheckpoint",
    ]


def test_codebook_usage_logger_logs_target_codebooks() -> None:
    class _LogModule:
        def __init__(self) -> None:
            self.logged: dict[str, Tensor] = {}
            self.log_kwargs: dict[str, Any] = {}

        def log_dict(self, values: dict[str, Tensor], **kwargs: Any) -> None:
            self.logged.update(values)
            self.log_kwargs = kwargs

    module = _LogModule()
    batch = _paired_batch(FakeCodec())
    callback = CodebookUsageLogger(every_n_steps=1)

    callback.on_train_batch_end(
        cast(Any, SimpleNamespace(global_step=0, world_size=1)),
        cast(Any, module),
        cast(Any, None),
        batch,
        0,
    )

    assert "codebook/semantic/book_0/usage/perplexity" in module.logged
    assert "codebook/acoustic/book_0/usage/perplexity" in module.logged
    assert "codebook/acoustic/book_1/active/ratio" in module.logged
    assert module.log_kwargs == {
        "on_step": True,
        "on_epoch": False,
        "logger": True,
        "sync_dist": False,
    }


def test_training_module_adds_repa_loss_with_teacher() -> None:
    backend = FakeCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(
            layers=2,
            heads=2,
            ffn_ratio=2,
            repa_feature_dim=FakeRepaTeacher.feature_dim,
            repa_student_layer=1,
            repa_loss_weight=0.25,
        ),
    )
    module = build_module(
        backend,
        config,
        batch,
        normalize_features=True,
        repa_teacher=FakeRepaTeacher(),
    )

    output = module.training_step(batch, 0)
    loss = output["loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert "flow" in output
    assert "repa" in output


def test_training_module_requires_repa_teacher() -> None:
    backend = FakeCodec()
    batch = collate_codes(
        [torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(
            layers=2,
            heads=2,
            ffn_ratio=2,
            repa_feature_dim=FakeRepaTeacher.feature_dim,
            repa_loss_weight=0.25,
        ),
    )

    with pytest.raises(ValueError, match="REPA requires a teacher"):
        build_module(backend, config, batch, normalize_features=True)


def test_feature_stats_use_only_valid_frames() -> None:
    backend = FakeCodec()
    batch = collate_codes(
        [
            torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long),
            torch.tensor([[5, 3, 4]], dtype=torch.long),
        ],
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )

    mean, std = feature_stats(backend, batch)

    assert len(mean) == backend.acoustic_feature_dim
    assert len(std) == backend.acoustic_feature_dim
    assert all(value > 0 for value in std)


def test_dataset_feature_stats_combine_multiple_batches() -> None:
    backend = FakeCodec()
    batches = [
        collate_codes(
            [torch.tensor([[1, 0, 0]], dtype=torch.long)],
            semantic_pad_id=backend.semantic_codebook.size(0),
            acoustic_pad_ids=backend.acoustic_codebook_sizes,
        ),
        collate_codes(
            [torch.tensor([[2, 4, 6]], dtype=torch.long)],
            semantic_pad_id=backend.semantic_codebook.size(0),
            acoustic_pad_ids=backend.acoustic_codebook_sizes,
        ),
    ]

    mean, std = dataset_feature_stats(backend, batches)

    assert mean == pytest.approx((2.0, 3.0, 0.0, 0.0))
    assert std == pytest.approx((2.0, 3.0, 1e-5, 1e-5))


def test_validation_metrics_are_paired_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeCodec()
    batch = _paired_batch(backend)
    module = build_module(
        backend,
        GeneratorConfig(
            route=Route.FM,
            condition_dim=10,
            decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
            sampling=SamplingConfig(flow_steps=1),
        ),
        batch,
        normalize_features=True,
        validation_seed=7,
    )
    states: list[Tensor] = []
    logged: list[str] = []
    original = module.support.sample_features

    def sample_features(*args: Any, generator: torch.Generator | None = None, **kwargs: Any):
        assert generator is not None
        states.append(generator.get_state().clone())
        return original(*args, generator=generator, **kwargs)

    monkeypatch.setattr(module.support, "sample_features", sample_features)
    monkeypatch.setattr(module, "log", lambda name, *_args, **_kwargs: logged.append(name))

    first = module.validation_step(batch, 3)
    second = module.validation_step(batch, 3)

    assert set(first) == {
        "val/without_reference_feature_mse",
        "val/with_reference_feature_mse",
        "val/reference_gain_feature_mse",
    }
    for name in first:
        torch.testing.assert_close(first[name], second[name])
    torch.testing.assert_close(
        first["val/reference_gain_feature_mse"],
        first["val/without_reference_feature_mse"] - first["val/with_reference_feature_mse"],
    )
    assert len(states) == 4
    assert all(torch.equal(state, states[0]) for state in states[1:])
    assert logged == list(first) + list(second)


def test_validation_epoch_metric_is_invariant_to_batch_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeCodec()
    values = [
        torch.tensor([[1, 1, 1]], dtype=torch.long),
        torch.tensor([[2, 3, 1], [3, 3, 1], [4, 3, 1]], dtype=torch.long),
    ]
    combined = collate_codes(
        values,
        semantic_pad_id=backend.semantic_codebook.size(0),
        acoustic_pad_ids=backend.acoustic_codebook_sizes,
    )
    split = [
        collate_codes(
            [value],
            semantic_pad_id=backend.semantic_codebook.size(0),
            acoustic_pad_ids=backend.acoustic_codebook_sizes,
        )
        for value in values
    ]

    def validation_error(
        batch: GeneratorBatch,
        *,
        reference_features: Tensor | None,
        reference_mask: Tensor | None,
        generator: torch.Generator,
    ) -> Tensor:
        del reference_features, reference_mask, generator
        error = batch.acoustic_codes[..., 0].float()
        mask = batch.acoustic_mask.to(device=error.device)
        return error[mask].mean()

    def collate_batch(batches: list[GeneratorBatch]) -> GeneratorBatch:
        assert len(batches) == 1
        return batches[0]

    def validate(batches: list[GeneratorBatch]) -> float:
        module = build_module(
            backend,
            GeneratorConfig(
                route=Route.FM,
                condition_dim=10,
                decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
                sampling=SamplingConfig(flow_steps=1),
            ),
            combined,
            normalize_features=True,
        )
        monkeypatch.setattr(module, "_validation_error", validation_error)
        loader = torch.utils.data.DataLoader(
            batches,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_batch,
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        result = trainer.validate(module, dataloaders=loader, verbose=False)
        return float(result[0]["val/without_reference_feature_mse"])

    combined_metric = validate([combined])
    split_metric = validate(split)

    assert combined_metric == pytest.approx(2.5)
    assert split_metric == pytest.approx(combined_metric)
