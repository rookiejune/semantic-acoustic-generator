from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes

from scripts import eval_artifact, eval_artifact_set, eval_speech_manifest, smoke
from semantic_acoustic_generator.backend import BackendConfig
from semantic_acoustic_generator.config import DecoderConfig
from semantic_acoustic_generator.evaluation import (
    evaluate_artifact_sample,
    factor_accuracy,
    seeded_generator,
    write_pcm16_wav,
)
from semantic_acoustic_generator.runtime import GeneratorRuntime
from semantic_acoustic_generator.types import GeneratorBatch, PairMetadata


class EvalBackend:
    name = "eval-test"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    semantic_codebook = torch.arange(32, dtype=torch.float32).view(8, 4)
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (8,)
    acoustic_feature_dim = 1

    def __init__(self, layout: AcousticLayout) -> None:
        if layout is not AcousticLayout.FRAME_ALIGNED:
            raise ValueError("evaluation backend must be frame-aligned.")
        self.acoustic_layout = layout
        self.acoustic_unit_length = None
        self.detokenized: list[SemanticAcousticCodes] = []
        self.feature_inputs: list[torch.Tensor] = []

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        semantic = torch.zeros(audio.size(0), 2, 1, dtype=torch.long, device=audio.device)
        acoustic = torch.zeros(
            audio.size(0), semantic.size(1), 1, dtype=torch.long, device=audio.device
        )
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    def detokenize(self, codes: SemanticAcousticCodes) -> torch.Tensor:
        self.detokenized.append(codes)
        value = codes.semantic.float().sum() + codes.acoustic.float().sum()
        return value.expand(codes.semantic.size(0), 1, codes.acoustic.size(1)).clone()

    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        self.feature_inputs.append(acoustic_codes.detach().clone())
        return acoustic_codes.float()

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        del semantic_codes
        return acoustic_features.transpose(1, 2).contiguous()


class EvalRuntime:
    sample_rate = EvalBackend.sample_rate

    def __init__(self, units: int, *, backend: EvalBackend | None = None) -> None:
        self.units = units
        self.backend = backend
        self.calls: list[dict[str, Any]] = []
        self.generated: list[torch.Tensor] = []

    def sample_features(
        self,
        semantic_codes: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        reference_features: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if generator is None:
            raise AssertionError("evaluation must provide a seeded generator")
        self.calls.append(
            {
                "generator": generator,
                "generator_state": generator.get_state().clone(),
                "mask": mask,
                "reference_features": reference_features,
                "reference_mask": reference_mask,
                "cfg_scale": cfg_scale,
            }
        )
        noise = torch.rand(
            semantic_codes.size(0),
            self.units,
            1,
            generator=generator,
            device=semantic_codes.device,
        )
        output = noise if reference_features is None else noise + 1
        self.generated.append(output)
        return output

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        features: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del semantic_codes, mask
        return features.transpose(1, 2).contiguous()


def test_seeded_generator_rejects_invalid_device() -> None:
    with pytest.raises(RuntimeError, match="device type"):
        seeded_generator("invalid", 0)


def test_speech_manifest_falls_back_to_pcm16_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample.wav"
    expected = torch.tensor([[[-1.0, -0.25, 0.0, 0.5, 1.0]]])
    write_pcm16_wav(path, expected, sample_rate=16_000)

    def fail_torchcodec(*args: object, **kwargs: object) -> tuple[torch.Tensor, int]:
        del args, kwargs
        raise RuntimeError("libtorchcodec unavailable")

    monkeypatch.setattr(eval_speech_manifest.torchaudio, "load", fail_torchcodec)
    audio, sample_rate = eval_speech_manifest._load_audio(path)

    assert sample_rate == 16_000
    torch.testing.assert_close(audio, expected[0], atol=1 / 32_768, rtol=0)


def test_factor_accuracy_names_retargeted_later_codebooks() -> None:
    predicted = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
    labels = torch.tensor([[[1, 0], [3, 4], [0, 6]]])
    valid = torch.tensor([[True, True, False]])

    metrics = factor_accuracy(
        predicted,
        labels,
        valid,
        prefix="retargeted_",
        codebook_offset=1,
    )

    assert metrics == {
        "retargeted_codebook_1_factor_a_accuracy": 1.0,
        "retargeted_codebook_1_factor_b_accuracy": 0.5,
    }


def test_artifact_sample_evaluation_owns_generic_domain_logic() -> None:
    backend = EvalBackend(AcousticLayout.FRAME_ALIGNED)
    runtime = EvalRuntime(units=2, backend=backend)
    batch = _pair(AcousticLayout.FRAME_ALIGNED)

    result = evaluate_artifact_sample(
        cast(GeneratorRuntime, cast(object, runtime)),
        batch,
        seed=13,
    )

    assert set(result.audio) == {
        "target_reconstruction",
        "generated_without_reference_raw",
    }
    assert set(result.metrics) == {"raw_feature_mse"}
    assert len(runtime.calls) == 1


def test_artifact_set_args_expose_evaluation_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_artifact_set.py",
            "--artifact",
            str(tmp_path / "artifact"),
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--role",
            "source",
            "--speaker-id",
            "speaker",
        ],
    )

    args = eval_artifact_set._args()

    assert args.role == "source"
    assert args.speaker_id == "speaker"


def test_eval_artifact_generates_seeded_pair_metrics() -> None:
    backend = EvalBackend(AcousticLayout.FRAME_ALIGNED)
    batch = _pair(AcousticLayout.FRAME_ALIGNED)
    runtime = EvalRuntime(units=batch.acoustic_codes.size(1))

    audio, metrics = eval_artifact._evaluate(
        cast(GeneratorRuntime, cast(object, runtime)),
        backend,
        batch,
        device=torch.device("cpu"),
        seed=13,
        cfg_scale=2.5,
    )

    assert len(runtime.calls) == 2
    assert runtime.calls[0]["generator"] is not runtime.calls[1]["generator"]
    assert torch.equal(runtime.calls[0]["generator_state"], runtime.calls[1]["generator_state"])
    assert runtime.calls[0]["reference_features"] is None
    assert runtime.calls[0]["reference_mask"] is None
    assert runtime.calls[0]["cfg_scale"] is None
    reference_acoustic = batch.reference_acoustic_codes
    reference_acoustic_mask = batch.reference_acoustic_mask
    assert reference_acoustic is not None
    assert reference_acoustic_mask is not None
    torch.testing.assert_close(
        cast(torch.Tensor, runtime.calls[1]["reference_features"]),
        reference_acoustic.float(),
    )
    assert torch.equal(
        cast(torch.Tensor, runtime.calls[1]["reference_mask"]),
        reference_acoustic_mask,
    )
    assert runtime.calls[1]["cfg_scale"] == 2.5
    target = batch.acoustic_codes.float()
    expected_without = float((runtime.generated[0] - target).pow(2).mean())
    expected_with = float((runtime.generated[1] - target).pow(2).mean())
    assert metrics == pytest.approx(
        {
            "feature_mse_without_reference": expected_without,
            "feature_mse_with_reference": expected_with,
            "reference_gain": expected_without - expected_with,
        }
    )
    assert set(audio) == {
        "generated_without_reference",
        "generated_with_reference",
        "target_reconstruction",
        "reference_reconstruction",
    }
    assert len(backend.detokenized) == 2


def test_eval_artifact_main_loads_cross_text_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = EvalBackend(AcousticLayout.FRAME_ALIGNED)
    batch = _pair(AcousticLayout.FRAME_ALIGNED)
    runtime = EvalRuntime(units=batch.acoustic_codes.size(1))
    runtime.backend = backend
    loaded: dict[str, Any] = {}
    args = argparse.Namespace(
        artifact=tmp_path / "artifact",
        codec="longcat",
        data_pairing="cross_text",
        data_root=None,
        split="train",
        sample_index=2,
        max_seconds=None,
        overlong="error",
        device="cpu",
        seed=5,
        cfg_scale=3.0,
        output_json=None,
        include_private_metadata=False,
        without_reference_wav=None,
        with_reference_wav=None,
        target_reconstruction_wav=None,
        reference_reconstruction_wav=None,
    )

    def load_pair(data, **kwargs):
        loaded["data"] = data
        loaded["kwargs"] = kwargs
        return batch

    def load_eval_backend(config, **kwargs):
        loaded["backend_config"] = config
        loaded["backend_kwargs"] = kwargs
        return backend

    monkeypatch.setattr(eval_artifact, "_args", lambda: args)
    monkeypatch.setattr(eval_artifact, "load_backend", load_eval_backend)
    monkeypatch.setattr(eval_artifact, "load_artifact", lambda *args, **kwargs: object())
    monkeypatch.setattr(eval_artifact, "load_batch", load_pair)
    monkeypatch.setattr(eval_artifact, "GeneratorRuntime", lambda *args: runtime)

    eval_artifact.main()

    assert loaded["backend_config"].name == "longcat"
    assert loaded["backend_config"].model_dir is None
    assert loaded["backend_config"].revision is None
    assert loaded["backend_config"].local_files_only is True
    assert loaded["backend_config"].allow_unpinned_revision is False
    assert loaded["backend_kwargs"]["device"] == torch.device("cpu")
    assert loaded["data"].dataset == "qwen"
    assert loaded["data"].pairing == "cross_text"
    assert loaded["data"].sample_index == 2
    assert loaded["kwargs"]["codec"] == "longcat"
    assert loaded["kwargs"]["acoustic_layout"] is AcousticLayout.FRAME_ALIGNED
    result = json.loads(capsys.readouterr().out)
    assert result["artifact"] == "artifact"
    assert result["data_root"] is None
    assert result["dataset"] == "qwen"
    assert result["pairing"] == "cross_text"
    assert result["cfg_scale"] == 3.0
    assert result["pair"]["target_index"] == 0
    assert "target_utterance_id" not in result["pair"]
    assert "reference_utterance_id" not in result["pair"]
    assert "target_speaker_id" not in result["pair"]
    assert "reference_speaker_id" not in result["pair"]
    assert "target_text" not in result["pair"]
    assert "reference_text" not in result["pair"]
    assert result["generated_without_reference"]["finite"] is True
    assert result["generated_with_reference"]["finite"] is True
    assert result["reference_gain"] == pytest.approx(
        result["feature_mse_without_reference"] - result["feature_mse_with_reference"]
    )


def test_eval_artifact_can_include_private_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = EvalBackend(AcousticLayout.FRAME_ALIGNED)
    batch = _pair(AcousticLayout.FRAME_ALIGNED)
    runtime = EvalRuntime(units=batch.acoustic_codes.size(1))
    runtime.backend = backend
    args = argparse.Namespace(
        artifact=tmp_path / "artifact",
        codec="longcat",
        data_pairing="cross_text",
        data_root=tmp_path / "data",
        split="train",
        sample_index=2,
        max_seconds=None,
        overlong="error",
        device="cpu",
        seed=5,
        cfg_scale=1.0,
        output_json=None,
        include_private_metadata=True,
        without_reference_wav=None,
        with_reference_wav=None,
        target_reconstruction_wav=None,
        reference_reconstruction_wav=None,
    )

    def load_eval_backend(config, **kwargs):
        assert config == BackendConfig(name="longcat")
        assert kwargs["device"] == torch.device("cpu")
        return backend

    monkeypatch.setattr(eval_artifact, "_args", lambda: args)
    monkeypatch.setattr(eval_artifact, "load_backend", load_eval_backend)
    monkeypatch.setattr(eval_artifact, "load_artifact", lambda *args, **kwargs: object())
    monkeypatch.setattr(eval_artifact, "load_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(eval_artifact, "GeneratorRuntime", lambda *args: runtime)

    eval_artifact.main()

    result = json.loads(capsys.readouterr().out)
    assert result["artifact"] == str(tmp_path / "artifact")
    assert result["data_root"] == str(tmp_path / "data")
    assert result["pair"]["target_utterance_id"] == "target"
    assert result["pair"]["reference_utterance_id"] == "reference"
    assert result["pair"]["target_speaker_id"] == "speaker"
    assert result["pair"]["reference_speaker_id"] == "speaker"
    assert result["pair"]["target_text"] == "target text"
    assert result["pair"]["reference_text"] == "reference text"


def test_smoke_uses_independent_fake_reference_and_seeded_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = smoke._batch()
    reference = smoke._reference_batch()
    assert len(target) == len(reference)
    for target_value, reference_value in zip(target, reference):
        assert target_value.data_ptr() != reference_value.data_ptr()
        assert not torch.equal(target_value, reference_value)

    states: list[tuple[int, torch.Generator, torch.Tensor]] = []
    decode_calls: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
    original_generator = smoke._generator
    original_decode = GeneratorRuntime.decode

    def generator(seed: int) -> torch.Generator:
        value = original_generator(seed)
        states.append((seed, value, value.get_state().clone()))
        return value

    def decode(
        runtime: GeneratorRuntime,
        semantic_codes: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        reference_features: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        decode_calls.append((reference_features, reference_mask))
        return original_decode(
            runtime,
            semantic_codes,
            mask=mask,
            reference_features=reference_features,
            reference_mask=reference_mask,
            generator=generator,
        )

    monkeypatch.setattr(smoke, "_generator", generator)
    monkeypatch.setattr(GeneratorRuntime, "decode", decode)
    smoke._artifact_smoke(
        smoke.FakeCodec(),
        DecoderConfig(hidden_dim=12, layers=1, heads=2, ffn_ratio=2),
    )

    assert [item[0] for item in states] == [0, 0, 1, 1, 0, 0]
    for left, right in ((0, 1), (2, 3), (4, 5)):
        assert states[left][1] is not states[right][1]
        assert torch.equal(states[left][2], states[right][2])
    assert len(decode_calls) == 2
    assert decode_calls[0] == (None, None)
    assert decode_calls[1][0] is not None
    assert decode_calls[1][1] is not None


def test_smoke_real_data_defaults_to_cross_text_and_loads_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["smoke.py"])
    assert smoke._args().data_pairing == "cross_text"

    backend = EvalBackend(AcousticLayout.FRAME_ALIGNED)
    batch = _pair(AcousticLayout.FRAME_ALIGNED)
    loaded: dict[str, Any] = {}

    def load_pair(data, **kwargs):
        loaded["data"] = data
        loaded["kwargs"] = kwargs
        return batch

    monkeypatch.setattr(smoke, "load_semantic_acoustic", lambda *args, **kwargs: backend)
    monkeypatch.setattr(smoke, "load_batch", load_pair)
    smoke._data_smoke(
        None,
        codec="longcat",
        pairing="cross_text",
        split="train",
        index=0,
        device="cpu",
    )

    assert loaded["data"].dataset == "qwen"
    assert loaded["data"].pairing == "cross_text"
    assert loaded["kwargs"]["acoustic_layout"] is AcousticLayout.FRAME_ALIGNED
    assert len(backend.feature_inputs) == 2
    reference_acoustic = batch.reference_acoustic_codes
    assert reference_acoustic is not None
    assert torch.equal(backend.feature_inputs[0], batch.acoustic_codes)
    assert torch.equal(backend.feature_inputs[1], reference_acoustic)


def _pair(layout: AcousticLayout) -> GeneratorBatch:
    target_semantic = torch.tensor([[[1], [2]]], dtype=torch.long)
    reference_semantic = torch.tensor([[[3], [4], [5]]], dtype=torch.long)
    if layout is AcousticLayout.FRAME_ALIGNED:
        target_acoustic = torch.tensor([[[2], [2]]], dtype=torch.long)
        reference_acoustic = torch.tensor([[[4], [4], [4]]], dtype=torch.long)
    else:
        target_acoustic = torch.tensor([[[2], [2], [2]]], dtype=torch.long)
        reference_acoustic = torch.tensor([[[4], [4], [4]]], dtype=torch.long)
    target_mask = torch.ones(target_semantic.shape[:2], dtype=torch.bool)
    target_acoustic_mask = torch.ones(target_acoustic.shape[:2], dtype=torch.bool)
    reference_mask = torch.ones(reference_semantic.shape[:2], dtype=torch.bool)
    reference_acoustic_mask = torch.ones(reference_acoustic.shape[:2], dtype=torch.bool)
    metadata = PairMetadata(
        target_index=0,
        reference_index=1,
        target_text_index=0,
        reference_text_index=1,
        target_source_index=10,
        reference_source_index=11,
        target_role="default",
        reference_role="default",
        target_utterance_id="target",
        reference_utterance_id="reference",
        target_speaker_id="speaker",
        reference_speaker_id="speaker",
        target_text="target text",
        reference_text="reference text",
    )
    return GeneratorBatch(
        semantic_codes=target_semantic,
        acoustic_codes=target_acoustic,
        mask=target_mask,
        semantic_pad_id=8,
        acoustic_pad_ids=(8,),
        acoustic_mask=target_acoustic_mask,
        acoustic_layout=layout,
        reference_semantic_codes=reference_semantic,
        reference_acoustic_codes=reference_acoustic,
        reference_mask=reference_mask,
        reference_acoustic_mask=reference_acoustic_mask,
        metadata=(metadata,),
    )
