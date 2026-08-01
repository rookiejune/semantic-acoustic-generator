from __future__ import annotations

import json
from typing import Any, cast

import pytest
import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from torch import nn

import semantic_acoustic_codec.runtime.semantic as runtime_semantic
from semantic_acoustic_codec.config import DecoderConfig, Route, RVQPredictor
from semantic_acoustic_codec.model.condition import ReferenceConditioner, SemanticConditioner
from semantic_acoustic_codec.model.decoder import CodecUnitGenerator
from semantic_acoustic_codec.model.routes import RouteModules
from semantic_acoustic_codec.runtime import (
    SamplingConfig,
    SemanticCodecRuntime,
    SemanticCodecSupport,
    SemanticSupportConfig,
    build_support,
    save_artifact,
)


class FakeBackend:
    name = "runtime-reference-test"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None
    semantic_codebook = torch.arange(32, dtype=torch.float32).view(8, 4) / 8
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (5,)
    acoustic_feature_dim = 4

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        semantic = torch.zeros(audio.size(0), 2, 1, dtype=torch.long, device=audio.device)
        acoustic = torch.zeros_like(semantic)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    def detokenize(self, codes: SemanticAcousticCodes) -> torch.Tensor:
        return self.decode_features(codes.semantic, self.acoustic_codes_to_features(codes.acoustic))

    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        return acoustic_codes.float().expand(-1, -1, self.acoustic_feature_dim).contiguous()

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        del semantic_codes
        return acoustic_features.flatten(1).unsqueeze(1)


class SeededGenerator(nn.Module):
    def __init__(self, feature_dim: int, codebook_size: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size
        self.conditions: list[torch.Tensor] = []

    def sample_features(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        flow_steps: int,
        temperature: float,
        top_p: float,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del feature_mean, feature_std, flow_steps, temperature, top_p
        _validate_frame_aligned(condition, mask, acoustic_layout, output_length)
        self.conditions.append(condition.detach().clone())
        noise = torch.randn(
            (*condition.shape[:2], self.feature_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        return condition[..., : self.feature_dim] + noise

    def sample_acoustic_codes(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED,
        output_length: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del temperature, top_p
        _validate_frame_aligned(condition, mask, acoustic_layout, output_length)
        self.conditions.append(condition.detach().clone())
        codes = torch.randint(
            self.codebook_size,
            condition.shape[:2],
            device=condition.device,
            generator=generator,
        )
        return codes[..., None]


def test_support_and_runtime_accept_optional_reference() -> None:
    semantic = torch.tensor([[[1], [2], [3]]], dtype=torch.long)
    semantic_mask = torch.ones(1, 3, dtype=torch.bool)
    reference = torch.tensor(
        [[[1.0, 0.0, -1.0, 2.0], [0.0, 2.0, 1.0, -2.0]]],
    )
    reference_mask = torch.tensor([[True, True]])
    support, generator = _support(Route.FM)
    runtime = SemanticCodecRuntime(support, FakeBackend())

    support_without = support.sample_features(
        semantic,
        mask=semantic_mask,
        reference_features=None,
        reference_mask=None,
        generator=_generator(),
    )
    support_without_condition = generator.conditions[-1]
    support_with = support.sample_features(
        semantic,
        mask=semantic_mask,
        reference_features=reference,
        reference_mask=reference_mask,
        generator=_generator(),
    )
    support_with_condition = generator.conditions[-1]

    runtime_without = runtime.sample_features(
        semantic,
        mask=semantic_mask,
        reference_features=None,
        reference_mask=None,
        generator=_generator(),
    )
    runtime_with = runtime.sample_features(
        semantic,
        mask=semantic_mask,
        reference_features=reference,
        reference_mask=reference_mask,
        generator=_generator(),
    )
    decoded_without = runtime.decode(
        semantic,
        mask=semantic_mask,
        reference_features=None,
        reference_mask=None,
        generator=_generator(),
    )
    decoded_with = runtime.decode(
        semantic,
        mask=semantic_mask,
        reference_features=reference,
        reference_mask=reference_mask,
        generator=_generator(),
    )

    assert support_without.shape == support_with.shape == (1, 3, 4)
    assert not torch.allclose(support_without_condition, support_with_condition)
    assert torch.allclose(
        support_with - support_without,
        support_with_condition - support_without_condition,
    )
    assert runtime_without.shape == runtime_with.shape == (1, 3, 4)
    assert not torch.allclose(runtime_without, runtime_with)
    assert decoded_without.shape == decoded_with.shape == (1, 1, 12)
    assert not torch.allclose(decoded_without, decoded_with)


def test_support_acoustic_sampling_accepts_optional_reference() -> None:
    semantic = torch.tensor([[[1], [2], [3]]], dtype=torch.long)
    semantic_mask = torch.ones(1, 3, dtype=torch.bool)
    reference = torch.tensor(
        [[[1.0, 0.0, -1.0, 2.0], [0.0, 2.0, 1.0, -2.0]]],
    )
    reference_mask = torch.tensor([[True, True]])
    support, generator = _support(Route.RVQ)

    without = support.sample_acoustic_codes(
        semantic,
        mask=semantic_mask,
        reference_features=None,
        reference_mask=None,
        generator=_generator(),
    )
    without_condition = generator.conditions[-1]
    with_reference = support.sample_acoustic_codes(
        semantic,
        mask=semantic_mask,
        reference_features=reference,
        reference_mask=reference_mask,
        generator=_generator(),
    )
    with_condition = generator.conditions[-1]

    assert without.shape == with_reference.shape == (1, 3, 1)
    # Fresh generators with the same seed isolate the reference-conditioned input.
    assert torch.equal(without, with_reference)
    assert not torch.allclose(without_condition, with_condition)


def test_runtime_rvq_sample_features_zeroes_padding_frames() -> None:
    class NonZeroPadBackend(FakeBackend):
        def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
            return (acoustic_codes.float() + 1).expand(
                -1,
                -1,
                self.acoustic_feature_dim,
            ).contiguous()

    support, _ = _support(Route.RVQ)
    runtime = SemanticCodecRuntime(support, NonZeroPadBackend())
    semantic = torch.tensor([[[1], [2], [0]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])

    features = runtime.sample_features(semantic, mask=mask, generator=_generator())

    assert features.shape == (1, 3, 4)
    assert bool((features[0, :2] != 0).all())
    assert torch.equal(features[0, 2], torch.zeros(4))


def test_schema_seven_artifact_roundtrip_preserves_decoder_fields(
    tmp_path,
    monkeypatch,
) -> None:
    backend = FakeBackend()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    )
    save_artifact(tmp_path, support, config, backend=backend)

    config_path = tmp_path / "codec.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 7
    assert data["backend"]["name"] == backend.name
    assert data["backend"]["sample_rate"] == backend.sample_rate
    assert data["backend"]["frame_rate"] == backend.frame_rate
    assert data["backend"]["semantic_frame_rate"] == backend.semantic_frame_rate
    assert data["config"]["decoder"]["rvq_predictor"] == RVQPredictor.MTP.value

    captured: list[SemanticSupportConfig] = []
    original = runtime_semantic.build_support

    def capture(options: SemanticSupportConfig, **kwargs: Any) -> SemanticCodecSupport:
        captured.append(options)
        return original(options, **kwargs)

    monkeypatch.setattr(runtime_semantic, "build_support", capture)
    loaded = runtime_semantic.load_artifact(tmp_path)

    assert len(captured) == 1
    assert captured[0].decoder.rvq_predictor is RVQPredictor.MTP
    assert loaded.acoustic_layout is AcousticLayout.FRAME_ALIGNED
    assert loaded.acoustic_unit_length is None
    assert loaded.state_dict().keys() == support.state_dict().keys()
    for key, value in support.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_schema_seven_artifact_rejects_missing_decoder_fields(tmp_path) -> None:
    backend = FakeBackend()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    )
    save_artifact(tmp_path, support, config, backend=backend)

    config_path = tmp_path / "codec.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    del data["config"]["decoder"]["rvq_predictor"]
    config_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'rvq_predictor'"):
        runtime_semantic.load_artifact(tmp_path)


def _support(route: Route) -> tuple[SemanticCodecSupport, SeededGenerator]:
    backend = FakeBackend()
    generator = SeededGenerator(backend.acoustic_feature_dim, backend.acoustic_codebook_sizes[0])
    reference = ReferenceConditioner(backend.acoustic_feature_dim, condition_dim=4)
    with torch.no_grad():
        reference.projection.weight.copy_(torch.eye(4))
        reference.projection.bias.zero_()
        reference.gate.fill_(2.0)
    modules = RouteModules(
        conditioner=SemanticConditioner(backend.semantic_codebook, condition_dim=4),
        reference_conditioner=reference,
        generator=cast(CodecUnitGenerator, cast(object, generator)),
        route=route,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    )
    support = SemanticCodecSupport(
        modules,
        backend.acoustic_feature_dim,
        sampling=SamplingConfig(flow_steps=1),
    )
    return support, generator


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(17)


def _validate_frame_aligned(
    condition: torch.Tensor,
    mask: torch.Tensor,
    acoustic_layout: AcousticLayout,
    output_length: int | None,
) -> None:
    if acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
        raise AssertionError("test generator only supports frame-aligned conditions")
    if output_length is not None:
        raise AssertionError("frame-aligned generation must not provide output_length")
    if mask.shape != condition.shape[:2]:
        raise AssertionError("condition and mask must align")
