from __future__ import annotations

import json
from typing import Any, cast

import pytest
import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes, semantic_acoustic_spec
from torch import nn

import semantic_acoustic_generator.runtime.semantic as runtime_semantic
from semantic_acoustic_generator.config import (
    AnchorContext,
    AnchorTarget,
    DecoderConfig,
    FactorPredictor,
    FeatureAdapter,
    FMMode,
    Route,
    RVQPredictor,
)
from semantic_acoustic_generator.model.condition import ReferenceConditioner, SemanticConditioner
from semantic_acoustic_generator.model.decoder import AcousticUnitGenerator
from semantic_acoustic_generator.model.routes import RouteModules
from semantic_acoustic_generator.runtime import (
    GeneratorConfig,
    GeneratorRuntime,
    GeneratorSupport,
    SamplingConfig,
    build_support,
)
from semantic_acoustic_generator.runtime.artifact import load_artifact, save_artifact


@pytest.mark.parametrize("value", [True, "16"])
def test_sampling_config_rejects_non_integer_flow_steps(value: object) -> None:
    with pytest.raises(TypeError, match="flow_steps"):
        SamplingConfig(flow_steps=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["temperature", "top_p", "cfg_scale"])
@pytest.mark.parametrize("value", [True, "1.0"])
def test_sampling_config_rejects_non_numeric_fields(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        SamplingConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["temperature", "top_p", "cfg_scale"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_sampling_config_rejects_non_finite_fields(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        SamplingConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("route", "fm", "route must be a Route"),
        ("condition_dim", True, "condition_dim must be an integer"),
        ("condition_dim", "4", "condition_dim must be an integer"),
        ("decoder", object(), "decoder must be a DecoderConfig"),
        ("initialization", "codec", "initialization must be an Initialization"),
        ("seed", True, "seed must be an integer"),
        ("seed", "0", "seed must be an integer"),
        ("sampling", object(), "sampling must be a SamplingConfig"),
        ("feature_adapter", "none", "feature_adapter must be a FeatureAdapter"),
    ],
)
def test_semantic_support_config_rejects_invalid_field_types(
    field: str,
    value: object,
    message: str,
) -> None:
    options: dict[str, object] = {"route": Route.FM, "condition_dim": 4}
    options[field] = value

    with pytest.raises(TypeError, match=message):
        GeneratorConfig(**options)  # type: ignore[arg-type]


def test_factor_target_requires_longcat_first_codebook_adapter() -> None:
    with pytest.raises(ValueError, match="requires feature_adapter=longcat_first_codebook"):
        GeneratorConfig(
            route=Route.FM,
            condition_dim=4,
            decoder=DecoderConfig(
                fm_mode=FMMode.ANCHOR,
                anchor_target=AnchorTarget.FACTOR,
            ),
        )


def test_factor_target_accepts_longcat_multi_codebook_adapter() -> None:
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        feature_adapter=FeatureAdapter.LONGCAT_CODEBOOKS,
        feature_codebooks=3,
        decoder=DecoderConfig(
            fm_mode=FMMode.ANCHOR,
            anchor_target=AnchorTarget.FACTOR,
        ),
    )

    assert config.feature_codebooks == 3


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("feature_mean", (True,), TypeError),
        ("feature_std", ("1.0",), TypeError),
        ("feature_mean", (float("nan"),), ValueError),
        ("feature_std", (float("inf"),), ValueError),
    ],
)
def test_semantic_support_config_requires_finite_numeric_feature_stats(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    options: dict[str, object] = {
        "route": Route.FM,
        "condition_dim": 4,
        "feature_mean": (0.0,),
        "feature_std": (1.0,),
    }
    options[field] = value

    with pytest.raises(error, match=field):
        GeneratorConfig(**options)  # type: ignore[arg-type]


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
        self.unconditional_conditions: list[torch.Tensor | None] = []
        self.cfg_scales: list[float] = []

    def sample_features(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        flow_steps: int,
        unconditional_condition: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del feature_mean, feature_std, flow_steps
        _validate_frame_aligned(condition, mask)
        self.conditions.append(condition.detach().clone())
        self.unconditional_conditions.append(
            None if unconditional_condition is None else unconditional_condition.detach().clone()
        )
        self.cfg_scales.append(float(cfg_scale))
        noise = torch.randn(
            (*condition.shape[:2], self.feature_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        base = condition[..., : self.feature_dim]
        if unconditional_condition is not None:
            unconditional = unconditional_condition[..., : self.feature_dim]
            base = unconditional + cfg_scale * (base - unconditional)
        return base + noise

    def sample_acoustic_codes(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del temperature, top_p
        _validate_frame_aligned(condition, mask)
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
    runtime = GeneratorRuntime(support, FakeBackend())

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


def test_fm_support_applies_cfg_against_null_reference() -> None:
    semantic = torch.tensor([[[1], [2], [3]]], dtype=torch.long)
    semantic_mask = torch.ones(1, 3, dtype=torch.bool)
    reference = torch.tensor(
        [[[1.0, 0.0, -1.0, 2.0], [0.0, 2.0, 1.0, -2.0]]],
    )
    reference_mask = torch.tensor([[True, True]])
    support, generator = _support(Route.FM, sampling=SamplingConfig(flow_steps=1, cfg_scale=2.0))

    guided = support.sample_features(
        semantic,
        mask=semantic_mask,
        reference_features=reference,
        reference_mask=reference_mask,
        generator=_generator(),
    )
    conditional = generator.conditions[-1]
    unconditional = generator.unconditional_conditions[-1]
    cfg_scale = generator.cfg_scales[-1]
    without = support.sample_features(
        semantic,
        mask=semantic_mask,
        reference_features=None,
        reference_mask=None,
        generator=_generator(),
    )

    assert unconditional is not None
    assert cfg_scale == 2.0
    assert torch.allclose(guided - without, cfg_scale * (conditional - unconditional))


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
            return (
                (acoustic_codes.float() + 1)
                .expand(
                    -1,
                    -1,
                    self.acoustic_feature_dim,
                )
                .contiguous()
            )

    support, _ = _support(Route.RVQ)
    runtime = GeneratorRuntime(support, NonZeroPadBackend())
    semantic = torch.tensor([[[1], [2], [0]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])

    features = runtime.sample_features(semantic, mask=mask, generator=_generator())

    assert features.shape == (1, 3, 4)
    assert bool((features[0, :2] != 0).all())
    assert torch.equal(features[0, 2], torch.zeros(4))


def test_sampling_prepares_semantic_input_once_per_public_call(monkeypatch) -> None:
    original = GeneratorSupport._semantic_input
    calls: list[Route] = []

    def record_input(
        self: GeneratorSupport,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(self.route)
        return original(self, value, mask, validate=validate)

    monkeypatch.setattr(GeneratorSupport, "_semantic_input", record_input)
    semantic = torch.tensor([[[1], [2]]], dtype=torch.long)
    mask = torch.ones(1, 2, dtype=torch.bool)
    fm_support, _ = _support(Route.FM)
    rvq_support, _ = _support(Route.RVQ)

    fm_support.sample_features(semantic, mask=mask, generator=_generator())
    assert calls == [Route.FM]
    calls.clear()
    GeneratorRuntime(fm_support, FakeBackend()).decode(
        semantic,
        mask=mask,
        generator=_generator(),
    )
    assert calls == [Route.FM]
    calls.clear()
    GeneratorRuntime(rvq_support, FakeBackend()).sample_features(
        semantic,
        mask=mask,
        generator=_generator(),
    )
    assert calls == [Route.RVQ]


def test_schema_eight_artifact_roundtrip_preserves_decoder_fields(
    tmp_path,
    monkeypatch,
) -> None:
    backend = FakeBackend()
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
    )
    save_artifact(tmp_path, support, backend=backend)

    config_path = tmp_path / "generator.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 8
    assert not (tmp_path / "codec.json").exists()
    assert data["backend"]["name"] == backend.name
    assert data["backend"]["sample_rate"] == backend.sample_rate
    assert data["backend"]["frame_rate"] == backend.frame_rate
    assert data["backend"]["semantic_frame_rate"] == backend.semantic_frame_rate
    assert data["config"]["decoder"]["rvq_predictor"] == RVQPredictor.MTP.value
    assert data["config"]["decoder"]["factor_predictor"] == FactorPredictor.PARALLEL.value
    assert data["config"]["feature_adapter"] == FeatureAdapter.NONE.value
    assert data["config"]["feature_codebooks"] == 1
    assert data["config"]["sampling"]["cfg_scale"] == 1.0

    captured: list[GeneratorConfig] = []
    original = runtime_semantic.build_support

    def capture(options: GeneratorConfig, **kwargs: Any) -> GeneratorSupport:
        captured.append(options)
        return original(options, **kwargs)

    monkeypatch.setattr(runtime_semantic, "build_support", capture)
    loaded = load_artifact(tmp_path)

    assert len(captured) == 1
    assert captured[0].decoder.rvq_predictor is RVQPredictor.MTP
    assert captured[0].decoder.factor_predictor is FactorPredictor.PARALLEL
    assert captured[0].feature_adapter is FeatureAdapter.NONE
    assert captured[0].sampling.cfg_scale == 1.0
    assert loaded.acoustic_layout is AcousticLayout.FRAME_ALIGNED
    assert loaded.acoustic_unit_length is None
    assert loaded.state_dict().keys() == support.state_dict().keys()
    for key, value in support.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_schema_seven_artifact_remains_readable(tmp_path) -> None:
    backend = FakeBackend()
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
    )
    save_artifact(tmp_path, support, backend=backend)
    current = tmp_path / "generator.json"
    data = json.loads(current.read_text(encoding="utf-8"))
    data["schema_version"] = 7
    (tmp_path / "codec.json").write_text(json.dumps(data), encoding="utf-8")
    current.unlink()

    loaded = load_artifact(tmp_path)

    for key, value in support.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_early_schema_eight_artifact_defaults_new_fm_fields(tmp_path) -> None:
    backend = FakeBackend()
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
    )
    save_artifact(tmp_path, support, backend=backend)
    path = tmp_path / "generator.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["config"]["feature_adapter"]
    del data["config"]["feature_codebooks"]
    for key in (
        "fm_mode",
        "anchor_context",
        "anchor_target",
        "factor_predictor",
        "anchor_hidden_dim",
        "anchor_layers",
        "anchor_kernel_size",
        "anchor_cosine_weight",
        "anchor_factor_weight",
        "anchor_factor_temperature",
    ):
        del data["config"]["decoder"][key]
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_artifact(tmp_path)

    assert loaded.config is not None
    assert loaded.config.feature_adapter is FeatureAdapter.NONE
    assert loaded.config.feature_codebooks == 1
    assert loaded.config.decoder.fm_mode is FMMode.FLOW
    assert loaded.config.decoder.anchor_context is AnchorContext.LOCAL
    assert loaded.config.decoder.anchor_target is AnchorTarget.FEATURE
    assert loaded.config.decoder.factor_predictor is FactorPredictor.PARALLEL


def test_schema_eight_artifact_rejects_missing_decoder_fields(tmp_path) -> None:
    backend = FakeBackend()
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
    )
    save_artifact(tmp_path, support, backend=backend)

    config_path = tmp_path / "generator.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    del data["config"]["decoder"]["rvq_predictor"]
    config_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'rvq_predictor'"):
        load_artifact(tmp_path)


def test_schema_eight_artifact_rejects_missing_cfg_scale(tmp_path) -> None:
    backend = FakeBackend()
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
    )
    save_artifact(tmp_path, support, backend=backend)

    config_path = tmp_path / "generator.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    del data["config"]["sampling"]["cfg_scale"]
    config_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'cfg_scale'"):
        load_artifact(tmp_path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("config", "condition_dim", True, "config.condition_dim"),
        ("decoder", "layers", "1", "decoder.layers"),
        ("sampling", "temperature", "1.0", "sampling.temperature"),
    ],
)
def test_schema_eight_artifact_rejects_lossy_numeric_coercion(
    tmp_path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    backend = FakeBackend()
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
    )
    save_artifact(tmp_path, support, backend=backend)

    config_path = tmp_path / "generator.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    target = data["config"] if section == "config" else data["config"][section]
    target[field] = value
    config_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(TypeError, match=message.replace(".", r"\.")):
        load_artifact(tmp_path)


@pytest.mark.parametrize("value", [False, "0.0", float("nan")])
def test_schema_eight_artifact_rejects_invalid_feature_metadata(tmp_path, value: object) -> None:
    backend = FakeBackend()
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=1),
        feature_mean=(0.0, 0.0, 0.0, 0.0),
        feature_std=(1.0, 1.0, 1.0, 1.0),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        codec_spec=semantic_acoustic_spec(backend),
    )
    save_artifact(tmp_path, support, backend=backend)

    config_path = tmp_path / "generator.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["config"]["feature_mean"][0] = value
    config_path.write_text(json.dumps(data), encoding="utf-8")

    error = ValueError if isinstance(value, float) else TypeError
    with pytest.raises(error, match="feature_mean metadata"):
        load_artifact(tmp_path)


def _support(
    route: Route,
    *,
    sampling: SamplingConfig | None = None,
) -> tuple[GeneratorSupport, SeededGenerator]:
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
        generator=cast(AcousticUnitGenerator, cast(object, generator)),
        route=route,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    )
    support = GeneratorSupport(
        modules,
        semantic_acoustic_spec(backend),
        sampling=SamplingConfig(flow_steps=1) if sampling is None else sampling,
    )
    return support, generator


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(17)


def _validate_frame_aligned(
    condition: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if mask.shape != condition.shape[:2]:
        raise AssertionError("condition and mask must align")
