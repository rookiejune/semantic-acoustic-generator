from __future__ import annotations

import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes, masked_acoustic_features
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from anytrain.loss import (
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    MaskedFrameMSELoss,
)
from anytrain.module.qwen import QwenMTPCodebookPredictor

import semantic_acoustic_codec.model.rvq as rvq_module
from semantic_acoustic_codec.config import DecoderConfig, Route, RVQPredictor
from semantic_acoustic_codec.model import (
    AcousticRVQDecoder,
    DiTDecoder,
    FMFeatureGenerator,
    ReferenceConditioner,
    RVQCodeGenerator,
    SemanticConditioner,
    build_route,
)
from semantic_acoustic_codec.model.condition import FixedLengthConditioner
from semantic_acoustic_codec.runtime import (
    SamplingConfig,
    SemanticCodecRuntime,
    SemanticSupportConfig,
    build_support,
)
from semantic_acoustic_codec.runtime.artifact import (
    load_artifact,
    load_generator_artifact,
    save_artifact,
)
from semantic_acoustic_codec.types import SemanticCodecBatch


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
        self, semantic_codes: torch.Tensor, acoustic_features: torch.Tensor
    ) -> torch.Tensor:
        return acoustic_features.new_zeros(
            (semantic_codes.size(0), 1, semantic_codes.size(1) * 320)
        )


def test_decoder_config_defaults_to_temporal_mtp() -> None:
    assert DecoderConfig().rvq_predictor is RVQPredictor.MTP


def test_conditioner_shapes() -> None:
    conditioner = SemanticConditioner(
        torch.randn(16, 8),
        condition_dim=12,
    )
    semantic = torch.tensor([[[1], [2], [3]], [[4], [5], [0]]], dtype=torch.long)
    condition = conditioner(semantic)

    assert condition.shape == (2, 3, 12)
    assert conditioner.semantic_pad_id == 16
    assert torch.equal(
        conditioner.embedding.weight[conditioner.semantic_pad_id],
        torch.zeros_like(conditioner.embedding.weight[conditioner.semantic_pad_id]),
    )
    padded = conditioner(torch.tensor([[[conditioner.semantic_pad_id]]], dtype=torch.long))
    assert torch.equal(padded, torch.zeros_like(padded))


def test_conditioner_validates_semantic_ids() -> None:
    conditioner = SemanticConditioner(torch.randn(8, 6), condition_dim=6)

    conditioner(torch.tensor([[[conditioner.semantic_pad_id]]], dtype=torch.long))
    with pytest.raises(ValueError, match="negative"):
        conditioner(torch.tensor([[[-1]]], dtype=torch.long))
    with pytest.raises(ValueError, match="outside"):
        conditioner(torch.tensor([[[conditioner.semantic_pad_id + 1]]], dtype=torch.long))


def test_fm_loss_and_sample_shapes() -> None:
    condition = torch.randn(2, 4, 10)
    target = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    decoder = DiTDecoder(10, 6, layers=1, heads=2, ffn_ratio=2)

    runtime = ContinuousFlowRuntime()
    flow_sample = runtime.training_sample(target)
    prediction = decoder(
        flow_sample.x_t,
        flow_sample.t,
        condition=condition,
        mask=mask,
    )
    loss = MaskedFrameMSELoss()(
        prediction,
        flow_sample.velocity,
        mask,
        details={"t": flow_sample.t},
        detail_dtype=target.dtype,
    )
    sample = decoder.sample(condition, mask=mask, steps=2)

    assert loss.loss.shape == (2,)
    assert sample.shape == target.shape
    assert torch.equal(sample[1, 2:], torch.zeros_like(sample[1, 2:]))


def test_fm_sample_applies_classifier_free_guidance() -> None:
    class GuidanceProbe(torch.nn.Module):
        latent_dim = 2

        def __init__(self) -> None:
            super().__init__()
            self.validations: list[bool] = []

        def prepare_condition(self, condition: torch.Tensor) -> torch.Tensor:
            return condition

        def forward(
            self,
            latent: torch.Tensor,
            t: torch.Tensor,
            *,
            condition_state: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            validate: bool = True,
        ) -> torch.Tensor:
            del latent, t, mask
            self.validations.append(validate)
            if condition_state is None:
                raise AssertionError("condition_state is required")
            return condition_state

    decoder = DiTDecoder(2, 2, layers=1, heads=1, ffn_ratio=2)
    probe = GuidanceProbe()
    decoder.decoder = probe
    condition = torch.tensor([[[3.0, 5.0], [7.0, 11.0]]])
    unconditional = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    seed = 37

    sample = decoder.sample(
        condition,
        steps=1,
        unconditional_condition=unconditional,
        guidance_scale=1.5,
        generator=torch.Generator().manual_seed(seed),
    )
    initial = torch.randn(condition.shape, generator=torch.Generator().manual_seed(seed))
    expected = initial + unconditional + 1.5 * (condition - unconditional)

    assert torch.allclose(sample, expected)
    assert probe.validations == [False, False]


def test_fm_sample_validates_mask_before_the_flow_loop() -> None:
    decoder = DiTDecoder(2, 2, layers=1, heads=1, ffn_ratio=2)
    condition = torch.randn(1, 2, 2)

    with pytest.raises(ValueError, match="at least one valid frame"):
        decoder.sample(condition, mask=torch.zeros(1, 2, dtype=torch.bool))


def test_reference_conditioner_uses_explicit_and_null_reference() -> None:
    conditioner = ReferenceConditioner(feature_dim=4, condition_dim=6)
    features = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    explicit = conditioner(features, mask=mask, batch_size=2)
    null = conditioner(None, batch_size=2)
    mixed = conditioner(
        features,
        mask=mask,
        batch_size=2,
        use_reference=torch.tensor([True, False]),
    )

    assert explicit.shape == (2, 1, 6)
    assert null.shape == (2, 1, 6)
    assert torch.equal(mixed[1], null[1])
    assert "null_condition" in conditioner.state_dict()
    assert "default_feature" not in conditioner.state_dict()


def test_reference_conditioner_pool_before_projection_preserves_outputs_and_gradients() -> None:
    conditioner = ReferenceConditioner(feature_dim=4, condition_dim=6)
    legacy = ReferenceConditioner(feature_dim=4, condition_dim=6)
    with torch.no_grad():
        conditioner.gate.fill_(0.7)
    legacy.load_state_dict(conditioner.state_dict())
    features = torch.randn(2, 4, 4, requires_grad=True)
    legacy_features = features.detach().clone().requires_grad_(True)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    output_weight = torch.randn(2, 1, 6)

    actual = conditioner(features, mask=mask)
    projected = legacy.projection(legacy_features)
    weights = mask[..., None].to(dtype=projected.dtype)
    pooled = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    expected = legacy.norm(pooled)[:, None] * torch.tanh(legacy.gate)[None, None]
    (actual * output_weight).sum().backward()
    (expected * output_weight).sum().backward()

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert features.grad is not None
    assert legacy_features.grad is not None
    assert torch.allclose(features.grad, legacy_features.grad, atol=1e-6, rtol=1e-5)
    for (name, parameter), (legacy_name, legacy_parameter) in zip(
        conditioner.named_parameters(),
        legacy.named_parameters(),
    ):
        assert name == legacy_name
        if parameter.grad is None or legacy_parameter.grad is None:
            assert parameter.grad is legacy_parameter.grad is None
            continue
        assert torch.allclose(parameter.grad, legacy_parameter.grad, atol=1e-6, rtol=1e-5)


def test_reference_conditioner_projects_only_pooled_rows() -> None:
    conditioner = ReferenceConditioner(feature_dim=4, condition_dim=6)
    projected_shapes: list[tuple[int, ...]] = []

    def record_projection_batch(_module, inputs) -> None:
        projected_shapes.append(tuple(inputs[0].shape))

    handle = conditioner.projection.register_forward_pre_hook(record_projection_batch)
    output = conditioner(
        torch.randn(3, 2, 4),
        use_reference=torch.tensor([True, False, True]),
    )
    handle.remove()

    assert projected_shapes == [(3, 4)]
    assert torch.equal(output[1, 0], conditioner.null_condition)


def test_fixed_length_conditioner_uses_slot_queries_and_full_semantic_memory() -> None:
    conditioner = FixedLengthConditioner(condition_dim=4, slots=3)
    with torch.no_grad():
        conditioner.slot_queries.zero_()
        conditioner.slot_queries[0, 0] = 1
        conditioner.slot_queries[1, 1] = 1
        conditioner.slot_queries[2, 2] = 1
    memory = torch.tensor(
        [[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5], [9.0, 9.0, 9.0, 9.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, False]])

    output = conditioner(memory, mask, output_length=3)
    padded_changed = memory.detach().clone()
    padded_changed[:, 2] = -100
    unchanged = conditioner(padded_changed, mask, output_length=3)
    permuted = conditioner(memory.detach()[:, [1, 0, 2]], mask, output_length=3)
    weights = output.new_tensor([1.0, 2.0, 3.0, 4.0])
    (output * weights).sum().backward()

    assert conditioner.slot_queries.shape == (3, 4)
    assert output.shape == (1, 3, 4)
    assert not torch.allclose(output[:, 0], output[:, 1])
    assert torch.allclose(output, unchanged)
    assert not torch.allclose(output, permuted)
    assert memory.grad is not None
    assert bool((memory.grad[0, :2].abs().sum(dim=-1) > 0).all())
    assert torch.equal(memory.grad[0, 2], torch.zeros_like(memory.grad[0, 2]))


def test_fm_loss_returns_repa_features_when_configured() -> None:
    condition = torch.randn(2, 4, 10)
    target = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    decoder = DiTDecoder(
        10,
        6,
        layers=2,
        heads=2,
        ffn_ratio=2,
        repa_feature_dim=5,
        repa_student_layer=1,
    )

    runtime = ContinuousFlowRuntime()
    flow_sample = runtime.training_sample(target)
    prediction, representation = decoder.forward_with_features(
        flow_sample.x_t,
        flow_sample.t,
        condition=condition,
        mask=mask,
    )
    item = MaskedFrameMSELoss()(
        prediction,
        flow_sample.velocity,
        mask,
        details={"t": flow_sample.t},
        detail_dtype=target.dtype,
    )

    assert item.loss.shape == (2,)
    assert representation.shape == (2, 4, 5)


def test_repa_loss_detaches_teacher_and_ignores_padding() -> None:
    representation = torch.tensor(
        [[[1.0, 0.0], [float("nan"), float("inf")]]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[[1.0, 0.0], [float("nan"), float("inf")]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, False]])

    item = MaskedCosineAlignmentLoss()(representation, target, mask)
    item.loss.mean().backward()

    assert torch.isfinite(item.loss).all()
    assert representation.grad is not None
    assert target.grad is None
    assert torch.isfinite(representation.grad).all()
    assert torch.equal(representation.grad[:, 1], torch.zeros_like(representation.grad[:, 1]))


def test_build_route_and_masked_acoustic_features() -> None:
    backend = FakeCodec()
    modules = build_route(
        Route.FM,
        backend.semantic_codebook,
        backend.acoustic_feature_dim,
        backend.acoustic_codebook_sizes,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    acoustic = torch.tensor([[[1, 2], [3, 4], [5, 7]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    features = masked_acoustic_features(backend, acoustic, mask)

    assert modules.route is Route.FM
    assert isinstance(modules.generator, FMFeatureGenerator)
    assert backend.sample_rate == 16_000
    assert backend.frame_rate == 50.0
    assert backend.acoustic_feature_dim == 4
    assert backend.acoustic_codebook_sizes == (5, 7)
    assert features.shape == (1, 3, 4)
    assert torch.equal(features[:, 2], torch.zeros_like(features[:, 2]))


def test_semantic_support_decodes_and_roundtrips_artifact(tmp_path) -> None:
    backend = FakeCodec()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    ).eval()
    semantic = torch.tensor([[[1], [2], [8]]], dtype=torch.long)
    reference = torch.tensor([[[1, 2], [3, 4], [5, 7]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])

    features = support.sample_features(
        semantic, mask=mask, generator=torch.Generator().manual_seed(0)
    )
    condition = support.condition(
        semantic,
        mask=mask,
        reference_features=masked_acoustic_features(backend, reference, mask),
        reference_mask=mask,
    )
    runtime = SemanticCodecRuntime(support, backend)
    waveform = runtime.decode(semantic, mask=mask, generator=torch.Generator().manual_seed(1))
    save_artifact(tmp_path, support, backend=backend)
    loaded = load_artifact(tmp_path)
    acoustic = load_generator_artifact(tmp_path)
    loaded_features = loaded.sample_features(
        semantic,
        mask=mask,
        generator=torch.Generator().manual_seed(0),
    )

    assert features.shape == (1, 3, 4)
    assert condition.shape == (1, 3, 10)
    assert waveform.shape == (1, 1, 2 * 320)
    assert not hasattr(support, "backend")
    assert acoustic.spec.route is Route.FM
    assert acoustic.spec.condition_dim == 10
    assert acoustic.spec.backend_name == backend.name
    assert acoustic.spec.sample_rate == backend.sample_rate
    assert acoustic.spec.frame_rate == backend.frame_rate
    assert acoustic.spec.semantic_frame_rate == backend.semantic_frame_rate
    assert acoustic.spec.semantic_vocab_size == 8
    assert acoustic.spec.semantic_embedding_dim == 6
    assert acoustic.spec.acoustic_feature_dim == 4
    assert acoustic.spec.acoustic_codebook_sizes == (5, 7)
    assert acoustic.spec.acoustic_layout is AcousticLayout.FRAME_ALIGNED
    assert acoustic.spec.acoustic_unit_length is None
    acoustic.spec.validate_backend(backend)
    acoustic.spec.validate_acoustic_backend(backend)
    other_semantic = FakeCodec()
    other_semantic.semantic_codebook = torch.randn(11, 9)
    acoustic.spec.validate_acoustic_backend(other_semantic)
    with pytest.raises(ValueError, match="semantic_vocab_size"):
        acoustic.spec.validate_backend(other_semantic)
    other_identity = FakeCodec()
    other_identity.name = "other"
    with pytest.raises(ValueError, match="name"):
        acoustic.spec.validate_backend(other_identity)
    other_rate = FakeCodec()
    other_rate.sample_rate = 24_000
    with pytest.raises(ValueError, match="sample_rate"):
        acoustic.spec.validate_backend(other_rate)
    assert acoustic.generator.state_dict().keys() == support.generator.state_dict().keys()
    for key, value in support.generator.state_dict().items():
        assert torch.equal(acoustic.generator.state_dict()[key], value)
    assert torch.equal(features[:, 2], torch.zeros_like(features[:, 2]))
    assert torch.allclose(features, loaded_features)


def test_artifact_rejects_runtime_state_that_differs_from_construction_config(tmp_path) -> None:
    backend = FakeCodec()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
        sampling=SamplingConfig(flow_steps=2),
        feature_mean=(0.0, 0.0, 0.0, 0.0),
        feature_std=(1.0, 1.0, 1.0, 1.0),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    )
    support.sampling = replace(config.sampling, temperature=0.5)

    with pytest.raises(ValueError, match="support sampling"):
        save_artifact(tmp_path, support, backend=backend)

    assert not (tmp_path / "codec.json").exists()
    assert not (tmp_path / "model.ckpt").exists()


def test_generator_artifact_ignores_unrelated_state_but_remains_strict(tmp_path) -> None:
    backend = FakeCodec()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    ).eval()
    save_artifact(tmp_path, support, backend=backend)
    checkpoint = tmp_path / "model.ckpt"
    original = torch.load(checkpoint, weights_only=True)
    incompatible = dict(original)
    del incompatible["reference_conditioner.null_condition"]
    incompatible["conditioner.legacy.weight"] = torch.zeros(1)
    torch.save(incompatible, checkpoint)

    with pytest.raises(RuntimeError, match="state_dict"):
        load_artifact(tmp_path)
    acoustic = load_generator_artifact(tmp_path)
    for key, value in support.generator.state_dict().items():
        assert torch.equal(acoustic.generator.state_dict()[key], value)

    generator_keys = [key for key in original if key.startswith("generator.")]
    missing = dict(incompatible)
    del missing[generator_keys[0]]
    torch.save(missing, checkpoint)
    with pytest.raises(RuntimeError, match="state_dict"):
        load_generator_artifact(tmp_path)

    unexpected = dict(incompatible)
    unexpected["generator.unexpected"] = torch.zeros(1)
    torch.save(unexpected, checkpoint)
    with pytest.raises(RuntimeError, match="state_dict"):
        load_generator_artifact(tmp_path)


def test_artifact_loads_cpu_state_before_moving_to_target_device(tmp_path, monkeypatch) -> None:
    backend = FakeCodec()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    )
    save_artifact(tmp_path, support, backend=backend)
    load_options: list[tuple[object, object]] = []
    original_load = torch.load

    def record_load(*args, **kwargs):
        load_options.append((kwargs.get("map_location"), kwargs.get("weights_only")))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    loaded = load_artifact(tmp_path, device="meta")
    acoustic = load_generator_artifact(tmp_path, device="meta")

    assert load_options == [("cpu", True), ("cpu", True)]
    assert next(loaded.parameters()).device.type == "meta"
    assert next(acoustic.generator.parameters()).device.type == "meta"


def test_frame_aligned_decode_trims_right_padding() -> None:
    backend = FakeCodec()
    support = build_support(
        SemanticSupportConfig(
            route=Route.FM,
            condition_dim=10,
            decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
        ),
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    ).eval()
    runtime = SemanticCodecRuntime(support, backend)
    semantic = torch.tensor([[[1], [2], [8]]], dtype=torch.long)
    features = torch.randn(1, 3, backend.acoustic_feature_dim)
    mask = torch.tensor([[True, True, False]])

    waveform = runtime.decode_features(semantic, features, mask=mask)

    assert waveform.shape == (1, 1, 2 * 320)


def test_decode_rejects_mixed_lengths_and_non_prefix_masks() -> None:
    backend = FakeCodec()
    support = build_support(
        SemanticSupportConfig(
            route=Route.FM,
            condition_dim=10,
            decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
        ),
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
    ).eval()
    runtime = SemanticCodecRuntime(support, backend)
    semantic = torch.tensor([[[1], [2], [8]], [[3], [8], [8]]], dtype=torch.long)

    with pytest.raises(ValueError, match="equal valid semantic lengths"):
        runtime.decode(
            semantic,
            mask=torch.tensor([[True, True, False], [True, False, False]]),
        )
    with pytest.raises(ValueError, match="contiguous right padding"):
        runtime.decode(
            torch.tensor([[[1], [8], [2]]], dtype=torch.long),
            mask=torch.tensor([[True, False, True]]),
        )


def test_fm_route_trains_one_step() -> None:
    backend = FakeCodec()
    semantic = torch.tensor([[[1], [2], [8]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [5, 7]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    target = masked_acoustic_features(backend, acoustic, mask)

    modules = build_route(
        Route.FM,
        backend.semantic_codebook,
        backend.acoustic_feature_dim,
        backend.acoustic_codebook_sizes,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    optimizer = torch.optim.AdamW(
        list(modules.conditioner.parameters())
        + list(modules.reference_conditioner.parameters())
        + list(modules.generator.parameters()),
        lr=1e-3,
    )
    reference = modules.reference_conditioner(target, mask=mask, batch_size=1)
    condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
    generator = modules.generator
    assert isinstance(generator, FMFeatureGenerator)
    output = generator.loss(
        _batch(semantic, acoustic, mask),
        condition,
        target,
        feature_mean=torch.zeros(1, 1, backend.acoustic_feature_dim),
        feature_std=torch.ones(1, 1, backend.acoustic_feature_dim),
    )
    loss = output.loss
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_fm_generator_accepts_external_condition() -> None:
    backend = FakeCodec()
    semantic = torch.tensor([[[1], [2], [8]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [5, 7]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    target = masked_acoustic_features(backend, acoustic, mask)
    modules = build_route(
        Route.FM,
        backend.semantic_codebook,
        backend.acoustic_feature_dim,
        backend.acoustic_codebook_sizes,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    condition = modules.conditioner(semantic).masked_fill(~mask[..., None], 0)
    generator = modules.generator
    assert isinstance(generator, FMFeatureGenerator)

    output = generator.feature_loss_from_condition(
        condition,
        mask,
        target_features=target,
        feature_mean=torch.zeros(1, 1, backend.acoustic_feature_dim),
        feature_std=torch.ones(1, 1, backend.acoustic_feature_dim),
    )

    output.loss.backward()
    assert torch.isfinite(output.loss)


def test_rvq_route_trains_one_step_when_qwen3_builder_is_available() -> None:
    if not _has_qwen3_builder():
        pytest.skip("RVQ route requires transformers and anytrain.module.qwen.build_qwen3_model.")

    backend = FakeCodec()
    semantic = torch.tensor([[[1], [2], [8]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [5, 7]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    modules = build_route(
        Route.RVQ,
        backend.semantic_codebook,
        backend.acoustic_feature_dim,
        backend.acoustic_codebook_sizes,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    optimizer = torch.optim.AdamW(
        list(modules.conditioner.parameters())
        + list(modules.reference_conditioner.parameters())
        + list(modules.generator.parameters()),
        lr=1e-3,
    )
    target = masked_acoustic_features(backend, acoustic, mask)
    reference = modules.reference_conditioner(target, mask=mask, batch_size=1)
    condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
    generator = modules.generator
    assert isinstance(generator, RVQCodeGenerator)
    output = generator.loss(_batch(semantic, acoustic, mask), condition)
    loss = output.loss
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_rvq_mtp_route_trains_and_generates_when_transformers_is_available() -> None:
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("RVQ MTP route requires transformers.")

    backend = FakeCodec()
    semantic = torch.tensor([[[1], [2], [8]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [5, 7]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    modules = build_route(
        Route.RVQ,
        backend.semantic_codebook,
        backend.acoustic_feature_dim,
        backend.acoustic_codebook_sizes,
        condition_dim=10,
        decoder=DecoderConfig(
            layers=1,
            heads=2,
            ffn_ratio=2,
            rvq_predictor=RVQPredictor.MTP,
            mtp_layers=1,
            mtp_heads=2,
        ),
    )
    target = masked_acoustic_features(backend, acoustic, mask)
    reference = modules.reference_conditioner(target, mask=mask, batch_size=1)
    condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
    generator = modules.generator
    assert isinstance(generator, RVQCodeGenerator)
    logits = generator.core(condition, acoustic, mask=mask)
    output = generator.code_loss_from_condition(condition, mask, target_codes=acoustic)
    loss = output.loss
    loss.backward()
    generated = generator.sample_acoustic_codes(
        condition,
        mask,
        temperature=1.0,
        top_p=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert isinstance(generator.core, QwenMTPCodebookPredictor)
    assert torch.isfinite(loss)
    assert [value.shape for value in logits] == [(1, 3, 5), (1, 3, 7)]
    assert generated.shape == (1, 3, 2)
    assert torch.equal(generated[:, 2], torch.zeros_like(generated[:, 2]))
    assert bool((generated[..., 0][mask] < 5).all())
    assert bool((generated[..., 1][mask] < 7).all())


def test_fixed_length_rvq_rejects_codebook_ar_and_uses_mtp_slot_axis() -> None:
    if not _has_qwen3_builder():
        pytest.skip("fixed-length RVQ requires transformers and the Qwen3 builder.")

    condition = torch.randn(1, 5, 4)
    semantic_mask = torch.tensor([[True, True, True, True, False]])
    slots = 32
    with pytest.raises(ValueError, match="fixed-length RVQ requires the MTP predictor"):
        RVQCodeGenerator(
            4,
            (5,),
            DecoderConfig(
                hidden_dim=4,
                layers=1,
                heads=1,
                ffn_ratio=2,
                rvq_predictor=RVQPredictor.CODEBOOK_AR,
            ),
            fixed_length=slots,
        )

    generator = RVQCodeGenerator(
        4,
        (5,),
        DecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            rvq_predictor=RVQPredictor.MTP,
            mtp_layers=1,
            mtp_heads=1,
        ),
        fixed_length=slots,
    )
    acoustic = torch.arange(slots).remainder(5).view(1, -1, 1)
    batch = SemanticCodecBatch(
        semantic_codes=torch.tensor([[[1], [2], [3], [4], [8]]]),
        acoustic_codes=acoustic,
        mask=semantic_mask,
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
        acoustic_mask=torch.ones(1, slots, dtype=torch.bool),
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
    )
    output = generator.loss(batch, condition)
    output.loss.backward()
    generated = generator.sample_acoustic_codes(
        condition,
        semantic_mask,
        temperature=1.0,
        top_p=1.0,
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
        output_length=slots,
        generator=torch.Generator().manual_seed(0),
    )

    assert isinstance(generator.core, QwenMTPCodebookPredictor)
    assert torch.isfinite(output.loss)
    fixed_conditioner = generator.fixed_conditioner
    assert fixed_conditioner is not None
    assert fixed_conditioner.slot_queries.grad is not None
    assert bool(torch.isfinite(fixed_conditioner.slot_queries.grad).all())
    assert generated.shape == (1, slots, 1)
    assert bool((generated < 5).all())


def test_rvq_loss_validates_per_codebook_logits() -> None:
    logits = (torch.randn(1, 2, 5), torch.randn(1, 2, 7))
    labels = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long)
    mask = torch.tensor([[True, False]])

    item = MaskedCodebookCrossEntropyLoss()(logits, labels, mask)

    assert item.loss.shape == (1,)
    assert item.details is not None
    assert set(item.details) == {"codebook_0", "codebook_1", "frames"}


class _PackedDecoderCore(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(1, hidden_dim)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del use_cache, return_dict
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def test_rvq_decoder_exposes_valid_frames_without_scattering(monkeypatch) -> None:
    monkeypatch.setattr(
        rvq_module,
        "_qwen3_model",
        lambda **options: _PackedDecoderCore(options["hidden_dim"]),
    )
    decoder = AcousticRVQDecoder(
        4,
        2,
        (5, 7),
        hidden_dim=4,
        layers=1,
        heads=1,
        ffn_ratio=2,
    )
    condition = torch.randn(2, 3, 4)
    labels = torch.tensor(
        [
            [[1, 2], [3, 4], [-1, -1]],
            [[2, 6], [-1, -1], [-1, -1]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    packed = decoder.forward_packed(condition, labels, mask=mask)
    padded = decoder(condition, labels, mask=mask)

    assert packed.labels is not None
    assert torch.equal(packed.labels, torch.tensor([[1, 2], [3, 4], [2, 6]]))
    assert torch.equal(packed.row_indices, torch.tensor([0, 0, 1]))
    assert packed.batch_size == 2
    assert [tuple(value.shape) for value in packed.logits] == [(3, 5), (3, 7)]
    for packed_value, padded_value in zip(packed.logits, padded):
        assert torch.equal(padded_value[mask], packed_value)
        assert torch.equal(padded_value[~mask], torch.zeros_like(padded_value[~mask]))


def test_rvq_codebook_ar_loss_does_not_scatter_or_revalidate(monkeypatch) -> None:
    monkeypatch.setattr(
        rvq_module,
        "_qwen3_model",
        lambda **options: _PackedDecoderCore(options["hidden_dim"]),
    )
    generator = RVQCodeGenerator(
        4,
        (5, 7),
        DecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            rvq_predictor=RVQPredictor.CODEBOOK_AR,
        ),
    )
    condition = torch.randn(2, 3, 4)
    labels = torch.tensor(
        [
            [[1, 2], [3, 4], [-1, -1]],
            [[2, 6], [-1, -1], [-1, -1]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    def fail_scatter(*_args, **_kwargs):
        raise AssertionError("packed loss must not scatter logits")

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("decoder-owned packed values must not be revalidated")

    monkeypatch.setattr(rvq_module, "_scatter", fail_scatter)
    monkeypatch.setattr(
        "anytrain.loss.codebook._validate_packed_inputs",
        fail_validation,
    )
    output = generator.code_loss_from_condition(
        condition,
        mask,
        target_codes=labels,
        include_details=False,
    )

    assert torch.isfinite(output.loss)
    assert output.items["rvq"].details is None


def test_rvq_decoder_import_is_lazy() -> None:
    if _has_qwen3_builder():
        decoder = AcousticRVQDecoder(4, 2, (5, 7), hidden_dim=4, layers=1, heads=1, ffn_ratio=2)
        assert decoder.codebook_sizes == (5, 7)
        return

    with pytest.raises(ImportError, match="transformers"):
        AcousticRVQDecoder(4, 2, (5, 7), hidden_dim=4, layers=1, heads=1, ffn_ratio=2)


def _has_qwen3_builder() -> bool:
    if importlib.util.find_spec("transformers") is None:
        return False
    try:
        from anytrain.module.qwen import build_qwen3_model
    except ImportError:
        return False
    return callable(build_qwen3_model)


def _batch(
    semantic: torch.Tensor, acoustic: torch.Tensor, mask: torch.Tensor
) -> SemanticCodecBatch:
    return SemanticCodecBatch(
        semantic_codes=semantic,
        acoustic_codes=acoustic,
        mask=mask,
        semantic_pad_id=8,
        acoustic_pad_ids=(5, 7),
        acoustic_mask=mask,
    )


def test_rvq_mtp_decoder_import_is_lazy() -> None:
    if importlib.util.find_spec("transformers") is not None:
        decoder = QwenMTPCodebookPredictor(
            4,
            2,
            (5, 7),
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            mtp_layers=1,
            mtp_heads=1,
        )
        assert decoder.codebook_sizes == (5, 7)
        return

    with pytest.raises(ImportError, match="transformers"):
        QwenMTPCodebookPredictor(
            4,
            2,
            (5, 7),
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            mtp_layers=1,
            mtp_heads=1,
        )
