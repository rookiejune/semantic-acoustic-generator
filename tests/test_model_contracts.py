from __future__ import annotations

import importlib.util

import pytest
import torch

from semantic_acoustic_codec.backend import LongCatBackend
from semantic_acoustic_codec.config import AdapterType, DecoderConfig, Route, RVQPredictor
from semantic_acoustic_codec.data import SemanticCodecBatch
from semantic_acoustic_codec.loss import FlowLoss, RepaLoss, RVQLoss
from semantic_acoustic_codec.model import (
    AcousticRVQDecoder,
    AcousticRVQMTPDecoder,
    DiTDecoder,
    FMFeatureGenerator,
    RectifiedFlowRuntime,
    ReferenceConditioner,
    RVQCodeGenerator,
    SemanticConditioner,
    backend_features,
    build_route,
)
from semantic_acoustic_codec.runtime import (
    SemanticCodecRuntime,
    SemanticSupportConfig,
    build_support,
    load_artifact,
    save_artifact,
)


class FakeEncoder:
    input_sample_rate = 16_000
    hop_length = 320


class FakeDecoder:
    latent_dim = 4


class FakeCodec:
    sample_rate = 16_000
    encoder = FakeEncoder()
    decoders = {"default": FakeDecoder()}
    semantic_codebook = torch.randn(8, 6)
    codebook_sizes = (8, 5, 7)

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del sample_rate
        return audio.new_zeros((audio.size(0), 2, 3), dtype=torch.long)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.new_zeros((codes.size(0), 1, codes.size(1) * 320), dtype=torch.float32)

    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(acoustic_codes.float(), (0, 2))[:, :, :4]

    def decode_features(self, semantic_codes: torch.Tensor, acoustic_features: torch.Tensor) -> torch.Tensor:
        return acoustic_features.new_zeros((semantic_codes.size(0), 1, semantic_codes.size(1) * 320))


def test_conditioner_shapes() -> None:
    conditioner = SemanticConditioner(
        torch.randn(16, 8),
        condition_dim=12,
        adapter=AdapterType.LINEAR,
    )
    semantic = torch.tensor([[[1], [2], [3]], [[4], [5], [0]]], dtype=torch.long)
    condition = conditioner(semantic)

    assert condition.shape == (2, 3, 12)


def test_fm_loss_and_sample_shapes() -> None:
    condition = torch.randn(2, 4, 10)
    target = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    decoder = DiTDecoder(10, 6, layers=1, heads=2, ffn_ratio=2)

    loss = FlowLoss()(decoder, condition, target, mask, RectifiedFlowRuntime())
    sample = decoder.sample(condition, mask=mask, steps=2)

    assert loss.loss.shape == (2,)
    assert sample.shape == target.shape
    assert torch.equal(sample[1, 2:], torch.zeros_like(sample[1, 2:]))


def test_reference_conditioner_uses_explicit_and_default_reference() -> None:
    conditioner = ReferenceConditioner(feature_dim=4, condition_dim=6)
    features = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    explicit = conditioner(features, mask=mask, batch_size=2)
    default = conditioner(None, batch_size=2)

    assert explicit.shape == (2, 1, 6)
    assert default.shape == (2, 1, 6)
    assert "default_feature" in conditioner.state_dict()


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

    item, representation = FlowLoss().forward_with_features(
        decoder,
        condition,
        target,
        mask,
        RectifiedFlowRuntime(),
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

    item = RepaLoss()(representation, target, mask)
    item.loss.mean().backward()

    assert torch.isfinite(item.loss).all()
    assert representation.grad is not None
    assert target.grad is None
    assert torch.isfinite(representation.grad).all()
    assert torch.equal(representation.grad[:, 1], torch.zeros_like(representation.grad[:, 1]))


def test_build_route_and_backend_features() -> None:
    backend = LongCatBackend(FakeCodec())
    modules = build_route(
        Route.FM,
        backend.semantic_codebook,
        backend.acoustic_feature_dim,
        backend.acoustic_codebook_sizes,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    acoustic = torch.tensor([[[1, 2], [3, 4], [-1, -1]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    features = backend_features(backend, acoustic, mask)

    assert modules.route is Route.FM
    assert isinstance(modules.generator, FMFeatureGenerator)
    assert backend.sample_rate == 16_000
    assert backend.frame_rate == 50.0
    assert backend.acoustic_feature_dim == 4
    assert backend.acoustic_codebook_sizes == (5, 7)
    assert features.shape == (1, 3, 4)
    assert torch.equal(features[:, 2], torch.zeros_like(features[:, 2]))


def test_semantic_support_decodes_and_roundtrips_artifact(tmp_path) -> None:
    backend = LongCatBackend(FakeCodec())
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
    semantic = torch.tensor([[[1], [2], [-1]]], dtype=torch.long)
    reference = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])

    features = support.sample_features(semantic, mask=mask, generator=torch.Generator().manual_seed(0))
    condition = support.condition(
        semantic,
        mask=mask,
        reference_features=backend_features(backend, reference, mask),
        reference_mask=mask,
    )
    runtime = SemanticCodecRuntime(support, backend)
    waveform = runtime.decode(semantic, mask=mask, generator=torch.Generator().manual_seed(1))
    save_artifact(tmp_path, support, config)
    loaded = load_artifact(tmp_path)
    loaded_features = loaded.sample_features(
        semantic,
        mask=mask,
        generator=torch.Generator().manual_seed(0),
    )

    assert features.shape == (1, 3, 4)
    assert condition.shape == (1, 3, 10)
    assert waveform.shape == (1, 1, 3 * 320)
    assert not hasattr(support, "backend")
    assert torch.equal(features[:, 2], torch.zeros_like(features[:, 2]))
    assert torch.allclose(features, loaded_features)


def test_fm_route_trains_one_step() -> None:
    backend = LongCatBackend(FakeCodec())
    semantic = torch.tensor([[[1], [2], [0]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    target = backend_features(backend, acoustic, mask)

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
    output = modules.generator.loss(
        SemanticCodecBatch(semantic_codes=semantic, acoustic_codes=acoustic, mask=mask),
        condition,
        target,
        feature_mean=torch.zeros(1, 1, backend.acoustic_feature_dim),
        feature_std=torch.ones(1, 1, backend.acoustic_feature_dim),
    )
    loss = output.loss
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_rvq_route_trains_one_step_when_qwen3_builder_is_available() -> None:
    if not _has_qwen3_builder():
        pytest.skip("RVQ route requires transformers and anytrain.module.qwen.build_qwen3_model.")

    backend = LongCatBackend(FakeCodec())
    semantic = torch.tensor([[[1], [2], [0]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
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
    target = backend_features(backend, acoustic, mask)
    reference = modules.reference_conditioner(target, mask=mask, batch_size=1)
    condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
    output = modules.generator.loss(
        SemanticCodecBatch(semantic_codes=semantic, acoustic_codes=acoustic, mask=mask),
        condition,
        feature_mean=torch.zeros(1, 1, backend.acoustic_feature_dim),
        feature_std=torch.ones(1, 1, backend.acoustic_feature_dim),
    )
    loss = output.loss
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_rvq_mtp_route_trains_and_generates_when_transformers_is_available() -> None:
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("RVQ MTP route requires transformers.")

    backend = LongCatBackend(FakeCodec())
    semantic = torch.tensor([[[1], [2], [0]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
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
    target = backend_features(backend, acoustic, mask)
    reference = modules.reference_conditioner(target, mask=mask, batch_size=1)
    condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
    generator = modules.generator
    assert isinstance(generator, RVQCodeGenerator)
    logits = generator.core(condition, acoustic, mask=mask)
    loss = RVQLoss()(logits, acoustic, mask).loss.mean()
    loss.backward()
    generated = generator.sample_acoustic_codes(
        condition,
        mask,
        temperature=1.0,
        top_p=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert isinstance(generator.core, AcousticRVQMTPDecoder)
    assert torch.isfinite(loss)
    assert [value.shape for value in logits] == [(1, 3, 5), (1, 3, 7)]
    assert generated.shape == (1, 3, 2)
    assert torch.equal(generated[:, 2], torch.zeros_like(generated[:, 2]))
    assert bool((generated[..., 0][mask] < 5).all())
    assert bool((generated[..., 1][mask] < 7).all())


def test_rvq_loss_validates_per_codebook_logits() -> None:
    logits = (torch.randn(1, 2, 5), torch.randn(1, 2, 7))
    labels = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long)
    mask = torch.tensor([[True, False]])

    item = RVQLoss()(logits, labels, mask)

    assert item.loss.shape == (1,)
    assert item.details is not None
    assert set(item.details) == {"codebook_0", "codebook_1", "frames"}


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


def test_rvq_mtp_decoder_import_is_lazy() -> None:
    if importlib.util.find_spec("transformers") is not None:
        decoder = AcousticRVQMTPDecoder(
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
        AcousticRVQMTPDecoder(
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
