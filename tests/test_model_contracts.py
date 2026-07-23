from __future__ import annotations

import importlib.util

import pytest
import torch

from semantic_acoustic_codec.config import AdapterType, DecoderConfig, Route
from semantic_acoustic_codec.loss import FlowLoss, RepaLoss, RVQLoss
from semantic_acoustic_codec.model import (
    AcousticRVQDecoder,
    DiTDecoder,
    RectifiedFlowRuntime,
    ReferenceConditioner,
    SemanticConditioner,
    build_route,
    teacher_features,
)
from semantic_acoustic_codec.runtime import (
    SemanticCodecConfig,
    build_codec,
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


def test_build_route_and_teacher_features() -> None:
    from semantic_acoustic_codec.teacher import LongCatTeacher

    teacher = LongCatTeacher(FakeCodec())
    modules = build_route(
        Route.FM,
        teacher,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    acoustic = torch.tensor([[[1, 2], [3, 4], [-1, -1]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    features = teacher_features(teacher, acoustic, mask)

    assert modules.route is Route.FM
    assert teacher.sample_rate == 16_000
    assert teacher.frame_rate == 50.0
    assert teacher.acoustic_feature_dim == 4
    assert teacher.acoustic_codebook_sizes == (5, 7)
    assert features.shape == (1, 3, 4)
    assert torch.equal(features[:, 2], torch.zeros_like(features[:, 2]))


def test_semantic_codec_decodes_and_roundtrips_artifact(tmp_path) -> None:
    from semantic_acoustic_codec.teacher import LongCatTeacher

    teacher = LongCatTeacher(FakeCodec())
    config = SemanticCodecConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    codec = build_codec(teacher, config).eval()
    semantic = torch.tensor([[[1], [2], [-1]]], dtype=torch.long)
    reference = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])

    features = codec.sample_features(semantic, mask=mask, generator=torch.Generator().manual_seed(0))
    reference_features = codec.sample_features(
        semantic,
        mask=mask,
        reference_acoustic_codes=reference,
        reference_mask=mask,
        generator=torch.Generator().manual_seed(0),
    )
    waveform = codec.decode(
        semantic,
        mask=mask,
        reference_acoustic_codes=reference,
        reference_mask=mask,
        generator=torch.Generator().manual_seed(1),
    )
    save_artifact(tmp_path, codec, config)
    loaded = load_artifact(tmp_path, teacher=teacher)
    loaded_features = loaded.sample_features(
        semantic,
        mask=mask,
        generator=torch.Generator().manual_seed(0),
    )

    assert features.shape == (1, 3, 4)
    assert reference_features.shape == (1, 3, 4)
    assert waveform.shape == (1, 1, 3 * 320)
    assert torch.equal(features[:, 2], torch.zeros_like(features[:, 2]))
    assert torch.allclose(features, loaded_features)


def test_fm_route_trains_one_step() -> None:
    from semantic_acoustic_codec.teacher import LongCatTeacher

    teacher = LongCatTeacher(FakeCodec())
    semantic = torch.tensor([[[1], [2], [0]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    target = teacher_features(teacher, acoustic, mask)

    modules = build_route(
        Route.FM,
        teacher,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    optimizer = torch.optim.AdamW(
        list(modules.conditioner.parameters())
        + list(modules.reference_conditioner.parameters())
        + list(modules.decoder.parameters()),
        lr=1e-3,
    )
    reference = modules.reference_conditioner(target, mask=mask, batch_size=1)
    condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
    item = FlowLoss()(modules.decoder, condition, target, mask, RectifiedFlowRuntime())
    loss = item.loss.mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_rvq_route_trains_one_step_when_transformers_is_available() -> None:
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("RVQ route requires transformers.")

    from semantic_acoustic_codec.teacher import LongCatTeacher

    teacher = LongCatTeacher(FakeCodec())
    semantic = torch.tensor([[[1], [2], [0]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    modules = build_route(
        Route.RVQ,
        teacher,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    optimizer = torch.optim.AdamW(
        list(modules.conditioner.parameters())
        + list(modules.reference_conditioner.parameters())
        + list(modules.decoder.parameters()),
        lr=1e-3,
    )
    target = teacher_features(teacher, acoustic, mask)
    reference = modules.reference_conditioner(target, mask=mask, batch_size=1)
    condition = (modules.conditioner(semantic) + reference).masked_fill(~mask[..., None], 0)
    logits = modules.decoder(condition, acoustic, mask=mask)
    loss = RVQLoss()(logits, acoustic, mask).loss.mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_rvq_loss_validates_per_codebook_logits() -> None:
    logits = (torch.randn(1, 2, 5), torch.randn(1, 2, 7))
    labels = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long)
    mask = torch.tensor([[True, False]])

    item = RVQLoss()(logits, labels, mask)

    assert item.loss.shape == (1,)
    assert item.details is not None
    assert set(item.details) == {"codebook_0", "codebook_1", "frames"}


def test_rvq_decoder_import_is_lazy() -> None:
    if importlib.util.find_spec("transformers") is not None:
        decoder = AcousticRVQDecoder(4, 2, (5, 7), hidden_dim=4, layers=1, heads=1, ffn_ratio=2)
        assert decoder.codebook_sizes == (5, 7)
        return

    with pytest.raises(ImportError, match="transformers"):
        AcousticRVQDecoder(4, 2, (5, 7), hidden_dim=4, layers=1, heads=1, ffn_ratio=2)

