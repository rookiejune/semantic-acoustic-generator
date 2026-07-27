from __future__ import annotations

import importlib.util

import pytest
import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes, masked_acoustic_features
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from anytrain.loss import MaskedCodebookCrossEntropyLoss, MaskedCosineAlignmentLoss
from anytrain.module.qwen import QwenMTPCodebookPredictor

from semantic_acoustic_codec.config import DecoderConfig, Route, RVQPredictor
from semantic_acoustic_codec.loss import FlowLoss
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
    SemanticCodecRuntime,
    SemanticSupportConfig,
    build_support,
    load_artifact,
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

    def decode_features(self, semantic_codes: torch.Tensor, acoustic_features: torch.Tensor) -> torch.Tensor:
        return acoustic_features.new_zeros((semantic_codes.size(0), 1, semantic_codes.size(1) * 320))


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

    loss = FlowLoss()(decoder, condition, target, mask, ContinuousFlowRuntime())
    sample = decoder.sample(condition, mask=mask, steps=2)

    assert loss.loss.shape == (2,)
    assert sample.shape == target.shape
    assert torch.equal(sample[1, 2:], torch.zeros_like(sample[1, 2:]))


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

    item, representation = FlowLoss().forward_with_features(
        decoder,
        condition,
        target,
        mask,
        ContinuousFlowRuntime(),
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

    features = support.sample_features(semantic, mask=mask, generator=torch.Generator().manual_seed(0))
    condition = support.condition(
        semantic,
        mask=mask,
        reference_features=masked_acoustic_features(backend, reference, mask),
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
    output = modules.generator.loss(
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

    output = modules.generator.loss_from_condition(
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
    output = modules.generator.loss(
        _batch(semantic, acoustic, mask),
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
    output = generator.loss_from_condition(condition, mask, target_codes=acoustic)
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
    output = generator.loss(
        batch,
        condition,
        feature_mean=torch.zeros(1, 1, 1),
        feature_std=torch.ones(1, 1, 1),
    )
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


def _batch(semantic: torch.Tensor, acoustic: torch.Tensor, mask: torch.Tensor) -> SemanticCodecBatch:
    return SemanticCodecBatch(
        semantic_codes=semantic,
        acoustic_codes=acoustic,
        mask=mask,
        semantic_pad_id=8,
        acoustic_pad_ids=(5, 7),
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
