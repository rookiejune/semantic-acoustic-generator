from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from anytrain.codec import (
    AcousticLayout,
    SemanticAcousticCodes,
    semantic_acoustic_spec,
)
from torch import nn

from semantic_acoustic_generator.backend import (
    LongCatCodebookAdapter,
    LongCatFirstCodebookAdapter,
    adapt_backend,
)
from semantic_acoustic_generator.config import (
    AnchorTarget,
    DecoderConfig,
    FactorPredictor,
    FeatureAdapter,
    FMMode,
    Route,
)
from semantic_acoustic_generator.evaluation import evaluate_first_codebook_oracle
from semantic_acoustic_generator.model import FMFeatureGenerator
from semantic_acoustic_generator.model import rvq as rvq_module
from semantic_acoustic_generator.pl_module import build_module
from semantic_acoustic_generator.runtime import GeneratorConfig, GeneratorRuntime, build_support
from semantic_acoustic_generator.runtime.artifact import (
    load_artifact,
    load_generator_artifact,
    save_artifact,
)
from semantic_acoustic_generator.types import GeneratorBatch


class _Stage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codebook_size_a = 90
        self.codebook_size_b = 90
        self.codebook_a = nn.Embedding(90, 8)
        self.codebook_b = nn.Embedding(90, 8)
        self.out_proj_a = nn.Conv1d(8, 512, kernel_size=1, bias=True)
        self.out_proj_b = nn.Conv1d(8, 512, kernel_size=1, bias=True)
        self.retarget_code = 0
        with torch.no_grad():
            self.codebook_a.weight.copy_(torch.arange(90 * 8).view(90, 8) / 100)
            self.codebook_b.weight.copy_(torch.arange(90 * 8).view(90, 8) / 50 + 20)
            self.out_proj_a.bias.fill_(0.1)
            self.out_proj_b.bias.fill_(-0.2)

    def forward(self, residual: torch.Tensor):
        indices = torch.full(
            residual.shape[:1] + residual.shape[2:],
            self.retarget_code,
            dtype=torch.long,
            device=residual.device,
        )
        zero = residual.new_zeros(residual.size(0))
        return torch.zeros_like(residual), zero, zero, indices, residual


class _DepthCore(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(1, hidden_dim)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del return_dict
        hidden = (
            inputs_embeds.cumsum(dim=1)
            if past_key_values is None
            else inputs_embeds + past_key_values[:, -1:]
        )
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=hidden[:, -1:] if use_cache else None,
        )


class _Quantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList([_Stage(), _Stage(), _Stage()])

    def from_codes(self, codes: torch.Tensor):
        projected: torch.Tensor | None = None
        features = []
        for index in range(codes.size(1)):
            stage = self.quantizers[index]
            composite = codes[:, index]
            codes_a = torch.div(composite, stage.codebook_size_b, rounding_mode="floor")
            codes_b = composite.remainder(stage.codebook_size_b)
            features_a = stage.codebook_a(codes_a).transpose(1, 2)
            features_b = stage.codebook_b(codes_b).transpose(1, 2)
            value = torch.cat(
                (stage.out_proj_a(features_a), stage.out_proj_b(features_b)),
                dim=1,
            )
            projected = value if projected is None else projected + value
            features.extend((features_a, features_b))
        assert projected is not None
        return projected, torch.cat(features, dim=1), codes


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.acoustic_quantizer = _Quantizer()


class _LongCat(nn.Module):
    name = "longcat"
    sample_rate = 16_000
    frame_rate = 16.6667
    semantic_frame_rate = 16.6667
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (8100, 8100, 8100)
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None
    acoustic_feature_dim = 1024

    def __init__(self) -> None:
        super().__init__()
        self.model = _Decoder()
        self.semantic_embedding = nn.Embedding(8, 4)
        self.decoded_features: torch.Tensor | None = None

    @property
    def semantic_codebook(self) -> torch.Tensor:
        return self.semantic_embedding.weight

    def _decoder(self) -> _Decoder:
        return self.model

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        semantic = torch.zeros(audio.size(0), 2, 1, dtype=torch.long, device=audio.device)
        acoustic = torch.zeros(audio.size(0), 2, 3, dtype=torch.long, device=audio.device)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    def detokenize(self, codes: SemanticAcousticCodes) -> torch.Tensor:
        features = torch.zeros(
            codes.semantic.size(0),
            codes.semantic.size(1),
            self.acoustic_feature_dim,
            device=codes.semantic.device,
        )
        return self.decode_features(codes.semantic, features)

    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        projected, _, _ = self.model.acoustic_quantizer.from_codes(
            acoustic_codes.transpose(1, 2).contiguous()
        )
        return projected.transpose(1, 2).contiguous()

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        del semantic_codes
        self.decoded_features = acoustic_features.detach().clone()
        return acoustic_features.transpose(1, 2).contiguous()


def test_longcat_first_codebook_adapter_exposes_quantized_embeddings() -> None:
    backend = _LongCat()
    adapted = LongCatFirstCodebookAdapter(backend)
    codes = torch.tensor(
        [[[1, 20, 30], [8099, 40, 50]]],
        dtype=torch.long,
    )

    features = adapted.acoustic_codes_to_features(codes)

    stage = backend._decoder().acoustic_quantizer.quantizers[0]
    expected = torch.stack(
        (
            torch.cat((stage.codebook_a.weight[0], stage.codebook_b.weight[1])),
            torch.cat((stage.codebook_a.weight[89], stage.codebook_b.weight[89])),
        )
    )[None]
    assert adapted.acoustic_feature_dim == 16
    torch.testing.assert_close(features, expected)


def test_longcat_first_codebook_adapter_projects_features_before_decode() -> None:
    backend = _LongCat()
    adapted = LongCatFirstCodebookAdapter(backend)
    semantic = torch.tensor([[[1], [2]]], dtype=torch.long)
    codes = torch.tensor([[[90, 0, 0], [180, 0, 0]]], dtype=torch.long)
    features = adapted.acoustic_codes_to_features(codes)

    waveform = adapted.decode_features(semantic, features)

    stage = backend._decoder().acoustic_quantizer.quantizers[0]
    expected = torch.cat(
        (
            stage.out_proj_a(features[..., :8].transpose(1, 2)),
            stage.out_proj_b(features[..., 8:].transpose(1, 2)),
        ),
        dim=1,
    ).transpose(1, 2)
    assert backend.decoded_features is not None
    torch.testing.assert_close(backend.decoded_features, expected)
    torch.testing.assert_close(waveform, expected.transpose(1, 2))


def test_longcat_factor_adapter_sums_multiple_stage_projections() -> None:
    backend = _LongCat()
    adapted = LongCatCodebookAdapter(backend, codebooks=3)
    semantic = torch.tensor([[[1], [2]]], dtype=torch.long)
    codes = torch.tensor(
        [[[90, 180, 270], [360, 450, 540]]],
        dtype=torch.long,
    )

    features = adapted.acoustic_codes_to_features(codes)
    waveform = adapted.decode_features(semantic, features)
    native = adapted.native_features(codes)

    assert adapted.feature_codebooks == 3
    assert adapted.acoustic_feature_dim == 48
    assert len(adapted.factor_codebooks) == 6
    assert adapted.factor_codes(codes).shape == (1, 2, 6)
    assert backend.decoded_features is not None
    torch.testing.assert_close(backend.decoded_features, native)
    torch.testing.assert_close(waveform, native.transpose(1, 2))


def test_longcat_factor_adapter_skips_inactive_projection_bias() -> None:
    backend = _LongCat()
    adapted = LongCatCodebookAdapter(backend, codebooks=3)
    codes = torch.tensor([[[90, 180, 270], [360, 450, 540]]], dtype=torch.long)
    features = adapted.acoustic_codes_to_features(codes)

    stage0 = adapted.project_features(features, active_codebooks=1)
    first = backend._decoder().acoustic_quantizer.quantizers[0]
    expected = torch.cat(
        (
            first.out_proj_a(features[..., :8].transpose(1, 2)),
            first.out_proj_b(features[..., 8:16].transpose(1, 2)),
        ),
        dim=1,
    ).transpose(1, 2)
    zeroed = features.clone()
    zeroed[..., 16:] = 0
    zero_embedding_projection = adapted.project_features(zeroed)

    torch.testing.assert_close(stage0, expected)
    assert not torch.allclose(stage0, zero_embedding_projection)


def test_longcat_factor_adapter_retargets_later_residual_stages() -> None:
    backend = _LongCat()
    adapted = LongCatCodebookAdapter(backend, codebooks=3)
    stages = backend._decoder().acoustic_quantizer.quantizers
    stages[1].retarget_code = 2 * 90 + 3
    stages[2].retarget_code = 4 * 90 + 5
    acoustic = torch.tensor([[[90, 180, 270], [360, 450, 540]]], dtype=torch.long)
    predicted = torch.tensor(
        [[[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]],
        dtype=torch.long,
    )

    retargeted = adapted.retarget_factor_codes(acoustic, predicted)
    features = adapted.factor_codes_to_features(retargeted)

    assert torch.equal(retargeted[..., :2], predicted[..., :2])
    assert torch.equal(retargeted[..., 2:4], torch.tensor([2, 3]).expand(1, 2, 2))
    assert torch.equal(retargeted[..., 4:6], torch.tensor([4, 5]).expand(1, 2, 2))
    assert features.shape == (1, 2, 48)


def test_longcat_first_codebook_adapter_snaps_each_factor_by_cosine() -> None:
    backend = _LongCat()
    adapted = LongCatFirstCodebookAdapter(backend)
    stage = backend._decoder().acoustic_quantizer.quantizers[0]
    with torch.no_grad():
        stage.codebook_a.weight.normal_()
        stage.codebook_b.weight.normal_()
    codes = torch.tensor([[[1, 0, 0], [8099, 0, 0]]], dtype=torch.long)
    features = adapted.acoustic_codes_to_features(codes)
    perturbed = features + torch.randn_like(features) * 1e-4

    snapped = adapted.snap_features(perturbed)

    torch.testing.assert_close(snapped, features)
    assert torch.equal(
        adapted.factor_codes(codes),
        torch.tensor([[[0, 1], [89, 89]]]),
    )


def test_feature_adapter_is_default_off_and_idempotent() -> None:
    backend = _LongCat()

    assert adapt_backend(backend, FeatureAdapter.NONE) is backend
    adapted = adapt_backend(backend, FeatureAdapter.LONGCAT_FIRST_CODEBOOK)
    assert isinstance(adapted, LongCatFirstCodebookAdapter)
    assert adapt_backend(adapted, FeatureAdapter.LONGCAT_FIRST_CODEBOOK) is adapted


def test_longcat_factor_targets_ignore_padded_code_ids() -> None:
    backend = _LongCat()
    adapted = LongCatFirstCodebookAdapter(backend)
    batch = GeneratorBatch(
        semantic_codes=torch.tensor([[[1], [2]], [[3], [8]]]),
        acoustic_codes=torch.tensor(
            [
                [[1, 0, 0], [8099, 0, 0]],
                [[90, 0, 0], [8100, 8100, 8100]],
            ]
        ),
        mask=torch.tensor([[True, True], [True, False]]),
        semantic_pad_id=8,
        acoustic_pad_ids=(8100, 8100, 8100),
        acoustic_mask=torch.tensor([[True, True], [True, False]]),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        feature_adapter=FeatureAdapter.LONGCAT_FIRST_CODEBOOK,
        decoder=DecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            fm_mode=FMMode.ANCHOR,
            anchor_target=AnchorTarget.FACTOR,
            anchor_hidden_dim=4,
            anchor_layers=1,
        ),
    )
    module = build_module(adapted, config, batch, normalize_features=True)

    targets = module._factor_targets(batch)
    generator = module.support.generator
    assert isinstance(generator, FMFeatureGenerator)
    assert generator.anchor is not None
    with torch.no_grad():
        generator.anchor.output.weight.zero_()
        generator.anchor.output.bias.zero_()
        generator.anchor.output.bias[1] = 5
        generator.anchor.output.bias[90] = 5
    metrics = module.validation_step(batch, 0)

    assert targets is not None
    assert torch.equal(
        targets,
        torch.tensor(
            [
                [[0, 1], [89, 89]],
                [[1, 0], [0, 0]],
            ]
        ),
    )
    assert set(metrics) == {"val/without_reference_factor_code_error"}
    assert float(metrics["val/without_reference_factor_code_error"]) == pytest.approx(2 / 3)


def test_first_codebook_oracle_compares_native_raw_and_snap_paths() -> None:
    backend = _LongCat()
    adapted = LongCatFirstCodebookAdapter(backend)
    semantic = torch.tensor([[[1], [2]]], dtype=torch.long)
    acoustic = torch.tensor([[[1, 2, 3], [8099, 4, 5]]], dtype=torch.long)
    mask = torch.ones(1, 2, dtype=torch.bool)
    batch = GeneratorBatch(
        semantic_codes=semantic,
        acoustic_codes=acoustic,
        mask=mask,
        semantic_pad_id=8,
        acoustic_pad_ids=(8100, 8100, 8100),
        acoustic_mask=mask,
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )

    result = evaluate_first_codebook_oracle(adapted, batch, sigmas=(0.1,), seed=3)

    assert set(result.audio) == {
        "full_reconstruction",
        "stage0_code_reconstruction",
        "exact_16d_reconstruction",
        "raw_sigma_0p1",
        "snap_sigma_0p1",
    }
    assert result.metrics["native_projection_max_abs"] == 0.0
    assert set(result.metrics["groups"]) == {"raw_sigma_0p1", "snap_sigma_0p1"}


def test_artifact_restores_longcat_feature_adapter_for_raw_backend(tmp_path) -> None:
    backend = _LongCat()
    adapted = adapt_backend(backend, FeatureAdapter.LONGCAT_FIRST_CODEBOOK)
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        feature_adapter=FeatureAdapter.LONGCAT_FIRST_CODEBOOK,
        decoder=DecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            fm_mode=FMMode.ANCHOR,
            anchor_hidden_dim=4,
            anchor_layers=1,
        ),
    )
    support = build_support(
        config,
        semantic_codebook=adapted.semantic_codebook,
        codec_spec=semantic_acoustic_spec(adapted),
    )
    save_artifact(tmp_path, support, backend=adapted)

    loaded = load_artifact(tmp_path)
    runtime = GeneratorRuntime(loaded, backend)
    generator = load_generator_artifact(tmp_path)

    assert loaded.acoustic_feature_dim == 16
    assert loaded.config is not None
    assert loaded.config.feature_adapter is FeatureAdapter.LONGCAT_FIRST_CODEBOOK
    assert isinstance(runtime.backend, LongCatFirstCodebookAdapter)
    assert generator.spec.feature_adapter is FeatureAdapter.LONGCAT_FIRST_CODEBOOK
    assert generator.spec.decoder.fm_mode is FMMode.ANCHOR
    generator.spec.validate_backend(backend)


def test_factor_artifact_restores_codebook_buffers_without_codec_weights(tmp_path) -> None:
    backend = _LongCat()
    adapted = LongCatFirstCodebookAdapter(backend)
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        feature_adapter=FeatureAdapter.LONGCAT_FIRST_CODEBOOK,
        decoder=DecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            fm_mode=FMMode.ANCHOR,
            anchor_target=AnchorTarget.FACTOR,
            anchor_hidden_dim=4,
            anchor_layers=1,
        ),
    )
    support = build_support(
        config,
        semantic_codebook=adapted.semantic_codebook,
        codec_spec=semantic_acoustic_spec(adapted),
        factor_codebooks=adapted.factor_codebooks,
    )
    save_artifact(tmp_path, support, backend=adapted)

    loaded = load_artifact(tmp_path)
    acoustic = load_generator_artifact(tmp_path)

    assert loaded.config is not None
    assert loaded.config.decoder.anchor_target is AnchorTarget.FACTOR
    for restored in (loaded.generator, acoustic.generator):
        assert isinstance(restored, FMFeatureGenerator)
        assert restored.factor_codebook_a is not None
        assert restored.factor_codebook_b is not None
        torch.testing.assert_close(restored.factor_codebook_a, adapted.factor_codebooks[0])
        torch.testing.assert_close(restored.factor_codebook_b, adapted.factor_codebooks[1])


def test_multi_codebook_factor_artifact_roundtrip(tmp_path) -> None:
    backend = _LongCat()
    adapted = LongCatCodebookAdapter(backend, codebooks=3)
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        feature_adapter=FeatureAdapter.LONGCAT_CODEBOOKS,
        feature_codebooks=3,
        decoder=DecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            fm_mode=FMMode.ANCHOR,
            anchor_target=AnchorTarget.FACTOR,
            anchor_hidden_dim=4,
            anchor_layers=1,
        ),
    )
    support = build_support(
        config,
        semantic_codebook=adapted.semantic_codebook,
        codec_spec=semantic_acoustic_spec(adapted),
        factor_codebooks=adapted.factor_codebooks,
    )
    save_artifact(tmp_path, support, backend=backend)

    loaded = load_artifact(tmp_path)
    runtime = GeneratorRuntime(loaded, backend)
    acoustic = load_generator_artifact(tmp_path)

    assert loaded.config is not None
    assert loaded.config.feature_codebooks == 3
    assert isinstance(runtime.backend, LongCatCodebookAdapter)
    assert runtime.backend.feature_codebooks == 3
    assert acoustic.spec.feature_codebooks == 3
    assert "factor_codebook_2_b" in loaded.generator.state_dict()
    for index, codebook in enumerate(adapted.factor_codebooks):
        key = (
            ("factor_codebook_a", "factor_codebook_b")[index]
            if index < 2
            else f"factor_codebook_{index // 2}_{'a' if index % 2 == 0 else 'b'}"
        )
        torch.testing.assert_close(loaded.generator.state_dict()[key], codebook)


def test_depth_factor_artifact_roundtrip(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rvq_module,
        "_qwen3_model",
        lambda **options: _DepthCore(options["hidden_dim"]),
    )
    backend = _LongCat()
    adapted = LongCatCodebookAdapter(backend, codebooks=2)
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=4,
        feature_adapter=FeatureAdapter.LONGCAT_CODEBOOKS,
        feature_codebooks=2,
        decoder=DecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            fm_mode=FMMode.ANCHOR,
            anchor_target=AnchorTarget.FACTOR,
            factor_predictor=FactorPredictor.DEPTH_AR,
            anchor_hidden_dim=4,
            anchor_layers=1,
        ),
    )
    support = build_support(
        config,
        semantic_codebook=adapted.semantic_codebook,
        codec_spec=semantic_acoustic_spec(adapted),
        factor_codebooks=adapted.factor_codebooks,
    )
    condition = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    expected = support.generator.sample_factor_codes(condition, mask)
    save_artifact(tmp_path, support, backend=backend)

    loaded = load_artifact(tmp_path)
    assert loaded.config is not None
    assert loaded.config.decoder.factor_predictor is FactorPredictor.DEPTH_AR
    assert isinstance(loaded.generator, FMFeatureGenerator)
    assert loaded.generator.factor_depth is not None
    assert "factor_depth.factor_codebook_1_b" in loaded.generator.state_dict()
    torch.testing.assert_close(
        loaded.generator.sample_factor_codes(condition, mask),
        expected,
    )
