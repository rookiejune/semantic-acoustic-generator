from __future__ import annotations

import torch
from anytrain.codec import (
    AcousticLayout,
    SemanticAcousticCodes,
    semantic_acoustic_spec,
)
from torch import nn

from semantic_acoustic_generator.backend import (
    LongCatFirstCodebookAdapter,
    adapt_backend,
)
from semantic_acoustic_generator.config import DecoderConfig, FeatureAdapter, FMMode, Route
from semantic_acoustic_generator.evaluation import evaluate_first_codebook_oracle
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
        self.out_proj_a = nn.Conv1d(8, 512, kernel_size=1, bias=False)
        self.out_proj_b = nn.Conv1d(8, 512, kernel_size=1, bias=False)
        with torch.no_grad():
            self.codebook_a.weight.copy_(torch.arange(90 * 8).view(90, 8) / 100)
            self.codebook_b.weight.copy_(torch.arange(90 * 8).view(90, 8) / 50 + 20)


class _Quantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList([_Stage()])

    def from_codes(self, codes: torch.Tensor):
        stage = self.quantizers[0]
        composite = codes[:, 0]
        codes_a = torch.div(composite, stage.codebook_size_b, rounding_mode="floor")
        codes_b = composite.remainder(stage.codebook_size_b)
        features_a = stage.codebook_a(codes_a).transpose(1, 2)
        features_b = stage.codebook_b(codes_b).transpose(1, 2)
        projected = torch.cat(
            (stage.out_proj_a(features_a), stage.out_proj_b(features_b)),
            dim=1,
        )
        return projected, torch.cat((features_a, features_b), dim=1), codes


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
        return torch.zeros(
            *acoustic_codes.shape[:2],
            self.acoustic_feature_dim,
            device=acoustic_codes.device,
        )

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
