from __future__ import annotations

import pytest
import torch
from torch import Tensor

pytest.importorskip("lightning")

try:
    from semantic_acoustic_codec.backend import LongCatBackend
    from semantic_acoustic_codec.config import DecoderConfig, Route
    from semantic_acoustic_codec.datamodule import collate_codes
    from semantic_acoustic_codec.pl_module import build_module, feature_stats
    from semantic_acoustic_codec.runtime import SemanticSupportConfig, load_artifact
except TypeError as exc:
    if "SPEAKER_ID" not in str(exc):
        raise
    pytest.skip(
        "anydataset TextMeta currently defines SPEAKER_ID twice; training tests require that third_party fix.",
        allow_module_level=True,
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

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        return acoustic_features.new_zeros((semantic_codes.size(0), 1, semantic_codes.size(1) * 320))


class FakeRepaTeacher:
    feature_dim = 3

    def __call__(
        self,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if semantic_codes.shape[:2] != acoustic_codes.shape[:2] or mask.shape != semantic_codes.shape[:2]:
            raise ValueError("fake teacher inputs must align")
        return torch.ones(mask.shape + (self.feature_dim,), device=semantic_codes.device)


def test_training_module_trains_and_exports_artifact(tmp_path) -> None:
    backend = LongCatBackend(FakeCodec())
    batch = collate_codes([torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)])
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)

    output = module.training_step(batch, 0)
    loss = output["loss"]
    loss.backward()
    module.export_artifact(tmp_path)
    loaded = load_artifact(tmp_path, backend=backend)

    assert torch.isfinite(loss)
    assert module.config.feature_mean is not None
    assert module.support.reference_conditioner.gate.grad is not None
    assert loaded.feature_mean.shape == (1, 1, backend.acoustic_feature_dim)
    assert loaded.sample_features(batch.semantic_codes, mask=batch.mask).shape == (
        1,
        2,
        backend.acoustic_feature_dim,
    )


def test_training_module_adds_repa_loss_with_teacher() -> None:
    backend = LongCatBackend(FakeCodec())
    batch = collate_codes([torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)])
    config = SemanticSupportConfig(
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
    assert "repa" in output


def test_training_module_requires_repa_teacher() -> None:
    backend = LongCatBackend(FakeCodec())
    batch = collate_codes([torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long)])
    config = SemanticSupportConfig(
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
    backend = LongCatBackend(FakeCodec())
    batch = collate_codes(
        [
            torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.long),
            torch.tensor([[5, 3, 4]], dtype=torch.long),
        ]
    )

    mean, std = feature_stats(backend, batch)

    assert len(mean) == backend.acoustic_feature_dim
    assert len(std) == backend.acoustic_feature_dim
    assert all(value > 0 for value in std)
