from __future__ import annotations

import pytest
import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes

from semantic_acoustic_generator.config import DecoderConfig, Route
from semantic_acoustic_generator.datamodule import collate_structured_codes
from semantic_acoustic_generator.pl_module import build_module
from semantic_acoustic_generator.runtime import GeneratorConfig
from semantic_acoustic_generator.types import GeneratorBatch, PairMetadata


class FixedBackend:
    name = "fixed-test"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    acoustic_layout = AcousticLayout.FIXED_LENGTH
    acoustic_unit_length = 3
    semantic_codebook = torch.randn(8, 6)
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (5,)
    acoustic_feature_dim = 4

    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        return acoustic_codes.float().expand(-1, -1, self.acoustic_feature_dim).contiguous()

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        if bool((semantic_codes >= self.semantic_codebook.size(0)).any()):
            raise ValueError("semantic codes must be valid codec ids")
        return acoustic_features.new_zeros(
            (semantic_codes.size(0), 1, acoustic_features.size(1) * 8)
        )

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        semantic = audio.new_zeros((audio.size(0), 2, 1), dtype=torch.long)
        acoustic = audio.new_zeros((audio.size(0), self.acoustic_unit_length, 1), dtype=torch.long)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    def detokenize(self, codes: SemanticAcousticCodes) -> torch.Tensor:
        return self.decode_features(codes.semantic, self.acoustic_codes_to_features(codes.acoustic))


class RecordingFixedBackend(FixedBackend):
    def __init__(self) -> None:
        self.detokenized: list[SemanticAcousticCodes] = []

    def detokenize(self, codes: SemanticAcousticCodes) -> torch.Tensor:
        self.detokenized.append(
            SemanticAcousticCodes(
                semantic=codes.semantic.detach().clone(),
                acoustic=codes.acoustic.detach().clone(),
            )
        )
        value = codes.semantic.float().sum(dim=(1, 2)) * 100
        value = value + codes.acoustic.float().sum(dim=(1, 2))
        return value[:, None, None].expand(-1, 1, self.acoustic_unit_length * 8).clone()


def _batch() -> GeneratorBatch:
    values = [
        SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2], [3]], dtype=torch.long),
            acoustic=torch.tensor([[1], [2], [3]], dtype=torch.long),
        ),
        SemanticAcousticCodes(
            semantic=torch.tensor([[4], [5]], dtype=torch.long),
            acoustic=torch.tensor([[2], [1], [0]], dtype=torch.long),
        ),
    ]
    return collate_structured_codes(
        values,
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
    )


def _paired_batch() -> GeneratorBatch:
    target = collate_structured_codes(
        [
            SemanticAcousticCodes(
                semantic=torch.tensor([[1], [2], [3]], dtype=torch.long),
                acoustic=torch.tensor([[1], [2], [3]], dtype=torch.long),
            )
        ],
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
    )
    reference = collate_structured_codes(
        [
            SemanticAcousticCodes(
                semantic=torch.tensor([[6], [7]], dtype=torch.long),
                acoustic=torch.tensor([[4], [0], [1]], dtype=torch.long),
            )
        ],
        semantic_pad_id=8,
        acoustic_pad_ids=(5,),
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
    )
    metadata = PairMetadata(
        target_index=0,
        reference_index=1,
        target_text_index=0,
        reference_text_index=1,
        target_source_index=0,
        reference_source_index=1,
        target_role="target",
        reference_role="target",
        target_utterance_id="target",
        reference_utterance_id="reference",
        target_speaker_id="speaker",
        reference_speaker_id="speaker",
        target_text="target text",
        reference_text="reference text",
    )
    return GeneratorBatch(
        semantic_codes=target.semantic_codes,
        acoustic_codes=target.acoustic_codes,
        mask=target.mask,
        semantic_pad_id=target.semantic_pad_id,
        acoustic_pad_ids=target.acoustic_pad_ids,
        acoustic_mask=target.acoustic_mask,
        acoustic_layout=target.acoustic_layout,
        reference_semantic_codes=reference.semantic_codes,
        reference_acoustic_codes=reference.acoustic_codes,
        reference_mask=reference.mask,
        reference_acoustic_mask=reference.acoustic_mask,
        metadata=(metadata,),
    )


def test_fixed_layout_backend_is_rejected_before_training() -> None:
    backend = FixedBackend()
    config = GeneratorConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )

    with pytest.raises(ValueError, match="frame-aligned"):
        build_module(backend, config, normalize_features=False)


def test_fixed_layout_collation_is_rejected() -> None:
    with pytest.raises(ValueError, match="frame-aligned"):
        _batch()


def test_fixed_layout_paired_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="frame-aligned"):
        _paired_batch()
