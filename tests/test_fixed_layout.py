from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes

from semantic_acoustic_codec.callback import SampleLogConfig, SampleLogger
from semantic_acoustic_codec.config import DecoderConfig, Route
from semantic_acoustic_codec.datamodule import collate_structured_codes
from semantic_acoustic_codec.pl_module import build_module
from semantic_acoustic_codec.runtime import (
    SemanticCodecRuntime,
    SemanticSupportConfig,
    load_artifact,
)
from semantic_acoustic_codec.types import SemanticCodecBatch, SemanticCodecPairMetadata


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


def _batch() -> SemanticCodecBatch:
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


def _paired_batch() -> SemanticCodecBatch:
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
    metadata = SemanticCodecPairMetadata(
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
    return SemanticCodecBatch(
        semantic_codes=target.semantic_codes,
        acoustic_codes=target.acoustic_codes,
        mask=target.mask,
        semantic_pad_id=target.semantic_pad_id,
        acoustic_pad_ids=target.acoustic_pad_ids,
        acoustic_mask=target.target_acoustic_mask,
        acoustic_layout=target.acoustic_layout,
        reference_semantic_codes=reference.semantic_codes,
        reference_acoustic_codes=reference.acoustic_codes,
        reference_mask=reference.mask,
        reference_acoustic_mask=reference.target_acoustic_mask,
        metadata=(metadata,),
    )


def test_fixed_layout_fm_trains_and_runtime_uses_acoustic_axis(tmp_path) -> None:
    backend = FixedBackend()
    batch = _batch()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)

    output = module.training_step(batch, 0)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()

    module.export_artifact(tmp_path)
    loaded = load_artifact(tmp_path)
    assert loaded.acoustic_layout is AcousticLayout.FIXED_LENGTH
    assert loaded.acoustic_unit_length == backend.acoustic_unit_length

    runtime = SemanticCodecRuntime(loaded, backend)
    encoded = runtime.encode(torch.zeros((1, 1, 16)), 16_000)
    features = runtime.sample_features(
        batch.semantic_codes[:1],
        mask=batch.mask[:1],
        generator=torch.Generator().manual_seed(0),
    )
    waveform = runtime.decode(batch.semantic_codes[:1], mask=batch.mask[:1])

    assert encoded.shape == (1, 2, 1)
    assert features.shape == (1, backend.acoustic_unit_length, backend.acoustic_feature_dim)
    assert waveform.shape == (1, 1, backend.acoustic_unit_length * 8)


def test_fixed_runtime_masks_semantic_padding_before_backend_decode() -> None:
    backend = FixedBackend()
    batch = _batch()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    support = build_module(backend, config, batch, normalize_features=True).support

    waveform = SemanticCodecRuntime(support, backend).decode(
        batch.semantic_codes[1:],
        mask=batch.mask[1:],
    )

    assert waveform.shape == (1, 1, backend.acoustic_unit_length * 8)


def test_fixed_sample_logger_emits_reference_token_passthrough(tmp_path) -> None:
    backend = RecordingFixedBackend()
    batch = _paired_batch()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=10,
        decoder=DecoderConfig(layers=1, heads=2, ffn_ratio=2),
    )
    module = build_module(backend, config, batch, normalize_features=True)
    experiment = SimpleNamespace(audio=[])

    def add_audio(tag: str, value: torch.Tensor, *, global_step: int, sample_rate: int) -> None:
        experiment.audio.append((tag, value, global_step, sample_rate))

    experiment.add_audio = add_audio
    trainer = SimpleNamespace(
        is_global_zero=True,
        global_step=1,
        loggers=[SimpleNamespace(experiment=experiment)],
    )
    callback = SampleLogger(
        tmp_path,
        batch,
        SampleLogConfig(every_n_train_steps=1, seed=11),
    )

    callback.on_train_batch_end(cast(Any, trainer), module, None, object(), 99)

    assert len(backend.detokenized) == 3
    target, reference, passthrough = backend.detokenized
    reference_semantic = batch.reference_semantic_codes
    reference_acoustic = batch.reference_acoustic_codes
    assert reference_semantic is not None
    assert reference_acoustic is not None
    assert torch.equal(target.semantic, batch.semantic_codes)
    assert torch.equal(target.acoustic, batch.acoustic_codes)
    assert torch.equal(reference.semantic, reference_semantic)
    assert torch.equal(reference.acoustic, reference_acoustic)
    assert torch.equal(passthrough.semantic, batch.semantic_codes)
    assert torch.equal(passthrough.acoustic, reference_acoustic)

    audio = {tag: value for tag, value, _, _ in experiment.audio}
    assert "sample/reference_token_passthrough" in audio
    assert torch.equal(
        torch.unique(audio["sample/reference_token_passthrough"]),
        torch.tensor([605.0]),
    )
    events = json.loads((tmp_path / "sample_metrics.json").read_text(encoding="utf-8"))
    assert events[0]["reference_token_passthrough"]["finite"] is True
