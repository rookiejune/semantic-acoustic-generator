from __future__ import annotations

from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence


class Codec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def decode(self, codes: Tensor) -> Tensor: ...


class Teacher(Protocol):
    @property
    def feature_dim(self) -> int: ...

    def __call__(
        self,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> Tensor: ...


class _Encoder(Protocol):
    layers: nn.ModuleList


class WavLMTeacher(nn.Module):
    """Frozen WavLM features aligned to codec frames for the FM route."""

    def __init__(
        self,
        codec: Codec,
        *,
        checkpoint: str = "microsoft/wavlm-base",
        layer: int = 9,
        sample_rate: int = 16_000,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.layer = layer
        self.sample_rate = sample_rate
        try:
            from transformers import WavLMModel
        except ImportError as error:
            raise ImportError("WavLMTeacher requires semantic-acoustic-codec[rvq].") from error
        self.model = WavLMModel.from_pretrained(checkpoint)
        if not 0 <= layer <= self.model.config.num_hidden_layers:
            raise ValueError("WavLM teacher layer is outside hidden_states")
        if layer < self.model.config.num_hidden_layers:
            _truncate_wavlm_encoder(cast(nn.Module, self.model), layer)
        self.model.requires_grad_(False)
        self.model.eval()
        if device is not None:
            cast(nn.Module, self.model).to(device)

    @property
    def feature_dim(self) -> int:
        return self.model.config.hidden_size

    def train(self, mode: bool = True) -> WavLMTeacher:
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if semantic_codes.shape[:2] != acoustic_codes.shape[:2]:
            raise ValueError("teacher semantic and acoustic codes must align")
        if mask.shape != semantic_codes.shape[:2]:
            raise ValueError("teacher mask must align with codec frames")
        if mask.dtype != torch.bool:
            raise TypeError("teacher mask must be boolean")
        if mask.size(0) < 1 or not bool(mask.any(dim=1).all()):
            raise ValueError("each teacher mask row must contain a valid frame")

        inputs, lengths = self._waveforms(semantic_codes, acoustic_codes, mask)
        sample_mask = torch.arange(inputs.size(1), device=self._device)[None] < lengths[:, None]
        output = self.model(
            inputs,
            attention_mask=sample_mask,
            output_hidden_states=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("WavLM did not return hidden states")
        features = hidden_states[self.layer]
        feature_lengths = self._feature_lengths(lengths)
        aligned = features.new_zeros(mask.shape + (self.feature_dim,))
        for row, (feature_length, frame_count) in enumerate(
            zip(feature_lengths.tolist(), mask.sum(dim=1).tolist())
        ):
            source = features[row, :feature_length].transpose(0, 1)[None]
            value = F.interpolate(
                source,
                size=frame_count,
                mode="linear",
                align_corners=False,
            )[0].transpose(0, 1)
            valid = mask[row].to(device=aligned.device)
            aligned[row, valid] = value
        return aligned

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _waveforms(
        self,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode padded codec rows, batching equal frame lengths together.

        Training pad ids sit at codebook size and are out of range for embedding
        lookup, so a mixed-length rectangle cannot be decoded as-is. Rows that
        share a valid length are trimmed and decoded in one ``codec.decode`` call,
        matching per-row decode while avoiding a Python loop over every sample.
        """
        _require_prefix_mask(mask)
        codes = torch.cat((semantic_codes, acoustic_codes), dim=-1)
        frame_lengths = mask.sum(dim=1)
        waveforms: list[Tensor | None] = [None] * codes.size(0)
        for length in frame_lengths.unique(sorted=True).tolist():
            rows = (frame_lengths == length).nonzero(as_tuple=True)[0]
            decoded = self.codec.decode(codes.index_select(0, rows)[:, : int(length)])
            mono = _mono_batch(decoded)
            if self.codec.sample_rate != self.sample_rate:
                from torchaudio.functional import resample

                mono = resample(mono, self.codec.sample_rate, self.sample_rate)
            for local, row in enumerate(rows.tolist()):
                waveforms[row] = _normalize(mono[local])
        if any(waveform is None for waveform in waveforms):
            raise RuntimeError("teacher decode missed at least one batch row")
        resolved = cast(list[Tensor], waveforms)
        lengths = torch.tensor(
            [waveform.numel() for waveform in resolved],
            device=self._device,
        )
        return pad_sequence(resolved, batch_first=True).to(self._device), lengths

    def _feature_lengths(self, lengths: Tensor) -> Tensor:
        output = lengths
        kernels = cast(list[int], self.model.config.conv_kernel)
        strides = cast(list[int], self.model.config.conv_stride)
        for kernel, stride in zip(kernels, strides):
            output = torch.div(output - kernel, stride, rounding_mode="floor") + 1
        return output


def _truncate_wavlm_encoder(model: nn.Module, layer: int) -> None:
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if not isinstance(encoder, nn.Module) or not isinstance(layers, nn.ModuleList):
        raise TypeError("WavLM model must expose encoder.layers as a ModuleList")
    cast(_Encoder, encoder).layers = nn.ModuleList(list(layers[:layer]))


def decode_group_metrics(mask: Tensor) -> dict[str, float]:
    """Stats for length-grouped codec decode; singleton_fraction=1 means fully per-row."""
    if mask.dtype != torch.bool:
        raise TypeError("teacher mask must be boolean")
    if mask.ndim != 2 or mask.size(0) < 1:
        raise ValueError("teacher mask must have shape [batch, frame]")
    _, counts = mask.sum(dim=1).unique(sorted=True, return_counts=True)
    groups = int(counts.numel())
    batch = int(mask.size(0))
    singletons = int((counts == 1).sum().item())
    return {
        "decode_groups": float(groups),
        "decode_group_mean": batch / groups,
        "decode_group_max": float(counts.max().item()),
        "decode_singleton_fraction": singletons / groups,
    }


def _require_prefix_mask(mask: Tensor) -> None:
    lengths = mask.sum(dim=1)
    expected = torch.arange(mask.size(1), device=mask.device)[None] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("teacher mask must describe contiguous right padding")


def _mono_batch(waveform: Tensor) -> Tensor:
    value = waveform.float()
    if value.dim() == 3:
        value = value.mean(dim=1)
    if value.dim() != 2:
        raise ValueError(
            "codec teacher decode must produce [batch, time] or [batch, channel, time]"
        )
    return value


def _normalize(waveform: Tensor) -> Tensor:
    return (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)


__all__ = ["Teacher", "WavLMTeacher", "decode_group_metrics"]
