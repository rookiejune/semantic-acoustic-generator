from __future__ import annotations

from dataclasses import dataclass

import torch
from anytrain.codec import AcousticLayout
from torch import Tensor

from semantic_acoustic_codec._tensor import is_signed_integer_dtype


@dataclass(frozen=True)
class SemanticCodecPairMetadata:
    target_index: int
    reference_index: int
    target_text_index: int
    reference_text_index: int
    target_source_index: int
    reference_source_index: int
    target_role: str
    reference_role: str
    target_utterance_id: str
    reference_utterance_id: str
    target_speaker_id: str
    reference_speaker_id: str
    target_text: str
    reference_text: str

    def __post_init__(self) -> None:
        for name, value in (
            ("target_index", self.target_index),
            ("reference_index", self.reference_index),
            ("target_text_index", self.target_text_index),
            ("reference_text_index", self.reference_text_index),
            ("target_source_index", self.target_source_index),
            ("reference_source_index", self.reference_source_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name, value in (
            ("target_role", self.target_role),
            ("reference_role", self.reference_role),
            ("target_utterance_id", self.target_utterance_id),
            ("reference_utterance_id", self.reference_utterance_id),
            ("target_speaker_id", self.target_speaker_id),
            ("reference_speaker_id", self.reference_speaker_id),
            ("target_text", self.target_text),
            ("reference_text", self.reference_text),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.target_index == self.reference_index:
            raise ValueError("target and reference indices must differ.")
        if self.target_text_index == self.reference_text_index:
            raise ValueError("target and reference text indices must differ.")
        if self.target_utterance_id == self.reference_utterance_id:
            raise ValueError("target and reference utterance IDs must differ.")
        if self.target_speaker_id != self.reference_speaker_id:
            raise ValueError("target and reference speakers must match.")
        if self.target_text == self.reference_text:
            raise ValueError("target and reference text must differ.")


@dataclass(eq=False)
class SemanticCodecBatch:
    semantic_codes: Tensor
    acoustic_codes: Tensor
    mask: Tensor
    semantic_pad_id: int
    acoustic_pad_ids: tuple[int, ...]
    acoustic_mask: Tensor | None = None
    acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED
    reference_semantic_codes: Tensor | None = None
    reference_acoustic_codes: Tensor | None = None
    reference_mask: Tensor | None = None
    reference_acoustic_mask: Tensor | None = None
    metadata: tuple[SemanticCodecPairMetadata, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.acoustic_layout, AcousticLayout):
            raise TypeError("acoustic_layout must be an AcousticLayout.")
        acoustic_mask = self.acoustic_mask
        if acoustic_mask is None:
            acoustic_mask = self.mask
            self.acoustic_mask = acoustic_mask
        _check_positive_int(self.semantic_pad_id, name="semantic_pad_id")
        for index, pad_id in enumerate(self.acoustic_pad_ids):
            _check_positive_int(pad_id, name=f"acoustic_pad_ids[{index}]")
        _validate_side(
            semantic_codes=self.semantic_codes,
            acoustic_codes=self.acoustic_codes,
            mask=self.mask,
            acoustic_mask=acoustic_mask,
            semantic_pad_id=self.semantic_pad_id,
            acoustic_pad_ids=self.acoustic_pad_ids,
            acoustic_layout=self.acoustic_layout,
            name="target",
        )
        references = (
            self.reference_semantic_codes,
            self.reference_acoustic_codes,
            self.reference_mask,
            self.reference_acoustic_mask,
        )
        if all(value is None for value in references):
            if self.metadata:
                raise ValueError("pair metadata requires reference codec units.")
            return
        if any(value is None for value in references):
            raise ValueError("reference codec units and masks must be provided together.")
        reference_semantic = _tensor(self.reference_semantic_codes)
        reference_acoustic = _tensor(self.reference_acoustic_codes)
        reference_mask = _tensor(self.reference_mask)
        reference_acoustic_mask = _tensor(self.reference_acoustic_mask)
        _validate_side(
            semantic_codes=reference_semantic,
            acoustic_codes=reference_acoustic,
            mask=reference_mask,
            acoustic_mask=reference_acoustic_mask,
            semantic_pad_id=self.semantic_pad_id,
            acoustic_pad_ids=self.acoustic_pad_ids,
            acoustic_layout=self.acoustic_layout,
            name="reference",
        )
        if reference_semantic.size(0) != self.semantic_codes.size(0):
            raise ValueError("target and reference batch sizes must match.")
        if len(self.metadata) != self.semantic_codes.size(0):
            raise ValueError("pair metadata must contain one item per batch row.")

    @property
    def semantic_codebook_size(self) -> int:
        return self.semantic_pad_id

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]:
        return self.acoustic_pad_ids

    @property
    def semantic_mask(self) -> Tensor:
        """Validity mask for the semantic time axis."""
        return self.mask

    @property
    def target_semantic_codes(self) -> Tensor:
        return self.semantic_codes

    @property
    def target_acoustic_codes(self) -> Tensor:
        return self.acoustic_codes

    @property
    def target_mask(self) -> Tensor:
        return self.mask

    @property
    def target_acoustic_mask(self) -> Tensor:
        if self.acoustic_mask is None:
            raise RuntimeError("SemanticCodecBatch must expose acoustic_mask after validation.")
        return self.acoustic_mask

    @property
    def has_reference(self) -> bool:
        return self.reference_acoustic_codes is not None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> SemanticCodecBatch:
        """Move all codec tensors while preserving pair metadata and layout."""

        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device=device, non_blocking=non_blocking)

        return SemanticCodecBatch(
            semantic_codes=self.semantic_codes.to(device=device, non_blocking=non_blocking),
            acoustic_codes=self.acoustic_codes.to(device=device, non_blocking=non_blocking),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
            semantic_pad_id=self.semantic_pad_id,
            acoustic_pad_ids=self.acoustic_pad_ids,
            acoustic_mask=self.target_acoustic_mask.to(
                device=device,
                non_blocking=non_blocking,
            ),
            acoustic_layout=self.acoustic_layout,
            reference_semantic_codes=move(self.reference_semantic_codes),
            reference_acoustic_codes=move(self.reference_acoustic_codes),
            reference_mask=move(self.reference_mask),
            reference_acoustic_mask=move(self.reference_acoustic_mask),
            metadata=self.metadata,
        )


def _validate_side(
    *,
    semantic_codes: Tensor,
    acoustic_codes: Tensor,
    mask: Tensor,
    acoustic_mask: Tensor,
    semantic_pad_id: int,
    acoustic_pad_ids: tuple[int, ...],
    acoustic_layout: AcousticLayout,
    name: str,
) -> None:
    if semantic_codes.dim() != 3 or semantic_codes.size(-1) != 1:
        raise ValueError(f"{name} semantic_codes must have shape [B, F, 1].")
    if acoustic_codes.dim() != 3 or acoustic_codes.size(-1) < 1:
        raise ValueError(f"{name} acoustic_codes must have shape [B, unit, K].")
    if semantic_codes.size(0) != acoustic_codes.size(0):
        raise ValueError(f"{name} semantic and acoustic codes must share the batch axis.")
    if mask.shape != semantic_codes.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(f"{name} mask must be boolean with shape [B, F].")
    if acoustic_mask.shape != acoustic_codes.shape[:2] or acoustic_mask.dtype != torch.bool:
        raise ValueError(f"{name} acoustic_mask must be boolean with shape [B, unit].")
    devices = {semantic_codes.device, acoustic_codes.device, mask.device, acoustic_mask.device}
    if len(devices) != 1:
        raise ValueError(f"{name} codec units and masks must use the same device.")
    if acoustic_codes.size(-1) != len(acoustic_pad_ids):
        raise ValueError(f"{name} acoustic codebooks must match acoustic_pad_ids.")
    if acoustic_layout is AcousticLayout.FRAME_ALIGNED:
        if semantic_codes.shape[:2] != acoustic_codes.shape[:2]:
            raise ValueError(f"frame-aligned {name} codes must share the [B, F] axis.")
        if not torch.equal(mask, acoustic_mask):
            raise ValueError(f"frame-aligned {name} semantic and acoustic masks must match.")
    if not is_signed_integer_dtype(semantic_codes.dtype):
        raise TypeError(f"{name} semantic_codes must use a signed integer dtype.")
    if not is_signed_integer_dtype(acoustic_codes.dtype):
        raise TypeError(f"{name} acoustic_codes must use a signed integer dtype.")
    if semantic_codes.size(0) < 1 or semantic_codes.size(1) < 1:
        raise ValueError(f"{name} codec batch must not be empty.")
    if not bool(mask.any(dim=1).all()):
        raise ValueError(f"each {name} batch row must contain at least one valid frame.")
    if not bool(acoustic_mask.any(dim=1).all()):
        raise ValueError(f"each {name} batch row must contain a valid acoustic unit.")
    if bool((semantic_codes < 0).any()) or bool((acoustic_codes < 0).any()):
        raise ValueError(f"{name} codec units must not contain negative IDs.")
    semantic = semantic_codes[..., 0]
    if bool((semantic[mask] >= semantic_pad_id).any()):
        raise ValueError(f"valid {name} semantic_codes contain an ID outside the codebook.")
    if bool((semantic[~mask] != semantic_pad_id).any()):
        raise ValueError(f"padded {name} semantic_codes must equal semantic_pad_id.")
    pad_ids = torch.tensor(
        acoustic_pad_ids,
        device=acoustic_codes.device,
        dtype=acoustic_codes.dtype,
    )
    if bool((acoustic_codes[acoustic_mask] >= pad_ids).any()):
        raise ValueError(f"valid {name} acoustic_codes contain an ID outside their codebooks.")
    if bool((acoustic_codes[~acoustic_mask] != pad_ids).any()):
        raise ValueError(f"padded {name} acoustic_codes must equal acoustic_pad_ids.")


def _tensor(value: Tensor | None) -> Tensor:
    if value is None:
        raise AssertionError("validated reference tensor is unexpectedly missing.")
    return value


def _check_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = ["SemanticCodecBatch", "SemanticCodecPairMetadata"]
