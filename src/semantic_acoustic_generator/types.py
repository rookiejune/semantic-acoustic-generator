from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from anytrain.codec import AcousticLayout
from torch import Tensor

from semantic_acoustic_generator._tensor import is_signed_integer_dtype


@dataclass(frozen=True)
class PairMetadata:
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

    def as_dict(self, *, include_private: bool = False) -> dict[str, int | str]:
        """Serialize pair metadata with an explicit public allowlist."""
        data: dict[str, int | str] = {
            "target_index": self.target_index,
            "reference_index": self.reference_index,
            "target_text_index": self.target_text_index,
            "reference_text_index": self.reference_text_index,
            "target_source_index": self.target_source_index,
            "reference_source_index": self.reference_source_index,
            "target_role": self.target_role,
            "reference_role": self.reference_role,
        }
        if include_private:
            data.update(
                {
                    "target_utterance_id": self.target_utterance_id,
                    "reference_utterance_id": self.reference_utterance_id,
                    "target_speaker_id": self.target_speaker_id,
                    "reference_speaker_id": self.reference_speaker_id,
                    "target_text": self.target_text,
                    "reference_text": self.reference_text,
                }
            )
        return data

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


@dataclass(frozen=True)
class BatchSide:
    semantic_codes: Tensor
    acoustic_codes: Tensor
    mask: Tensor
    acoustic_mask: Tensor


@dataclass(eq=False)
class GeneratorBatch:
    semantic_codes: Tensor
    acoustic_codes: Tensor
    mask: Tensor
    semantic_pad_id: int
    acoustic_pad_ids: tuple[int, ...]
    acoustic_mask: Tensor
    acoustic_layout: AcousticLayout = AcousticLayout.FRAME_ALIGNED
    reference_semantic_codes: Tensor | None = None
    reference_acoustic_codes: Tensor | None = None
    reference_mask: Tensor | None = None
    reference_acoustic_mask: Tensor | None = None
    metadata: tuple[PairMetadata, ...] = ()
    _semantic_valid_frames: int = field(init=False, repr=False)
    _semantic_padded_frames: int = field(init=False, repr=False)
    _acoustic_valid_units: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.acoustic_layout, AcousticLayout):
            raise TypeError("acoustic_layout must be an AcousticLayout.")
        if self.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
            raise ValueError(
                "GeneratorBatch supports only frame-aligned semantic/acoustic units."
            )
        _check_positive_int(self.semantic_pad_id, name="semantic_pad_id")
        for index, pad_id in enumerate(self.acoustic_pad_ids):
            _check_positive_int(pad_id, name=f"acoustic_pad_ids[{index}]")
        _validate_side(
            semantic_codes=self.semantic_codes,
            acoustic_codes=self.acoustic_codes,
            mask=self.mask,
            acoustic_mask=self.acoustic_mask,
            semantic_pad_id=self.semantic_pad_id,
            acoustic_pad_ids=self.acoustic_pad_ids,
            acoustic_layout=self.acoustic_layout,
            name="target",
        )
        self._semantic_valid_frames = int(self.mask.sum().item())
        self._semantic_padded_frames = self.mask.numel()
        self._acoustic_valid_units = int(self.acoustic_mask.sum().item())
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
    def semantic_valid_frames(self) -> int:
        return self._semantic_valid_frames

    @property
    def semantic_padded_frames(self) -> int:
        return self._semantic_padded_frames

    @property
    def acoustic_valid_units(self) -> int:
        return self._acoustic_valid_units

    @property
    def reference(self) -> BatchSide:
        return BatchSide(
            semantic_codes=_required_reference(
                self.reference_semantic_codes,
                name="reference_semantic_codes",
            ),
            acoustic_codes=_required_reference(
                self.reference_acoustic_codes,
                name="reference_acoustic_codes",
            ),
            mask=_required_reference(self.reference_mask, name="reference_mask"),
            acoustic_mask=_required_reference(
                self.reference_acoustic_mask,
                name="reference_acoustic_mask",
            ),
        )

    @property
    def has_reference(self) -> bool:
        return self.reference_acoustic_codes is not None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> GeneratorBatch:
        """Move all codec tensors while preserving pair metadata and layout."""

        def move(value: Tensor) -> Tensor:
            return value.to(device=device, non_blocking=non_blocking)

        return self._map_tensors(move)

    def pin_memory(self) -> GeneratorBatch:
        """Pin all codec tensors for an asynchronous accelerator transfer."""

        def pin(value: Tensor) -> Tensor:
            return value.pin_memory()

        return self._map_tensors(pin)

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> GeneratorBatch:
        transformed: dict[int, Tensor] = {}

        def required(value: Tensor) -> Tensor:
            key = id(value)
            result = transformed.get(key)
            if result is None:
                result = transform(value)
                transformed[key] = result
            return result

        def optional(value: Tensor | None) -> Tensor | None:
            return None if value is None else required(value)

        return self._from_validated(
            semantic_codes=required(self.semantic_codes),
            acoustic_codes=required(self.acoustic_codes),
            mask=required(self.mask),
            semantic_pad_id=self.semantic_pad_id,
            acoustic_pad_ids=self.acoustic_pad_ids,
            acoustic_mask=required(self.acoustic_mask),
            acoustic_layout=self.acoustic_layout,
            reference_semantic_codes=optional(self.reference_semantic_codes),
            reference_acoustic_codes=optional(self.reference_acoustic_codes),
            reference_mask=optional(self.reference_mask),
            reference_acoustic_mask=optional(self.reference_acoustic_mask),
            metadata=self.metadata,
            semantic_valid_frames=self.semantic_valid_frames,
            semantic_padded_frames=self.semantic_padded_frames,
            acoustic_valid_units=self.acoustic_valid_units,
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
        semantic_pad_id: int,
        acoustic_pad_ids: tuple[int, ...],
        acoustic_mask: Tensor,
        acoustic_layout: AcousticLayout,
        reference_semantic_codes: Tensor | None,
        reference_acoustic_codes: Tensor | None,
        reference_mask: Tensor | None,
        reference_acoustic_mask: Tensor | None,
        metadata: tuple[PairMetadata, ...],
        semantic_valid_frames: int,
        semantic_padded_frames: int,
        acoustic_valid_units: int,
    ) -> GeneratorBatch:
        """Copy an already validated batch after a tensor-preserving transform."""
        batch = object.__new__(cls)
        batch.semantic_codes = semantic_codes
        batch.acoustic_codes = acoustic_codes
        batch.mask = mask
        batch.semantic_pad_id = semantic_pad_id
        batch.acoustic_pad_ids = acoustic_pad_ids
        batch.acoustic_mask = acoustic_mask
        batch.acoustic_layout = acoustic_layout
        batch.reference_semantic_codes = reference_semantic_codes
        batch.reference_acoustic_codes = reference_acoustic_codes
        batch.reference_mask = reference_mask
        batch.reference_acoustic_mask = reference_acoustic_mask
        batch.metadata = metadata
        batch._semantic_valid_frames = semantic_valid_frames
        batch._semantic_padded_frames = semantic_padded_frames
        batch._acoustic_valid_units = acoustic_valid_units
        return batch


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
    if acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
        raise ValueError("generator batches require frame-aligned acoustic units.")
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


def _required_reference(value: Tensor | None, *, name: str) -> Tensor:
    if value is None:
        raise RuntimeError(f"{name} is required for paired semantic codec batches.")
    return value


def _check_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = ["GeneratorBatch", "PairMetadata", "BatchSide"]
