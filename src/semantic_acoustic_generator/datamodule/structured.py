from __future__ import annotations

from collections.abc import Sequence

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from torch import Tensor

from semantic_acoustic_generator._tensor import is_signed_integer_dtype
from semantic_acoustic_generator.types import GeneratorBatch


def collate_structured_codes(
    values: Sequence[SemanticAcousticCodes],
    *,
    semantic_pad_id: int,
    acoustic_pad_ids: Sequence[int],
    acoustic_layout: AcousticLayout,
) -> GeneratorBatch:
    """Pad structured codec units without collapsing their semantic/acoustic axes."""
    if not values:
        raise ValueError("cannot collate an empty structured codec batch.")
    if not isinstance(acoustic_layout, AcousticLayout):
        raise TypeError("acoustic_layout must be an AcousticLayout.")
    _positive_id(semantic_pad_id, name="semantic_pad_id")
    pads = tuple(int(value) for value in acoustic_pad_ids)
    if not pads or any(value <= 0 for value in pads):
        raise ValueError("acoustic_pad_ids must contain positive IDs.")

    samples = [_sample(value, index=index) for index, value in enumerate(values)]
    semantic_device = samples[0][0].device
    acoustic_device = samples[0][1].device
    if any(
        semantic.device != semantic_device or acoustic.device != acoustic_device
        for semantic, acoustic in samples
    ):
        raise ValueError("structured batch samples must use consistent devices.")
    codebooks = samples[0][1].size(1)
    if len(pads) != codebooks:
        raise ValueError("acoustic_pad_ids must match the acoustic codebook axis.")
    if any(acoustic.size(1) != codebooks for _, acoustic in samples):
        raise ValueError("structured acoustic codebooks must be consistent across samples.")
    if acoustic_layout is AcousticLayout.FRAME_ALIGNED and any(
        semantic.size(0) != acoustic.size(0) for semantic, acoustic in samples
    ):
        raise ValueError("frame-aligned structured samples must share semantic/acoustic lengths.")

    semantic_length = max(semantic.size(0) for semantic, _ in samples)
    acoustic_length = max(acoustic.size(0) for _, acoustic in samples)
    semantic = samples[0][0].new_full(
        (len(samples), semantic_length, 1),
        semantic_pad_id,
    )
    acoustic = samples[0][1].new_empty((len(samples), acoustic_length, codebooks))
    pad_tensor = torch.tensor(pads, dtype=acoustic.dtype, device=acoustic.device)
    acoustic[:] = pad_tensor
    semantic_mask = torch.zeros(
        (len(samples), semantic_length),
        dtype=torch.bool,
        device=semantic.device,
    )
    acoustic_mask = torch.zeros(
        (len(samples), acoustic_length),
        dtype=torch.bool,
        device=acoustic.device,
    )
    for index, (sample_semantic, sample_acoustic) in enumerate(samples):
        semantic_length_i = sample_semantic.size(0)
        acoustic_length_i = sample_acoustic.size(0)
        semantic[index, :semantic_length_i] = sample_semantic
        acoustic[index, :acoustic_length_i] = sample_acoustic
        semantic_mask[index, :semantic_length_i] = True
        acoustic_mask[index, :acoustic_length_i] = True
    return GeneratorBatch(
        semantic_codes=semantic,
        acoustic_codes=acoustic,
        mask=semantic_mask,
        semantic_pad_id=semantic_pad_id,
        acoustic_pad_ids=pads,
        acoustic_mask=acoustic_mask,
        acoustic_layout=acoustic_layout,
    )


def _sample(value: SemanticAcousticCodes, *, index: int) -> tuple[Tensor, Tensor]:
    semantic = value.semantic
    acoustic = value.acoustic
    if semantic.dim() != 2 or semantic.size(1) != 1:
        raise ValueError(f"structured sample {index} semantic must have shape [time, 1].")
    if acoustic.dim() != 2 or acoustic.size(1) < 1:
        raise ValueError(f"structured sample {index} acoustic must have shape [unit, codebook].")
    if not is_signed_integer_dtype(semantic.dtype):
        raise TypeError(f"structured sample {index} semantic must use a signed integer dtype.")
    if not is_signed_integer_dtype(acoustic.dtype):
        raise TypeError(f"structured sample {index} acoustic must use a signed integer dtype.")
    if semantic.size(0) < 1 or acoustic.size(0) < 1:
        raise ValueError(f"structured sample {index} must contain semantic and acoustic units.")
    if semantic.device != acoustic.device:
        raise ValueError(f"structured sample {index} semantic/acoustic devices must match.")
    if bool((semantic < 0).any()) or bool((acoustic < 0).any()):
        raise ValueError(f"structured sample {index} must not contain negative IDs.")
    return semantic.contiguous(), acoustic.contiguous()


def _positive_id(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = ["collate_structured_codes"]
