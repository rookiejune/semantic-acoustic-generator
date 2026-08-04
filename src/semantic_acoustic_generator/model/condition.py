from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from semantic_acoustic_generator._tensor import is_signed_integer_dtype
from semantic_acoustic_generator.config import Initialization

if TYPE_CHECKING:
    from torch import Tensor


class SemanticConditioner(nn.Module):
    """Frame-level semantic conditioner for codec semantic codes and token spans."""

    def __init__(
        self,
        semantic_codebook: Tensor,
        *,
        condition_dim: int,
        initialization: Initialization = Initialization.CODEC,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")
        weight = _semantic_weight(
            semantic_codebook,
            initialization=initialization,
            seed=seed,
        )
        pad = weight.new_zeros(1, weight.size(-1))
        weight = torch.cat([weight, pad], dim=0)
        self.embedding = nn.Embedding.from_pretrained(
            weight,
            freeze=False,
            padding_idx=weight.size(0) - 1,
        )
        self.projection = (
            nn.Identity()
            if self.embedding.embedding_dim == condition_dim
            else nn.Linear(self.embedding.embedding_dim, condition_dim)
        )

    @property
    def semantic_codebook_size(self) -> int:
        return self.embedding.num_embeddings - 1

    @property
    def semantic_pad_id(self) -> int:
        return self.semantic_codebook_size

    @property
    def condition_dim(self) -> int:
        value = self.projection(self.embedding.weight[:1])
        return int(value.size(-1))

    def forward(
        self,
        semantic_codes: Tensor,
        spans: Tensor | None = None,
        *,
        frames: int | None = None,
        validate: bool = True,
    ) -> Tensor:
        if semantic_codes.dim() == 3:
            if semantic_codes.size(-1) != 1:
                raise ValueError("semantic_codes with rank 3 must have shape [B, F, 1].")
            semantic_tokens = semantic_codes[..., 0]
        elif semantic_codes.dim() == 2:
            semantic_tokens = semantic_codes
        else:
            raise ValueError("semantic_codes must have shape [B, F, 1] or [B, T].")
        if validate:
            _validate_semantic_tokens(semantic_tokens, pad_id=self.semantic_pad_id)
        tokens = semantic_tokens.to(dtype=torch.long)
        condition = self.projection(self.embedding(tokens))
        condition = condition.masked_fill(tokens.eq(self.semantic_pad_id)[..., None], 0)
        if spans is None:
            return condition
        return repeat_condition(condition, spans, frames=frames)


class ReferenceConditioner(nn.Module):
    """Pool acoustic references or emit a learned null condition."""

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0 or condition_dim <= 0:
            raise ValueError("feature_dim and condition_dim must be positive.")
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.null_condition = nn.Parameter(torch.zeros(condition_dim))
        self.projection = nn.Linear(feature_dim, condition_dim)
        self.norm = nn.LayerNorm(condition_dim)
        self.gate = nn.Parameter(torch.zeros(condition_dim))

    def forward(
        self,
        features: Tensor | None,
        *,
        mask: Tensor | None = None,
        batch_size: int | None = None,
        use_reference: Tensor | None = None,
        row_indices: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        if features is None:
            if batch_size is None:
                raise ValueError("batch_size is required for the null reference.")
            if validate and batch_size < 1:
                raise ValueError("batch_size must be positive for the null reference.")
            if validate and (
                mask is not None or use_reference is not None or row_indices is not None
            ):
                raise ValueError("reference mask/presence require explicit reference features.")
            return self.null_condition.view(1, 1, -1).expand(batch_size, 1, -1)

        if validate and (features.dim() != 3 or features.size(-1) != self.feature_dim):
            raise ValueError("reference features must have shape [B, F, feature_dim].")
        if validate and use_reference is not None and row_indices is not None:
            raise ValueError("use_reference and row_indices are mutually exclusive.")
        if (
            validate
            and row_indices is None
            and batch_size is not None
            and features.size(0) != batch_size
        ):
            raise ValueError("reference batch must match semantic batch.")
        if mask is None:
            mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
        elif validate and mask.shape != features.shape[:2]:
            raise ValueError("reference mask must align with reference features.")
        elif validate and mask.dtype != torch.bool:
            raise TypeError("reference mask must be boolean.")
        mask = mask.to(device=features.device)
        if validate and not bool(mask.any(dim=1).all()):
            raise ValueError("each reference row must contain at least one valid frame.")
        if validate and (
            use_reference is not None
            and (use_reference.shape != (features.size(0),) or use_reference.dtype != torch.bool)
        ):
            raise ValueError("use_reference must be boolean with shape [B].")
        if row_indices is not None:
            if batch_size is None:
                raise ValueError("batch_size is required with packed reference rows.")
            if validate and (
                row_indices.shape != (features.size(0),)
                or not is_signed_integer_dtype(row_indices.dtype)
            ):
                raise ValueError("row_indices must be integer with shape [reference_batch].")
            if validate and bool(
                ((row_indices < 0) | (row_indices >= batch_size)).any()
            ):
                raise ValueError("row_indices must select rows within batch_size.")
            if validate and row_indices.unique().numel() != row_indices.numel():
                raise ValueError("row_indices must not contain duplicates.")

        if use_reference is not None:
            use_reference = use_reference.to(device=features.device)
        if row_indices is not None:
            row_indices = row_indices.to(device=features.device, dtype=torch.long)

        weights = mask[..., None].to(dtype=features.dtype)
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        pooled = self.norm(self.projection(pooled))
        conditioned = pooled[:, None] * torch.tanh(self.gate)[None, None]
        if row_indices is not None:
            if batch_size is None:
                raise RuntimeError("packed reference rows require batch_size.")
            null = self.null_condition.view(1, 1, -1).expand(batch_size, 1, -1)
            return null.index_copy(0, row_indices, conditioned)
        if use_reference is None:
            return conditioned
        null = self.null_condition.view(1, 1, -1).expand(features.size(0), 1, -1)
        return torch.where(use_reference[:, None, None], conditioned, null)


class FixedLengthConditioner(nn.Module):
    """Read semantic memory with one learned query per fixed acoustic slot."""

    def __init__(self, condition_dim: int, *, slots: int) -> None:
        super().__init__()
        if (
            isinstance(condition_dim, bool)
            or not isinstance(condition_dim, int)
            or isinstance(slots, bool)
            or not isinstance(slots, int)
        ):
            raise TypeError("condition_dim and slots must be integers.")
        if condition_dim <= 0 or slots <= 0:
            raise ValueError("condition_dim and slots must be positive.")
        self.condition_dim = condition_dim
        self.slots = slots
        self.slot_queries = nn.Parameter(torch.empty(slots, condition_dim))
        _ = nn.init.normal_(self.slot_queries, std=condition_dim**-0.5)
        self.query_norm = _norm(condition_dim)
        self.memory_norm = _norm(condition_dim)
        self.output_norm = _norm(condition_dim)
        self._position_cache: Tensor = torch.empty(0, condition_dim)

    def forward(
        self,
        memory: Tensor,
        mask: Tensor,
        *,
        output_length: int,
        validate: bool = True,
    ) -> Tensor:
        if validate and (memory.dim() != 3 or memory.size(-1) != self.condition_dim):
            raise ValueError("semantic memory must have shape [B, F, condition_dim].")
        if validate and (not torch.is_floating_point(memory) or torch.is_complex(memory)):
            raise TypeError("semantic memory must be floating point.")
        if validate and mask.shape != memory.shape[:2]:
            raise ValueError("semantic memory mask must align on [B, F].")
        if validate and mask.dtype != torch.bool:
            raise TypeError("semantic memory mask must be boolean.")
        if validate and mask.device != memory.device:
            raise ValueError("semantic memory and mask must use the same device.")
        if validate and not bool(mask.any(dim=1).all()):
            raise ValueError("each semantic memory row must contain at least one valid unit.")
        if validate and (isinstance(output_length, bool) or not isinstance(output_length, int)):
            raise TypeError("fixed-length output_length must be an integer.")
        if validate and (output_length < 1 or output_length > self.slots):
            raise ValueError(
                f"fixed-length output_length must be in [1, {self.slots}], got {output_length}."
            )

        positions = self._positions(memory)
        normalized_memory = self.memory_norm(memory + positions[None])
        queries = self.query_norm(self.slot_queries[:output_length].to(dtype=memory.dtype))
        queries = queries[None].expand(memory.size(0), -1, -1)
        scores = torch.matmul(queries, normalized_memory.transpose(1, 2))
        scores = scores * self.condition_dim**-0.5
        scores = scores.masked_fill(~mask[:, None], torch.finfo(scores.dtype).min)
        weights = scores.float().softmax(dim=-1).to(dtype=normalized_memory.dtype)
        context = torch.matmul(weights, normalized_memory)
        return self.output_norm(queries + context)

    def _positions(self, memory: Tensor) -> Tensor:
        length = memory.size(1)
        cache = self._position_cache
        same_type = cache.device == memory.device and cache.dtype == memory.dtype
        if not same_type or cache.size(0) < length:
            previous = cache.size(0) if same_type else 0
            capacity = max(length, max(1, previous * 2))
            cache = _sinusoidal_positions(
                capacity,
                self.condition_dim,
                device=memory.device,
                dtype=memory.dtype,
            )
            self._position_cache = cache
        return cache[:length]


class AlignedAnchor(nn.Module):
    """Predict one acoustic feature per semantic frame from bounded local context."""

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        *,
        hidden_dim: int,
        layers: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if min(condition_dim, feature_dim, hidden_dim, layers) <= 0:
            raise ValueError("aligned anchor dimensions and layers must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("aligned anchor kernel_size must be a positive odd integer.")
        self.input = nn.Linear(condition_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_AlignedAnchorBlock(hidden_dim, kernel_size=kernel_size) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, condition: Tensor, mask: Tensor) -> Tensor:
        if condition.dim() != 3 or mask.shape != condition.shape[:2]:
            raise ValueError("aligned anchor condition and mask must align on [batch, frame].")
        if mask.dtype != torch.bool:
            raise TypeError("aligned anchor mask must be boolean.")
        hidden = self.input(condition).masked_fill(~mask[..., None], 0)
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.output(self.norm(hidden)).masked_fill(~mask[..., None], 0)


class _AlignedAnchorBlock(nn.Module):
    def __init__(self, hidden_dim: int, *, kernel_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.local = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        value = self.norm(hidden).transpose(1, 2).contiguous()
        value = self.local(value).transpose(1, 2).contiguous()
        activation, gate = self.gate(value).chunk(2, dim=-1)
        value = self.output(torch.nn.functional.silu(activation) * gate.sigmoid())
        return (hidden + value).masked_fill(~mask[..., None], 0)


def repeat_condition(condition: Tensor, spans: Tensor, *, frames: int | None) -> Tensor:
    if condition.dim() != 3 or spans.dim() != 2:
        raise ValueError("condition and spans must have shapes [B, T, D] and [B, T].")
    if condition.shape[:2] != spans.shape:
        raise ValueError("condition and spans must align on batch and token axes.")
    if spans.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("semantic token spans must use an integer dtype.")
    if bool((spans < 0).any()):
        raise ValueError("semantic token spans must be non-negative.")

    frame_count = int(spans.sum(dim=1).max()) if frames is None else frames
    if frame_count < 1:
        raise ValueError("semantic token spans must cover at least one frame.")
    output = condition.new_zeros(condition.size(0), frame_count, condition.size(-1))
    span_rows = spans.to(device=condition.device, dtype=torch.long)
    for row_index in range(condition.size(0)):
        row_spans = span_rows[row_index]
        valid = row_spans > 0
        repeated = torch.repeat_interleave(
            condition[row_index, valid],
            row_spans[valid],
            dim=0,
        )
        if repeated.size(0) == 0:
            raise ValueError("semantic token spans must cover at least one frame.")
        if repeated.size(0) > frame_count:
            raise ValueError("semantic token spans exceed the target frame count.")
        output[row_index, : repeated.size(0)] = repeated
    return output


def _validate_semantic_tokens(values: Tensor, *, pad_id: int) -> None:
    if not is_signed_integer_dtype(values.dtype):
        raise TypeError("semantic_codes must use a signed integer dtype.")
    if bool((values < 0).any()):
        raise ValueError("semantic_codes must not contain negative IDs.")
    if bool((values > pad_id).any()):
        raise ValueError(
            "semantic_codes contain an ID outside the semantic codebook and pad token."
        )


def matched_random_weight(reference: Tensor, *, seed: int, rows: int | None = None) -> Tensor:
    shape = reference.shape if rows is None else (rows, reference.size(-1))
    output = reference.new_empty(shape)
    generator = torch.Generator(device=output.device).manual_seed(seed)
    return output.normal_(
        mean=float(reference.mean()),
        std=float(reference.std(correction=0)),
        generator=generator,
    )


def _semantic_weight(
    codebook: Tensor,
    *,
    initialization: Initialization,
    seed: int,
) -> Tensor:
    if codebook.dim() != 2 or not torch.is_floating_point(codebook):
        raise ValueError("semantic codebook must have shape [vocab, dim] and floating dtype.")
    if codebook.size(0) < 1 or codebook.size(1) < 1:
        raise ValueError("semantic codebook must be non-empty.")
    if not bool(torch.isfinite(codebook).all()):
        raise ValueError("semantic codebook must contain finite values.")
    if initialization is Initialization.CODEC:
        return codebook.detach().clone()
    if initialization is Initialization.RANDOM:
        return matched_random_weight(codebook.detach(), seed=seed)
    raise AssertionError(f"unsupported initialization: {initialization}")


def _sinusoidal_positions(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10_000.0) / dim)
    )
    angles = positions * frequencies[None]
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, : dim // 2].cos()
    return encoding.to(dtype=dtype)


def _norm(dim: int) -> nn.Module:
    return nn.Identity() if dim == 1 else nn.LayerNorm(dim)
