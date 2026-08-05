from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from semantic_acoustic_generator._tensor import is_signed_integer_dtype
from semantic_acoustic_generator.config import AnchorContext, Initialization

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


class AlignedAnchor(nn.Module):
    """Predict one acoustic feature per semantic frame without changing the frame axis."""

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        *,
        hidden_dim: int,
        layers: int,
        kernel_size: int,
        context: AnchorContext = AnchorContext.LOCAL,
        heads: int = 8,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        if min(condition_dim, feature_dim, hidden_dim, layers) <= 0:
            raise ValueError("aligned anchor dimensions and layers must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("aligned anchor kernel_size must be a positive odd integer.")
        if not isinstance(context, AnchorContext):
            raise TypeError("aligned anchor context must be an AnchorContext.")
        if heads <= 0 or ffn_ratio <= 0:
            raise ValueError("aligned anchor heads and ffn_ratio must be positive.")
        if context is AnchorContext.TRANSFORMER and hidden_dim % heads != 0:
            raise ValueError("transformer anchor hidden_dim must be divisible by heads.")
        self.context = context
        self.input = nn.Linear(condition_dim, hidden_dim)
        if context is AnchorContext.LOCAL:
            blocks = [_LocalAnchorBlock(hidden_dim, kernel_size=kernel_size) for _ in range(layers)]
        else:
            blocks = [
                _TransformerAnchorBlock(hidden_dim, heads=heads, ffn_ratio=ffn_ratio)
                for _ in range(layers)
            ]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, condition: Tensor, mask: Tensor) -> Tensor:
        if condition.dim() != 3 or mask.shape != condition.shape[:2]:
            raise ValueError("aligned anchor condition and mask must align on [batch, frame].")
        if mask.dtype != torch.bool:
            raise TypeError("aligned anchor mask must be boolean.")
        hidden = self.input(condition).masked_fill(~mask[..., None], 0)
        if self.context is AnchorContext.TRANSFORMER:
            hidden = (hidden + _sinusoidal_positions(hidden)).masked_fill(
                ~mask[..., None],
                0,
            )
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.output(self.norm(hidden)).masked_fill(~mask[..., None], 0)


class _LocalAnchorBlock(nn.Module):
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


class _TransformerAnchorBlock(nn.Module):
    def __init__(self, hidden_dim: int, *, heads: int, ffn_ratio: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
        )

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        value = self.attention_norm(hidden)
        value, _ = self.attention(
            value,
            value,
            value,
            key_padding_mask=~mask,
            need_weights=False,
        )
        hidden = (hidden + value).masked_fill(~mask[..., None], 0)
        value = self.ffn(self.ffn_norm(hidden))
        return (hidden + value).masked_fill(~mask[..., None], 0)


def _sinusoidal_positions(hidden: Tensor) -> Tensor:
    frames, dimensions = hidden.size(1), hidden.size(2)
    positions = torch.arange(frames, device=hidden.device, dtype=torch.float32)[:, None]
    dimensions_index = torch.arange(
        0,
        dimensions,
        2,
        device=hidden.device,
        dtype=torch.float32,
    )
    frequencies = torch.exp(-math.log(10_000.0) * dimensions_index / dimensions)
    angles = positions * frequencies[None]
    encoding = torch.zeros(frames, dimensions, device=hidden.device, dtype=torch.float32)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, : dimensions // 2].cos()
    return encoding.to(dtype=hidden.dtype)[None]


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
