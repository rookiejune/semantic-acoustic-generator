from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from anytrain.module.idspace import Embedding, Layout
from torch import nn

from semantic_acoustic_codec._tensor import is_signed_integer_dtype
from semantic_acoustic_codec.config import Initialization

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
        self.layout = Layout(semantic=(0, weight.size(0)))
        self.id_embedding = Embedding(self.layout, semantic=self.embedding)
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
    ) -> Tensor:
        if semantic_codes.dim() == 3:
            if semantic_codes.size(-1) != 1:
                raise ValueError("semantic_codes with rank 3 must have shape [B, F, 1].")
            semantic_tokens = semantic_codes[..., 0]
        elif semantic_codes.dim() == 2:
            semantic_tokens = semantic_codes
        else:
            raise ValueError("semantic_codes must have shape [B, F, 1] or [B, T].")
        _validate_semantic_tokens(semantic_tokens, pad_id=self.semantic_pad_id)
        tokens = semantic_tokens.to(dtype=torch.long)
        input_ids = self.layout.to_global("semantic", tokens)
        condition = self.projection(self.id_embedding(input_ids))
        condition = condition.masked_fill(tokens.eq(self.semantic_pad_id)[..., None], 0)
        if spans is None:
            return condition
        return repeat_condition(condition, spans, frames=frames)


class ReferenceConditioner(nn.Module):
    """Pool optional acoustic reference features into frame-level condition space."""

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0 or condition_dim <= 0:
            raise ValueError("feature_dim and condition_dim must be positive.")
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.default_feature = nn.Parameter(torch.zeros(1, feature_dim))
        self.projection = nn.Linear(feature_dim, condition_dim)
        self.norm = nn.LayerNorm(condition_dim)
        self.gate = nn.Parameter(torch.zeros(condition_dim))

    def forward(
        self,
        features: Tensor | None,
        *,
        mask: Tensor | None = None,
        batch_size: int | None = None,
    ) -> Tensor:
        if features is None:
            if batch_size is None or batch_size < 1:
                raise ValueError("batch_size is required for the default reference.")
            features = self.default_feature[None].expand(batch_size, 1, self.feature_dim)
            mask = torch.ones(batch_size, 1, device=features.device, dtype=torch.bool)
        else:
            if features.dim() != 3 or features.size(-1) != self.feature_dim:
                raise ValueError("reference features must have shape [B, F, feature_dim].")
            if batch_size is not None and features.size(0) != batch_size:
                raise ValueError("reference batch must match semantic batch.")
            if mask is None:
                mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
            elif mask.shape != features.shape[:2]:
                raise ValueError("reference mask must align with reference features.")
            elif mask.dtype != torch.bool:
                raise TypeError("reference mask must be boolean.")
            mask = mask.to(device=features.device)
            if not bool(mask.any(dim=1).all()):
                raise ValueError("each reference row must contain at least one valid frame.")

        reference = self.projection(features)
        weights = mask[..., None].to(dtype=reference.dtype)
        pooled = (reference * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        pooled = self.norm(pooled)
        return pooled[:, None] * torch.tanh(self.gate)[None, None]


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
        raise ValueError("semantic_codes contain an ID outside the semantic codebook and pad token.")


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
