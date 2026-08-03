from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from anytrain.loss import PackedCodebookLogits
from anytrain.module.qwen import top_p_filter
from torch import nn

from semantic_acoustic_codec._tensor import is_signed_integer_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


class AcousticRVQDecoder(nn.Module):
    """Frame-parallel, codebook-autoregressive acoustic code predictor.

    Reuses the shared anytrain Qwen3 builder by design. The repository convention
    is to avoid custom Transformer/cache implementations before the Qwen-based
    RVQ route has been validated and a dedicated optimization pass is planned.
    Transformers stay optional until the RVQ route is instantiated.
    """

    def __init__(
        self,
        condition_dim: int,
        codebooks: int,
        codebook_size: int | Sequence[int],
        *,
        codebook_embeddings: Sequence[Tensor] | None = None,
        hidden_dim: int | None = None,
        layers: int = 8,
        heads: int = 8,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or codebooks <= 0:
            raise ValueError("condition_dim and codebooks must be positive.")
        if layers <= 0 or heads <= 0 or ffn_ratio <= 0:
            raise ValueError("decoder depth, heads, and FFN ratio must be positive.")
        hidden_dim = condition_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("decoder hidden dimension must be positive.")
        attention_heads = _heads(hidden_dim, heads)
        sizes = (
            (codebook_size,) * codebooks if isinstance(codebook_size, int) else tuple(codebook_size)
        )
        if len(sizes) != codebooks or any(size <= 0 for size in sizes):
            raise ValueError("codebook_size must provide one positive size per codebook.")
        if codebook_embeddings is not None:
            _validate_embeddings(codebook_embeddings, sizes)
            embedding_dim = codebook_embeddings[0].size(-1)
        else:
            embedding_dim = hidden_dim

        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.codebooks = codebooks
        self.codebook_sizes = sizes
        self.embedding_dim = embedding_dim
        self.codebook_embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in sizes
        )
        if codebook_embeddings is None:
            for module in self.codebook_embeddings:
                embedding = cast(nn.Embedding, cast(object, module))
                nn.init.normal_(embedding.weight, std=embedding_dim**-0.5)
        else:
            with torch.no_grad():
                for index, module in enumerate(self.codebook_embeddings):
                    embedding = cast(nn.Embedding, cast(object, module))
                    embedding.weight.copy_(codebook_embeddings[index])

        self.embedding_projections = nn.ModuleList(
            nn.Identity() if embedding_dim == hidden_dim else nn.Linear(embedding_dim, hidden_dim)
            for _ in range(codebooks)
        )
        self.condition = (
            nn.Identity() if condition_dim == hidden_dim else nn.Linear(condition_dim, hidden_dim)
        )
        self.codebook_bos = nn.Parameter(torch.zeros(codebooks, hidden_dim))
        self.decoder = _qwen3_model(
            hidden_dim=hidden_dim,
            ffn_ratio=ffn_ratio,
            layers=layers,
            attention_heads=attention_heads,
        )
        self.decoder.embed_tokens.requires_grad_(False)
        self.codebook_embeddings[-1].requires_grad_(False)
        self.embedding_projections[-1].requires_grad_(False)
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, size) for size in sizes)

    def _validate_condition(self, condition: Tensor) -> None:
        if condition.dim() != 3 or condition.size(-1) != self.condition_dim:
            raise ValueError("condition must have shape [batch, frame, condition_dim].")

    def _embedding(self, codebook: int, codes: Tensor) -> Tensor:
        embedding = cast(nn.Embedding, cast(object, self.codebook_embeddings[codebook]))
        projection = cast(nn.Module, cast(object, self.embedding_projections[codebook]))
        value = embedding(codes.to(dtype=torch.long))
        return projection(value)

    def forward(
        self,
        condition: Tensor,
        target_acoustic_codes: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, ...]:
        """Return one teacher-forced [B, F, K_q] tensor per codebook."""
        packed, frame_mask, frame_indices = self._forward_packed(
            condition,
            target_acoustic_codes,
            mask=mask,
            validate=validate,
        )
        return tuple(_scatter(value, frame_mask, frame_indices) for value in packed.logits)

    def forward_packed(
        self,
        condition: Tensor,
        target_acoustic_codes: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> PackedCodebookLogits:
        """Return logits, labels, and batch-row indices for valid frames only."""
        packed, _, _ = self._forward_packed(
            condition,
            target_acoustic_codes,
            mask=mask,
            validate=validate,
        )
        return packed

    def _forward_packed(
        self,
        condition: Tensor,
        target_acoustic_codes: Tensor | None,
        *,
        mask: Tensor | None,
        validate: bool,
    ) -> tuple[PackedCodebookLogits, Tensor, Tensor]:
        if validate:
            self._validate_condition(condition)
        frame_mask = _frame_mask(condition, mask, validate=validate)
        frame_indices = frame_mask.flatten().nonzero().flatten()
        if target_acoustic_codes is not None:
            if validate:
                _validate_targets(
                    target_acoustic_codes,
                    condition,
                    self.codebooks,
                    self.codebook_sizes,
                    frame_indices,
                )
            packed_targets = target_acoustic_codes.flatten(0, 1).index_select(
                0,
                frame_indices,
            )
        else:
            packed_targets = None

        packed_condition = condition.flatten(0, 1).index_select(0, frame_indices)
        condition_hidden = self.condition(packed_condition)
        inputs = [condition_hidden + self.codebook_bos[0]]
        for codebook in range(1, self.codebooks):
            if packed_targets is None:
                previous = torch.zeros(
                    condition_hidden.size(0),
                    dtype=torch.long,
                    device=condition.device,
                )
            else:
                previous = packed_targets[..., codebook - 1]
            inputs.append(
                condition_hidden
                + self.codebook_bos[codebook]
                + self._embedding(codebook - 1, previous)
            )
        decoder_input = torch.stack(inputs, dim=1)
        hidden = self.decoder(
            inputs_embeds=decoder_input,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        packed_logits = tuple(
            cast(nn.Linear, cast(object, self.heads[codebook]))(hidden[..., codebook, :])
            for codebook in range(self.codebooks)
        )
        return (
            PackedCodebookLogits(
                logits=packed_logits,
                labels=packed_targets,
                row_indices=frame_indices.div(
                    condition.size(1),
                    rounding_mode="floor",
                ),
                batch_size=condition.size(0),
            ),
            frame_mask,
            frame_indices,
        )

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        self._validate_condition(condition)
        frame_mask = _frame_mask(condition, mask)
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1].")

        packed_condition = condition.flatten(0, 1)[frame_mask.flatten()]
        condition_hidden = self.condition(packed_condition)
        output: list[Tensor] = []
        past_key_values = None
        for codebook in range(self.codebooks):
            decoder_input = condition_hidden + self.codebook_bos[codebook]
            if output:
                decoder_input = decoder_input + self._embedding(codebook - 1, output[-1])
            state_output = self.decoder(
                inputs_embeds=decoder_input[:, None],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = state_output.past_key_values
            if past_key_values is None:
                raise RuntimeError("RVQ decoder did not return a generation cache.")
            state = state_output.last_hidden_state[:, -1]
            head = cast(nn.Linear, cast(object, self.heads[codebook]))
            logits = head(state) / temperature
            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)
            value = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)[:, 0]
            output.append(value)
        return _scatter(torch.stack(output, dim=-1), frame_mask)


def _qwen3_model(
    *,
    hidden_dim: int,
    ffn_ratio: int,
    layers: int,
    attention_heads: int,
) -> nn.Module:
    try:
        from anytrain.module.qwen import build_qwen3_model

        return build_qwen3_model(
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * ffn_ratio,
            num_layers=layers,
            num_attention_heads=attention_heads,
            num_key_value_heads=attention_heads,
            head_dim=hidden_dim // attention_heads,
            vocab_size=1,
            use_cache=True,
        )
    except ImportError as exc:
        raise ImportError(
            "AcousticRVQDecoder requires transformers with Qwen3Model; install semantic-acoustic-codec[rvq]."
        ) from exc


def _validate_embeddings(values: Sequence[Tensor], sizes: Sequence[int]) -> None:
    if len(values) != len(sizes):
        raise ValueError("codebook_embeddings must provide one tensor per codebook.")
    if any(not torch.is_floating_point(value) for value in values):
        raise TypeError("codebook_embeddings must be floating point.")
    if any(value.dim() != 2 for value in values):
        raise ValueError("each codebook embedding must have shape [size_q, dim].")
    if any(value.size(0) != size for value, size in zip(values, sizes)):
        raise ValueError("codebook embeddings must match codebook sizes.")
    embedding_dim = values[0].size(-1)
    if any(value.size(-1) != embedding_dim for value in values):
        raise ValueError("all codebook embeddings must have the same dimension.")


def _validate_targets(
    target_acoustic_codes: Tensor,
    condition: Tensor,
    codebooks: int,
    codebook_sizes: Sequence[int],
    frame_indices: Tensor,
) -> None:
    if target_acoustic_codes.shape != (condition.size(0), condition.size(1), codebooks):
        raise ValueError("target_acoustic_codes must have shape [B, F, codebooks].")
    if not is_signed_integer_dtype(target_acoustic_codes.dtype):
        raise TypeError("target_acoustic_codes must use a signed integer dtype.")
    packed_targets = target_acoustic_codes.flatten(0, 1).index_select(
        0,
        frame_indices,
    )
    limits = torch.tensor(codebook_sizes, device=packed_targets.device, dtype=torch.long)
    if bool(((packed_targets < 0) | (packed_targets >= limits)).any()):
        raise ValueError("target_acoustic_codes contains an ID outside its codebook.")


def _heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0 and (hidden_dim // heads) % 2 == 0:
            return heads
    raise RuntimeError("Qwen3 RVQ decoder requires an even attention head dimension")


def _frame_mask(condition: Tensor, mask: Tensor | None, *, validate: bool = True) -> Tensor:
    if mask is None:
        frame_mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=condition.device)
    else:
        if validate and mask.shape != condition.shape[:2]:
            raise ValueError("acoustic frame mask must align with condition.")
        if validate and mask.dtype != torch.bool:
            raise TypeError("acoustic frame mask must be boolean.")
        if validate and mask.device != condition.device:
            raise ValueError("acoustic frame mask and condition must use the same device.")
        frame_mask = mask
    if validate and (frame_mask.size(0) < 1 or not bool(frame_mask.any(dim=1).all())):
        raise ValueError("each acoustic condition row must contain a valid frame.")
    return frame_mask


def _scatter(
    values: Tensor,
    mask: Tensor,
    frame_indices: Tensor | None = None,
) -> Tensor:
    if frame_indices is None:
        frame_indices = mask.flatten().nonzero().flatten()
    output = values.new_zeros((mask.numel(), *values.shape[1:]))
    output = output.index_copy(0, frame_indices, values)
    return output.unflatten(0, mask.shape)
