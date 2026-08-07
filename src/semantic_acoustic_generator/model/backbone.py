"""Semantic input embedding and the shared lightweight Qwen backbone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from anytrain.module.qwen import build_qwen3_model
from torch import nn

from semantic_acoustic_generator.config import BackboneConfig
from semantic_acoustic_generator.model.condition import SemanticConditioner

if TYPE_CHECKING:
    from torch import Tensor


class QwenBackbone(nn.Module):
    """Codec-initialized semantic embedding followed by shared Qwen layers.

    ``input_ids`` use the owned semantic embedding. ``inputs_embeds`` follow the
    Hugging Face convention and bypass that embedding entirely.
    """

    def __init__(self, semantic_codebook: Tensor, config: BackboneConfig) -> None:
        super().__init__()
        if not isinstance(config, BackboneConfig):
            raise TypeError("backbone config must be a BackboneConfig.")
        attention_heads = _heads(config.hidden_dim, config.heads)
        self.hidden_dim = config.hidden_dim
        self.layers = config.layers
        self.embedding = SemanticConditioner(
            semantic_codebook,
            condition_dim=config.hidden_dim,
            initialization=config.embedding_initialization,
            seed=config.seed,
        )
        self.core = build_qwen3_model(
            config.hidden_dim,
            config.hidden_dim * config.ffn_ratio,
            num_layers=config.layers,
            num_attention_heads=attention_heads,
            num_key_value_heads=attention_heads,
            head_dim=config.hidden_dim // attention_heads,
            vocab_size=1,
            use_cache=False,
        )
        # The wrapper always supplies inputs_embeds so the Hugging Face placeholder
        # embedding must not introduce an unrelated trainable parameter group.
        self.core.embed_tokens = nn.Identity()

    @property
    def semantic_codebook_size(self) -> int:
        return self.embedding.semantic_codebook_size

    @property
    def semantic_pad_id(self) -> int:
        return self.embedding.semantic_pad_id

    def embed(self, input_ids: Tensor, *, validate: bool = True) -> Tensor:
        return self.embedding(input_ids, validate=validate)

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one of input_ids and inputs_embeds must be provided.")
        if input_ids is not None:
            hidden = self.embed(input_ids, validate=validate)
        else:
            if inputs_embeds is None:
                raise RuntimeError("inputs_embeds unexpectedly missing after input validation.")
            hidden = inputs_embeds
            if validate and (hidden.dim() != 3 or hidden.size(-1) != self.hidden_dim):
                raise ValueError(
                    "inputs_embeds must have shape [batch, semantic_unit, hidden_dim]."
                )
        if attention_mask is None:
            if input_ids is None:
                mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
            else:
                tokens = input_ids[..., 0] if input_ids.dim() == 3 else input_ids
                mask = tokens.ne(self.semantic_pad_id).to(device=hidden.device)
        else:
            if validate and (
                attention_mask.shape != hidden.shape[:2] or attention_mask.dtype != torch.bool
            ):
                raise ValueError("attention_mask must be boolean and align with semantic units.")
            mask = attention_mask.to(device=hidden.device)
        if validate and not bool(mask.any(dim=1).all()):
            raise ValueError("each semantic row must contain at least one valid unit.")
        output = self.core(
            inputs_embeds=hidden,
            attention_mask=mask.to(dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return output.masked_fill(~mask[..., None], 0)


def _heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0 and (hidden_dim // heads) % 2 == 0:
            return heads
    raise ValueError("Qwen hidden_dim must admit an even attention head dimension.")


__all__ = ["QwenBackbone"]
