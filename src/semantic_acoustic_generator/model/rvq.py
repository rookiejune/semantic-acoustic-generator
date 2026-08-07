from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from anytrain.loss import PackedCodebookLogits
from anytrain.module.qwen import top_p_filter
from torch import nn

from semantic_acoustic_generator._tensor import is_signed_integer_dtype

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
        used_sizes = sizes[:-1]
        self.codebook_embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in used_sizes
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
            for _ in used_sizes
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
        self.decoder.embed_tokens = nn.Identity()
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


class FactorDepthPredictor(nn.Module):
    """Frame-parallel, stage-autoregressive predictor for paired factor codes.

    Factor codebooks use the fixed order ``(stage_0_a, stage_0_b, stage_1_a,
    stage_1_b, ...)``. The two factors in one stage are predicted from the same
    depth hidden state, while each later stage consumes both ground-truth factors
    from the preceding stage during training.
    """

    def __init__(
        self,
        condition_dim: int,
        factor_codebooks: Sequence[Tensor],
        *,
        hidden_dim: int | None = None,
        layers: int = 4,
        heads: int = 8,
        ffn_ratio: int = 4,
        recurrent: bool = False,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")
        if layers <= 0 or heads <= 0 or ffn_ratio <= 0:
            raise ValueError("predictor depth, heads, and FFN ratio must be positive.")
        hidden_dim = condition_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("predictor hidden dimension must be positive.")
        if not isinstance(recurrent, bool):
            raise TypeError("recurrent must be a boolean.")

        values = tuple(factor_codebooks)
        _validate_factor_codebooks(values)
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.recurrent = recurrent
        self.stages = len(values) // 2
        self.factors = len(values)
        self.factor_sizes = tuple(value.size(0) for value in values)
        self.factor_dims = tuple(value.size(1) for value in values)
        self._factor_codebook_names = tuple(
            f"factor_codebook_{stage}_{side}"
            for stage in range(self.stages)
            for side in ("a", "b")
        )
        for name, value in zip(self._factor_codebook_names, values):
            setattr(self, name, nn.Buffer(value.detach().clone()))

        self.condition = (
            nn.Identity() if condition_dim == hidden_dim else nn.Linear(condition_dim, hidden_dim)
        )
        self.stage_bos = nn.Parameter(torch.zeros(self.stages, hidden_dim))
        self.previous_stage = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.factor_dims[stage * 2] + self.factor_dims[stage * 2 + 1], hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for stage in range(self.stages - 1)
        )
        self.decoder = (
            None
            if recurrent
            else _qwen3_model(
                hidden_dim=hidden_dim,
                ffn_ratio=ffn_ratio,
                layers=layers,
                attention_heads=_heads(hidden_dim, heads),
            )
        )
        if self.decoder is not None:
            self.decoder.embed_tokens = nn.Identity()
        self.recurrent_blocks = (
            nn.ModuleList(
                _FactorDepthBlock(hidden_dim, ffn_ratio=ffn_ratio) for _ in range(layers)
            )
            if recurrent
            else nn.ModuleList()
        )
        self.recurrent_output_norm = nn.LayerNorm(hidden_dim) if recurrent else nn.Identity()
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, size) for size in self.factor_sizes)

    @property
    def factor_codebooks(self) -> tuple[Tensor, ...]:
        result: list[Tensor] = []
        for name in self._factor_codebook_names:
            value = getattr(self, name, None)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"stored factor codebook {name!r} is missing.")
            result.append(value)
        return tuple(result)

    def _validate_condition(self, condition: Tensor) -> None:
        if condition.dim() != 3 or condition.size(-1) != self.condition_dim:
            raise ValueError("condition must have shape [batch, frame, condition_dim].")

    def _stage_embedding(self, stage: int, pair: Tensor) -> Tensor:
        codebook_a = self.factor_codebooks[stage * 2]
        codebook_b = self.factor_codebooks[stage * 2 + 1]
        features = torch.cat(
            (
                torch.nn.functional.embedding(pair[..., 0].to(dtype=torch.long), codebook_a),
                torch.nn.functional.embedding(pair[..., 1].to(dtype=torch.long), codebook_b),
            ),
            dim=-1,
        )
        projection = cast(nn.Module, cast(object, self.previous_stage[stage]))
        return projection(features)

    def forward(
        self,
        condition: Tensor,
        factor_targets: Tensor,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, ...]:
        """Return teacher-forced ``[B, F, K_factor]`` logits in fixed factor order."""
        packed, frame_mask, frame_indices = self._forward_packed(
            condition,
            factor_targets,
            mask=mask,
            validate=validate,
        )
        return tuple(
            _scatter(value, frame_mask, frame_indices) for value in packed.logits
        )

    def forward_packed(
        self,
        condition: Tensor,
        factor_targets: Tensor,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> PackedCodebookLogits:
        """Return valid-frame logits and labels without scattering padded frames."""
        packed, _, _ = self._forward_packed(
            condition,
            factor_targets,
            mask=mask,
            validate=validate,
        )
        return packed

    def forward_packed_retargeted(
        self,
        condition: Tensor,
        factor_targets: Tensor,
        targeter: Callable[[int, Tensor], Tensor],
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> PackedCodebookLogits:
        """Train recurrent stages on generated prefixes and recomputed residual labels."""
        if not self.recurrent:
            raise RuntimeError("residual retargeting requires the recurrent depth predictor.")
        if not callable(targeter):
            raise TypeError("factor targeter must be callable.")
        if validate:
            self._validate_condition(condition)
        frame_mask = _frame_mask(condition, mask, validate=validate)
        frame_indices = frame_mask.flatten().nonzero().flatten()
        if validate:
            _validate_factor_targets(
                factor_targets,
                condition,
                self.factor_sizes,
                frame_indices,
            )
        original = factor_targets.flatten(0, 1).index_select(0, frame_indices)
        packed_condition = condition.flatten(0, 1).index_select(0, frame_indices)
        state = self.condition(packed_condition) + self.stage_bos[0]
        logits: list[Tensor] = []
        labels: list[Tensor] = []
        prefix: list[Tensor] = []
        previous_pair: Tensor | None = None
        for stage in range(self.stages):
            if stage > 0:
                if previous_pair is None:
                    raise RuntimeError("retargeted factor predictor lost its generated prefix.")
                state = state + self.stage_bos[stage] + self._stage_embedding(
                    stage - 1,
                    previous_pair,
                )
            for block in self.recurrent_blocks:
                state = block(state)
            head_state = self.recurrent_output_norm(state)
            pair_logits = tuple(
                cast(nn.Linear, cast(object, self.heads[stage * 2 + side]))(head_state)
                for side in range(2)
            )
            logits.extend(pair_logits)
            if stage == 0:
                pair_labels = original[..., :2]
            else:
                pair_labels = targeter(stage, torch.cat(prefix, dim=-1))
                if validate:
                    _validate_retargeted_pair(
                        pair_labels,
                        rows=state.size(0),
                        sizes=self.factor_sizes[stage * 2 : stage * 2 + 2],
                    )
            labels.append(pair_labels)
            previous_pair = torch.stack(
                tuple(value.detach().argmax(dim=-1) for value in pair_logits),
                dim=-1,
            )
            prefix.append(previous_pair)
        return PackedCodebookLogits(
            logits=tuple(logits),
            labels=torch.cat(labels, dim=-1),
        )

    def _forward_packed(
        self,
        condition: Tensor,
        factor_targets: Tensor,
        *,
        mask: Tensor | None,
        validate: bool,
    ) -> tuple[PackedCodebookLogits, Tensor, Tensor]:
        if validate:
            self._validate_condition(condition)
        frame_mask = _frame_mask(condition, mask, validate=validate)
        frame_indices = frame_mask.flatten().nonzero().flatten()
        if validate:
            _validate_factor_targets(
                factor_targets,
                condition,
                self.factor_sizes,
                frame_indices,
            )
        packed_targets = factor_targets.flatten(0, 1).index_select(0, frame_indices)
        packed_condition = condition.flatten(0, 1).index_select(0, frame_indices)
        condition_hidden = self.condition(packed_condition)

        hidden = (
            self._recurrent_hidden(condition_hidden, packed_targets)
            if self.recurrent
            else self._transformer_hidden(condition_hidden, packed_targets)
        )
        logits = tuple(
            cast(nn.Linear, cast(object, self.heads[factor]))(
                hidden[..., factor // 2, :]
            )
            for factor in range(self.factors)
        )
        return (
            PackedCodebookLogits(
                logits=logits,
                labels=packed_targets,
            ),
            frame_mask,
            frame_indices,
        )

    def _transformer_hidden(self, condition_hidden: Tensor, packed_targets: Tensor) -> Tensor:
        inputs = [condition_hidden + self.stage_bos[0]]
        for stage in range(1, self.stages):
            previous = packed_targets[..., (stage - 1) * 2 : stage * 2]
            inputs.append(
                condition_hidden
                + self.stage_bos[stage]
                + self._stage_embedding(stage - 1, previous)
            )
        if self.decoder is None:
            raise RuntimeError("transformer factor depth predictor is missing its decoder.")
        return self.decoder(
            inputs_embeds=torch.stack(inputs, dim=1),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    def _recurrent_hidden(self, condition_hidden: Tensor, packed_targets: Tensor) -> Tensor:
        output: list[Tensor] = []
        state = condition_hidden + self.stage_bos[0]
        for stage in range(self.stages):
            if stage > 0:
                previous = packed_targets[..., (stage - 1) * 2 : stage * 2]
                state = state + self.stage_bos[stage] + self._stage_embedding(
                    stage - 1,
                    previous,
                )
            for block in self.recurrent_blocks:
                state = block(state)
            output.append(self.recurrent_output_norm(state))
        return torch.stack(output, dim=1)

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Greedily generate paired factors stage by stage and frames in parallel."""
        self._validate_condition(condition)
        frame_mask = _frame_mask(condition, mask)
        frame_indices = frame_mask.flatten().nonzero().flatten()
        packed_condition = condition.flatten(0, 1).index_select(0, frame_indices)
        condition_hidden = self.condition(packed_condition)

        output: list[Tensor] = []
        previous_pair: Tensor | None = None
        past_key_values = None
        state: Tensor | None = None
        for stage in range(self.stages):
            if self.recurrent:
                if state is None:
                    state = condition_hidden + self.stage_bos[stage]
                else:
                    if previous_pair is None:
                        raise RuntimeError("recurrent factor depth predictor lost its prefix pair.")
                    state = state + self.stage_bos[stage] + self._stage_embedding(
                        stage - 1,
                        previous_pair,
                    )
                for block in self.recurrent_blocks:
                    state = block(state)
            else:
                decoder_input = condition_hidden + self.stage_bos[stage]
                if previous_pair is not None:
                    decoder_input = decoder_input + self._stage_embedding(
                        stage - 1,
                        previous_pair,
                    )
                if self.decoder is None:
                    raise RuntimeError("transformer factor depth predictor is missing its decoder.")
                state_output = self.decoder(
                    inputs_embeds=decoder_input[:, None],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = state_output.past_key_values
                if past_key_values is None:
                    raise RuntimeError("factor depth predictor did not return a generation cache.")
                state = state_output.last_hidden_state[:, -1]
            if state is None:
                raise RuntimeError("factor depth predictor did not produce a stage state.")
            head_state = self.recurrent_output_norm(state) if self.recurrent else state
            pair = torch.stack(
                tuple(
                    cast(nn.Linear, cast(object, self.heads[stage * 2 + side]))(
                        head_state
                    ).argmax(dim=-1)
                    for side in range(2)
                ),
                dim=-1,
            )
            output.extend(pair.unbind(dim=-1))
            previous_pair = pair
        return _scatter(
            torch.stack(output, dim=-1),
            frame_mask,
            frame_indices,
        )


class _FactorDepthBlock(nn.Module):
    def __init__(self, hidden_dim: int, *, ffn_ratio: int) -> None:
        super().__init__()
        inner_dim = hidden_dim * ffn_ratio
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, inner_dim * 2)
        self.output = nn.Linear(inner_dim, hidden_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.gate(self.norm(hidden)).chunk(2, dim=-1)
        return hidden + self.output(torch.nn.functional.silu(gate) * value)


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
            "Depth-autoregressive acoustic predictors require transformers with Qwen3Model; "
            "install semantic-acoustic-generator[rvq]."
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


def _validate_factor_codebooks(values: Sequence[Tensor]) -> None:
    if len(values) < 2 or len(values) % 2 != 0:
        raise ValueError("factor codebooks must provide one A/B pair per stage.")
    if any(value.dim() != 2 or value.size(0) < 1 or value.size(1) < 1 for value in values):
        raise ValueError("factor codebooks must contain non-empty rank-2 tensors.")
    if any(not value.is_floating_point() or value.is_complex() for value in values):
        raise TypeError("factor codebooks must use a real floating point dtype.")
    if any(
        value.device != values[0].device or value.dtype != values[0].dtype
        for value in values[1:]
    ):
        raise ValueError("factor codebooks must share a device and dtype.")


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


def _validate_factor_targets(
    factor_targets: Tensor,
    condition: Tensor,
    factor_sizes: Sequence[int],
    frame_indices: Tensor,
) -> None:
    if factor_targets.shape != (
        condition.size(0),
        condition.size(1),
        len(factor_sizes),
    ):
        raise ValueError("factor_targets must have shape [B, F, 2 * stages].")
    if not is_signed_integer_dtype(factor_targets.dtype):
        raise TypeError("factor_targets must use a signed integer dtype.")
    packed_targets = factor_targets.flatten(0, 1).index_select(0, frame_indices)
    limits = torch.tensor(factor_sizes, device=packed_targets.device, dtype=torch.long)
    if bool(((packed_targets < 0) | (packed_targets >= limits)).any()):
        raise ValueError("factor_targets contains an ID outside its factor codebook.")


def _validate_retargeted_pair(
    pair: Tensor,
    *,
    rows: int,
    sizes: Sequence[int],
) -> None:
    if pair.shape != (rows, 2):
        raise ValueError("retargeted factor pair must have shape [valid_frames, 2].")
    if not is_signed_integer_dtype(pair.dtype):
        raise TypeError("retargeted factor pair must use a signed integer dtype.")
    limits = torch.tensor(sizes, device=pair.device, dtype=torch.long)
    if bool(((pair < 0) | (pair >= limits)).any()):
        raise ValueError("retargeted factor pair contains an ID outside its codebooks.")


def _heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0 and (hidden_dim // heads) % 2 == 0:
            return heads
    raise RuntimeError("Qwen3 depth predictor requires an even attention head dimension")


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
