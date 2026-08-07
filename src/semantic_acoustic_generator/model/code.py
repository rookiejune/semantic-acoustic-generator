"""Discrete acoustic-code generator implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from anytrain.loss import MaskedCodebookCrossEntropyLoss
from anytrain.module.qwen import top_p_filter
from torch import nn
from torch.nn import functional as F

from semantic_acoustic_generator.config import (
    FactorPredictor,
    HeadConfig,
    Initialization,
    Route,
)
from semantic_acoustic_generator.model.condition import matched_random_weight
from semantic_acoustic_generator.model.generator import (
    AcousticHead,
    DecoderLoss,
    aligned_condition,
)

if TYPE_CHECKING:
    from torch import Tensor

    from semantic_acoustic_generator.types import GeneratorBatch


class RVQCodeGenerator(AcousticHead):
    route = Route.RVQ

    def __init__(
        self,
        condition_dim: int,
        codebook_sizes: tuple[int, ...],
        config: HeadConfig,
        *,
        factor_codebooks: tuple[Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        if not codebook_sizes:
            raise ValueError("RVQ route requires acoustic codebooks.")
        if factor_codebooks is None:
            raise ValueError("RVQ route requires AGRVQ factor codebook pairs.")
        self.core = AGRVQPredictor(
            condition_dim,
            codebook_sizes,
            factor_codebooks,
            dependency=config.factor_predictor,
            initialization=config.codebook_initialization,
            seed=config.seed,
            temperature=config.anchor_factor_temperature,
        )
        self.rvq_loss = MaskedCodebookCrossEntropyLoss()

    @torch.no_grad()
    def sample_acoustic_codes(
        self,
        condition: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        target_condition, target_mask = aligned_condition(condition, mask)
        return self.core.generate(
            target_condition,
            mask=target_mask,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )

    def loss(
        self,
        batch: GeneratorBatch,
        condition: Tensor,
    ) -> DecoderLoss:
        labels = batch.acoustic_codes
        target_mask = batch.acoustic_mask
        target_condition, _ = aligned_condition(
            condition,
            batch.mask,
            target_mask=target_mask,
            validate=False,
        )
        return self.code_loss_from_condition(
            target_condition,
            target_mask,
            target_codes=labels,
            validate=False,
        )

    def code_loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_codes: Tensor,
        include_top1: bool = False,
        validate: bool = True,
    ) -> DecoderLoss:
        safe_targets = target_codes.masked_fill(~target_mask[..., None], 0)
        factor_targets = self.core.unpack(safe_targets, validate=validate)
        loss = self.rvq_loss(
            self.core(
                condition,
                safe_targets,
                mask=target_mask,
                validate=validate,
            ),
            factor_targets,
            target_mask,
            include_top1=include_top1,
            validate=False,
        )
        return DecoderLoss(loss=loss, losses={"rvq": loss}, primary="rvq")


class AGRVQPredictor(nn.Module):
    """Frame-parallel residual stages with two parallel factor heads per stage."""

    def __init__(
        self,
        condition_dim: int,
        composite_sizes: Sequence[int],
        factor_codebooks: Sequence[Tensor],
        *,
        dependency: FactorPredictor,
        initialization: Initialization,
        seed: int,
        temperature: float,
    ) -> None:
        super().__init__()
        values = tuple(factor_codebooks)
        sizes = tuple(int(value) for value in composite_sizes)
        if len(values) != len(sizes) * 2 or not sizes:
            raise ValueError("AGRVQ requires one A/B factor-codebook pair per residual stage.")
        if not isinstance(dependency, FactorPredictor):
            raise TypeError("AGRVQ dependency must be a FactorPredictor.")
        if not isinstance(initialization, Initialization):
            raise TypeError("AGRVQ initialization must be an Initialization.")
        if temperature <= 0:
            raise ValueError("AGRVQ classifier temperature must be positive.")
        _validate_factor_codebooks(values)
        factor_sizes = tuple(value.size(0) for value in values)
        for stage, composite in enumerate(sizes):
            if composite != factor_sizes[stage * 2] * factor_sizes[stage * 2 + 1]:
                raise ValueError("AGRVQ factor sizes must multiply to each composite codebook size.")

        self.condition_dim = condition_dim
        self.stages = len(sizes)
        self.composite_sizes = sizes
        self.factor_sizes = factor_sizes
        self.dependency = dependency
        self.temperature = float(temperature)
        self.classifiers = nn.ModuleList(
            _FactorClassifier(
                condition_dim,
                value,
                initialization=initialization,
                seed=seed + index,
                temperature=temperature,
            )
            for index, value in enumerate(values)
        )
        self.stage_bos = nn.Parameter(torch.zeros(self.stages, condition_dim))
        self.previous_stage = nn.ModuleList(
            nn.Sequential(
                nn.Linear(
                    values[stage * 2].size(1) + values[stage * 2 + 1].size(1),
                    condition_dim,
                ),
                nn.LayerNorm(condition_dim),
            )
            for stage in range(self.stages - 1)
        )
        self.depth_blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(condition_dim),
                nn.Linear(condition_dim, condition_dim * 2),
                nn.SiLU(),
                nn.Linear(condition_dim * 2, condition_dim),
            )
            for _ in range(self.stages)
        )
        self.recurrent = (
            nn.GRUCell(condition_dim, condition_dim)
            if dependency is FactorPredictor.DEPTH_RECURRENT
            else None
        )

    def forward(
        self,
        condition: Tensor,
        target_codes: Tensor,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, ...]:
        frame_mask = _frame_mask(condition, mask, validate=validate)
        targets = self.unpack(
            target_codes.masked_fill(~frame_mask[..., None], 0),
            validate=validate,
        )
        hidden = condition
        output: list[Tensor] = []
        for stage in range(self.stages):
            state = self._stage_hidden(hidden, stage)
            output.extend(self._logits(state, stage))
            if stage + 1 < self.stages:
                pair = targets[..., stage * 2 : stage * 2 + 2]
                hidden = self._next_hidden(condition, state, stage, pair)
        return tuple(value.masked_fill(~frame_mask[..., None], 0) for value in output)

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
        frame_mask = _frame_mask(condition, mask, validate=True)
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1].")
        hidden = condition
        factors: list[Tensor] = []
        for stage in range(self.stages):
            state = self._stage_hidden(hidden, stage)
            pair = torch.stack(
                tuple(
                    _sample(
                        logits,
                        temperature=temperature,
                        top_p=top_p,
                        generator=generator,
                    )
                    for logits in self._logits(state, stage)
                ),
                dim=-1,
            )
            factors.extend(pair.unbind(dim=-1))
            if stage + 1 < self.stages:
                hidden = self._next_hidden(condition, state, stage, pair)
        packed = self.pack(torch.stack(factors, dim=-1), validate=False)
        return packed.masked_fill(~frame_mask[..., None], 0)

    def unpack(self, codes: Tensor, *, validate: bool = True) -> Tensor:
        if validate:
            if codes.dim() != 3 or codes.size(-1) != self.stages:
                raise ValueError("AGRVQ codes must have shape [batch, frame, stage].")
            if codes.is_floating_point() or codes.is_complex():
                raise TypeError("AGRVQ codes must use an integer dtype.")
            limits = torch.tensor(self.composite_sizes, device=codes.device, dtype=torch.long)
            if bool(((codes < 0) | (codes >= limits)).any()):
                raise ValueError("AGRVQ codes contain an ID outside the composite codebook.")
        factors: list[Tensor] = []
        for stage in range(self.stages):
            size_b = self.factor_sizes[stage * 2 + 1]
            composite = codes[..., stage]
            factors.extend(
                (
                    torch.div(composite, size_b, rounding_mode="floor"),
                    composite.remainder(size_b),
                )
            )
        return torch.stack(factors, dim=-1)

    def pack(self, factors: Tensor, *, validate: bool = True) -> Tensor:
        if validate:
            if factors.dim() != 3 or factors.size(-1) != self.stages * 2:
                raise ValueError("AGRVQ factors must have shape [batch, frame, 2 * stage].")
            limits = torch.tensor(self.factor_sizes, device=factors.device, dtype=torch.long)
            if bool(((factors < 0) | (factors >= limits)).any()):
                raise ValueError("AGRVQ factors contain an ID outside the factor codebook.")
        return torch.stack(
            tuple(
                factors[..., stage * 2] * self.factor_sizes[stage * 2 + 1]
                + factors[..., stage * 2 + 1]
                for stage in range(self.stages)
            ),
            dim=-1,
        )

    def _stage_hidden(self, hidden: Tensor, stage: int) -> Tensor:
        value = hidden + self.stage_bos[stage]
        block = cast(nn.Module, cast(object, self.depth_blocks[stage]))
        return value + block(value)

    def _logits(self, hidden: Tensor, stage: int) -> tuple[Tensor, Tensor]:
        classifier_a = cast(_FactorClassifier, cast(object, self.classifiers[stage * 2]))
        classifier_b = cast(_FactorClassifier, cast(object, self.classifiers[stage * 2 + 1]))
        return classifier_a(hidden), classifier_b(hidden)

    def _next_hidden(
        self,
        condition: Tensor,
        hidden: Tensor,
        stage: int,
        pair: Tensor,
    ) -> Tensor:
        if self.dependency is FactorPredictor.PARALLEL:
            return condition
        classifier_a = cast(_FactorClassifier, cast(object, self.classifiers[stage * 2]))
        classifier_b = cast(_FactorClassifier, cast(object, self.classifiers[stage * 2 + 1]))
        embedded = torch.cat(
            (
                classifier_a.embedding(pair[..., 0]),
                classifier_b.embedding(pair[..., 1]),
            ),
            dim=-1,
        )
        projection = cast(nn.Module, cast(object, self.previous_stage[stage]))
        value = condition + projection(embedded)
        if self.recurrent is None:
            return value
        flat = self.recurrent(value.flatten(0, 1), hidden.flatten(0, 1))
        return flat.view_as(hidden)


class _FactorClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        codebook: Tensor,
        *,
        initialization: Initialization,
        seed: int,
        temperature: float,
    ) -> None:
        super().__init__()
        weight = (
            codebook.detach().clone()
            if initialization is Initialization.CODEC
            else matched_random_weight(codebook.detach(), seed=seed)
        )
        self.codebook = nn.Parameter(weight)
        self.projection = (
            nn.Identity()
            if hidden_dim == codebook.size(1)
            else nn.Linear(hidden_dim, codebook.size(1))
        )
        self.temperature = float(temperature)

    def forward(self, hidden: Tensor) -> Tensor:
        projected = F.normalize(self.projection(hidden).float(), dim=-1)
        codebook = F.normalize(self.codebook.float(), dim=-1)
        return (projected @ codebook.transpose(0, 1)).to(dtype=hidden.dtype) / self.temperature

    def embedding(self, codes: Tensor) -> Tensor:
        return F.embedding(codes.to(dtype=torch.long), self.codebook)


def _validate_factor_codebooks(values: Sequence[Tensor]) -> None:
    if any(value.dim() != 2 or min(value.shape) <= 0 for value in values):
        raise ValueError("AGRVQ factor codebooks must contain non-empty rank-2 tensors.")
    if any(not value.is_floating_point() or value.is_complex() for value in values):
        raise TypeError("AGRVQ factor codebooks must use a real floating dtype.")
    if any(value.device != values[0].device or value.dtype != values[0].dtype for value in values):
        raise ValueError("AGRVQ factor codebooks must share device and dtype.")


def _frame_mask(condition: Tensor, mask: Tensor | None, *, validate: bool) -> Tensor:
    if validate and condition.dim() != 3:
        raise ValueError("AGRVQ condition must have shape [batch, frame, hidden].")
    if mask is None:
        return torch.ones(condition.shape[:2], device=condition.device, dtype=torch.bool)
    if validate and (mask.shape != condition.shape[:2] or mask.dtype != torch.bool):
        raise ValueError("AGRVQ mask must be boolean and align with condition frames.")
    return mask.to(device=condition.device)


def _sample(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> Tensor:
    work = logits / temperature
    if top_p < 1.0:
        work = top_p_filter(work, top_p)
    shape = work.shape[:-1]
    sampled = torch.multinomial(
        work.reshape(-1, work.size(-1)).softmax(dim=-1),
        1,
        generator=generator,
    )
    return sampled.reshape(shape)


__all__ = ["AGRVQPredictor", "RVQCodeGenerator"]
