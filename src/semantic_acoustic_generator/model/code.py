"""Discrete acoustic-code generator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from anytrain.loss import MaskedCodebookCrossEntropyLoss
from anytrain.module.qwen import QwenMTPCodebookPredictor

from semantic_acoustic_generator.config import DecoderConfig, Route, RVQPredictor
from semantic_acoustic_generator.model.generator import (
    AcousticUnitGenerator,
    DecoderLoss,
    aligned_condition,
)
from semantic_acoustic_generator.model.rvq import AcousticRVQDecoder

if TYPE_CHECKING:
    from torch import Tensor

    from semantic_acoustic_generator.types import GeneratorBatch


class RVQCodeGenerator(AcousticUnitGenerator):
    route = Route.RVQ

    def __init__(
        self,
        condition_dim: int,
        codebook_sizes: tuple[int, ...],
        config: DecoderConfig,
    ) -> None:
        super().__init__()
        if not codebook_sizes:
            raise ValueError("RVQ route requires acoustic codebooks.")
        self.predictor = config.rvq_predictor
        if config.rvq_predictor is RVQPredictor.CODEBOOK_AR:
            self.core: AcousticRVQDecoder | QwenMTPCodebookPredictor = AcousticRVQDecoder(
                condition_dim,
                len(codebook_sizes),
                codebook_sizes,
                hidden_dim=config.hidden_dim,
                layers=config.layers,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
            )
        elif config.rvq_predictor is RVQPredictor.MTP:
            self.core = QwenMTPCodebookPredictor(
                condition_dim,
                len(codebook_sizes),
                codebook_sizes,
                hidden_dim=config.hidden_dim,
                layers=config.layers,
                heads=config.heads,
                ffn_ratio=config.ffn_ratio,
                mtp_layers=config.mtp_layers,
                mtp_heads=config.mtp_heads,
            )
        else:
            raise AssertionError(f"unsupported RVQ predictor: {config.rvq_predictor}")
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
            include_details=False,
        )

    def code_loss_from_condition(
        self,
        condition: Tensor,
        target_mask: Tensor,
        *,
        target_codes: Tensor,
        include_top1: bool = False,
        validate: bool = True,
        include_details: bool = True,
    ) -> DecoderLoss:
        if isinstance(self.core, AcousticRVQDecoder):
            packed = self.core.forward_packed(
                condition,
                target_codes,
                mask=target_mask,
                validate=validate,
            )
            item = self.rvq_loss.forward_packed(
                packed,
                include_top1=include_top1,
                validate=False,
                include_details=include_details,
            )
        else:
            item = self.rvq_loss(
                self.core(condition, target_codes, mask=target_mask, validate=validate),
                target_codes,
                target_mask,
                include_top1=include_top1,
                validate=validate,
                include_details=include_details,
            )
        return DecoderLoss(loss=item.loss.mean(), items={"rvq": item}, primary="rvq")


__all__ = ["RVQCodeGenerator"]
