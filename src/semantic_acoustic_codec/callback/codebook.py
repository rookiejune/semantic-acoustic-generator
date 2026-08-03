from __future__ import annotations

from collections.abc import Mapping

from anytrain.lightning import CodebookUsageLoggerCallback, CodebookUsageMonitor
from lightning import LightningModule

from semantic_acoustic_codec.types import SemanticCodecBatch


class CodebookUsageLogger(CodebookUsageLoggerCallback):
    """Log target semantic and acoustic codebook usage."""

    def __init__(
        self,
        *,
        every_n_steps: int | None = 100,
        tag: str = "codebook",
        log_histograms: bool = False,
    ) -> None:
        super().__init__(
            _monitors,
            every_n_steps=every_n_steps,
            tag=tag,
            log_histograms=log_histograms,
        )


def _monitors(
    outputs: object,
    batch: object,
    pl_module: LightningModule,
) -> Mapping[str, CodebookUsageMonitor]:
    del outputs, pl_module
    if not isinstance(batch, SemanticCodecBatch):
        raise TypeError("CodebookUsageLogger requires a SemanticCodecBatch.")
    semantic_indices = batch.target_semantic_codes[..., 0].masked_fill(
        ~batch.target_mask,
        -1,
    )
    acoustic_mask = batch.target_acoustic_mask.unsqueeze(-1).expand_as(
        batch.target_acoustic_codes
    )
    acoustic_indices = batch.target_acoustic_codes.masked_fill(~acoustic_mask, -1)
    return {
        "semantic": CodebookUsageMonitor(
            indices=semantic_indices,
            codebook_sizes=(batch.semantic_codebook_size,),
            active_codebook_mask=batch.target_mask,
        ),
        "acoustic": CodebookUsageMonitor(
            indices=acoustic_indices,
            codebook_sizes=batch.acoustic_codebook_sizes,
            active_codebook_mask=acoustic_mask,
        ),
    }


__all__ = ["CodebookUsageLogger"]
