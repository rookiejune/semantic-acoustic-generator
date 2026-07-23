from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class LossItem:
    loss: Tensor
    details: dict[str, Tensor] | None = None

    def weighted_mean(self, weight: Tensor) -> Tensor:
        if weight.shape != self.loss.shape:
            raise ValueError("loss weights must align with loss rows.")
        total = weight.sum()
        if bool(total.le(0)):
            raise ValueError("loss weights must contain a positive total.")
        return (self.loss * weight).sum() / total
