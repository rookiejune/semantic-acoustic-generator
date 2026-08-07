"""Pure assembly of the shared semantic backbone and one acoustic output head."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from semantic_acoustic_generator.model.backbone import QwenBackbone
from semantic_acoustic_generator.model.generator import AcousticHead

if TYPE_CHECKING:
    from torch import Tensor


class AcousticGeneratorModel(nn.Module):
    def __init__(self, backbone: QwenBackbone, head: AcousticHead) -> None:
        super().__init__()
        if not isinstance(backbone, QwenBackbone):
            raise TypeError("backbone must be a QwenBackbone.")
        if not isinstance(head, AcousticHead):
            raise TypeError("head must be an AcousticHead.")
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        return self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            validate=validate,
        )

    def condition(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        validate: bool = True,
    ) -> Tensor:
        return self(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            validate=validate,
        )


__all__ = ["AcousticGeneratorModel"]
