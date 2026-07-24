from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback

from semantic_acoustic_codec.pl_module.semantic import SemanticCodecModule

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


class ArtifactExport(Callback):
    def __init__(self, output_dir: str | Path) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        module = _module(pl_module)
        module.export_artifact(self.output_dir / "artifact")


def _module(module: LightningModule) -> SemanticCodecModule:
    if not isinstance(module, SemanticCodecModule):
        raise TypeError("ArtifactExport requires a SemanticCodecModule.")
    return module


__all__ = ["ArtifactExport"]
