from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from anytrain.lightning import find_ema_callback
from lightning.pytorch.callbacks import Callback

from semantic_acoustic_generator.pl_module.module import GeneratorModule

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
        path = self.output_dir / "artifact"
        ema = find_ema_callback(trainer)
        if ema is None:
            module.export_artifact(path)
            return
        with ema.average_parameters(module):
            module.export_artifact(path)


def _module(module: LightningModule) -> GeneratorModule:
    if not isinstance(module, GeneratorModule):
        raise TypeError("ArtifactExport requires a GeneratorModule.")
    return module


__all__ = ["ArtifactExport"]
