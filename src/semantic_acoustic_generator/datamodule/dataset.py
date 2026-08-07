from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anydataset.types import Role
from anytrain.codec import SemanticAcousticCodes
from torch.utils.data import Dataset

from semantic_acoustic_generator._compat import StrEnum, auto
from semantic_acoustic_generator.types import PairMetadata


class DatasetName(StrEnum):
    QWEN = auto()


class Pairing(StrEnum):
    NONE = auto()
    CROSS_TEXT = auto()


class Overlong(StrEnum):
    ERROR = auto()
    FILTER = auto()
    TRUNCATE = auto()


@dataclass(frozen=True)
class GeneratorSample:
    """Source-neutral codec units consumed by generator collation."""

    target: SemanticAcousticCodes
    reference: SemanticAcousticCodes | None = None
    metadata: PairMetadata | None = None

    def __post_init__(self) -> None:
        if (self.reference is None) != (self.metadata is None):
            raise ValueError("reference units and pair metadata must be provided together.")

    @property
    def raw_length(self) -> int:
        target = self.target.semantic.size(0)
        if self.reference is None:
            return target
        return max(target, self.reference.semantic.size(0))

    def pair(self) -> tuple[SemanticAcousticCodes, PairMetadata]:
        if self.reference is None or self.metadata is None:
            raise RuntimeError("paired generator sample requires reference units and metadata.")
        return self.reference, self.metadata


def build_dataset(
    name: DatasetName | str,
    *,
    pairing: Pairing | str,
    codec: str,
    root: str | None,
    split: str,
    role: Role,
    speaker_id: str,
    sample_limit: int | None,
) -> Dataset[GeneratorSample]:
    resolved = DatasetName(name)
    if resolved is DatasetName.QWEN:
        from semantic_acoustic_generator.datamodule.qwen import qwen_dataset

        return qwen_dataset(
            codec=codec,
            root=None if root is None else Path(root).expanduser(),
            split=split,
            role=role,
            speaker_id=speaker_id,
            pairing=Pairing(pairing),
            sample_limit=sample_limit,
        )
    raise ValueError(f"unsupported generator dataset: {resolved!r}.")


__all__ = [
    "DatasetName",
    "GeneratorSample",
    "Overlong",
    "Pairing",
    "build_dataset",
]
