from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendConfig:
    name: str = "longcat"
    model_dir: str | None = None
    revision: str | None = None
    local_files_only: bool = True
    allow_unpinned_revision: bool = False

    def __post_init__(self) -> None:
        _non_empty_string(self.name, "backend.name")
        if self.name == "bicodec":
            raise ValueError(
                "BiCodec is a semantic-global codec; semantic-acoustic-generator "
                "supports only frame-aligned semantic-acoustic backends."
            )
        _optional_non_empty_string(self.model_dir, "backend.model_dir")
        _optional_non_empty_string(self.revision, "backend.revision")
        _boolean(self.local_files_only, "backend.local_files_only")
        _boolean(self.allow_unpinned_revision, "backend.allow_unpinned_revision")


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be non-empty.")


def _optional_non_empty_string(value: object, name: str) -> None:
    if value is not None:
        _non_empty_string(value, name)


def _boolean(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
