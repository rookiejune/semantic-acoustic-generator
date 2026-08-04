from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

from semantic_acoustic_generator.training import run

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
