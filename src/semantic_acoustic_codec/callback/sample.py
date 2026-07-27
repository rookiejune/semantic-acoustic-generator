from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes, masked_acoustic_features
from anytrain.lightning.experiment import audio as experiment_audio
from anytrain.lightning.experiment import scalar as experiment_scalar
from lightning.pytorch.callbacks import Callback

from semantic_acoustic_codec.runtime import SemanticCodecRuntime
from semantic_acoustic_codec.types import SemanticCodecBatch

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor

    from semantic_acoustic_codec.pl_module.semantic import SemanticCodecModule


@dataclass(frozen=True)
class SampleLogConfig:
    every_n_train_steps: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.every_n_train_steps, bool) or not isinstance(
            self.every_n_train_steps, int
        ):
            raise TypeError("sample.every_n_train_steps must be an integer.")
        if self.every_n_train_steps <= 0:
            raise ValueError("sample.every_n_train_steps must be positive.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("sample.seed must be an integer.")


class SampleLogger(Callback):
    def __init__(
        self,
        output_dir: str | Path,
        fixed_sample: SemanticCodecBatch,
        config: SampleLogConfig | None = None,
    ) -> None:
        super().__init__()
        if not fixed_sample.has_reference:
            raise ValueError("SampleLogger requires a fixed target/reference pair.")
        self.output_dir = Path(output_dir)
        self.config = SampleLogConfig() if config is None else config
        self.last_logged_step = 0
        self.fixed_sample = _first_valid_batch(fixed_sample, device=torch.device("cpu"))

    @property
    def state_key(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            f":every={self.config.every_n_train_steps}:seed={self.config.seed}"
        )

    def state_dict(self) -> dict[str, int]:
        return {"last_logged_step": self.last_logged_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        value = state_dict.get("last_logged_step", 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("SampleLogger checkpoint state last_logged_step must be an integer.")
        if value < 0:
            raise ValueError("SampleLogger checkpoint state last_logged_step must be non-negative.")
        self.last_logged_step = value

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step == self.last_logged_step:
            return
        if step % self.config.every_n_train_steps != 0:
            return
        module = _module(pl_module)
        sample = _to(self.fixed_sample, module.device)
        was_training = module.training
        module.eval()
        try:
            event, audio, scalars = _sample(
                module,
                sample,
                step=step,
                seed=self.config.seed,
            )
        finally:
            module.train(was_training)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _append_json(self.output_dir / "sample_metrics.json", event)
        _add_audio(trainer, audio, step=step, sample_rate=int(module.backend.sample_rate))
        _add_scalars(trainer, scalars, step=step)
        self.last_logged_step = step


@torch.no_grad()
def _sample(
    module: SemanticCodecModule,
    sample: SemanticCodecBatch,
    *,
    step: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, float]]:
    runtime = SemanticCodecRuntime(module.support, module.backend)
    without_features = _generated_features(
        runtime,
        module,
        sample,
        with_reference=False,
        generator=_generator(module.device, seed),
    )
    without_audio = runtime.decode_features(sample.semantic_codes, without_features)
    target_features = _target_features(module, sample)
    without_mse = _feature_mse(without_features, target_features, sample)
    target_audio = _reconstruct_target(module, sample)
    with_features = _generated_features(
        runtime,
        module,
        sample,
        with_reference=True,
        generator=_generator(module.device, seed),
    )
    with_audio = runtime.decode_features(sample.semantic_codes, with_features)
    with_mse = _feature_mse(with_features, target_features, sample)
    reference_audio = _reconstruct_reference(module, sample)
    passthrough_audio = (
        _reference_passthrough(module, sample)
        if sample.acoustic_layout is AcousticLayout.FIXED_LENGTH
        else None
    )
    reference_gain = without_mse - with_mse
    without_stats = _audio_stats(without_audio)
    target_stats = _audio_stats(target_audio)
    event: dict[str, Any] = {
        "step": step,
        "metadata": asdict(sample.metadata[0]),
        "feature_mse_without_reference": without_mse,
        "feature_mse_with_reference": with_mse,
        "reference_gain": reference_gain,
        "generated_without_reference": without_stats,
        "generated_with_reference": _audio_stats(with_audio),
        "full_reconstruction": target_stats,
        "reference_full_reconstruction": _audio_stats(reference_audio),
        "reference_token_passthrough": (
            None if passthrough_audio is None else _audio_stats(passthrough_audio)
        ),
    }
    audio = {
        "sample/generated_without_reference": _audio_tensor(without_audio),
        "sample/generated_with_reference": _audio_tensor(with_audio),
        "sample/reconstruction_full_units": _audio_tensor(target_audio),
        "sample/reference_full_units": _audio_tensor(reference_audio),
    }
    if passthrough_audio is not None:
        audio["sample/reference_token_passthrough"] = _audio_tensor(passthrough_audio)
    scalars = {
        "sample/feature_mse_without_reference": without_mse,
        "sample/feature_mse_with_reference": with_mse,
        "sample/reference_gain": reference_gain,
    }
    return event, audio, scalars


def _first_valid_batch(batch: SemanticCodecBatch, *, device: torch.device) -> SemanticCodecBatch:
    semantic_mask = batch.mask[0]
    acoustic_mask = _acoustic_mask(batch)[0]
    semantic = batch.semantic_codes[0:1, semantic_mask].to(device=device)
    acoustic = batch.acoustic_codes[0:1, acoustic_mask].to(device=device)
    reference_semantic: Tensor | None = None
    reference_acoustic: Tensor | None = None
    reference_mask: Tensor | None = None
    reference_acoustic_mask: Tensor | None = None
    metadata = ()
    if batch.has_reference:
        source_semantic = _reference_semantic(batch)
        source_acoustic = _reference_acoustic(batch)
        source_mask = _reference_mask(batch)[0]
        source_acoustic_mask = _reference_acoustic_mask(batch)[0]
        reference_semantic = source_semantic[0:1, source_mask].to(device=device)
        reference_acoustic = source_acoustic[0:1, source_acoustic_mask].to(device=device)
        reference_mask = torch.ones(
            1,
            reference_semantic.size(1),
            dtype=torch.bool,
            device=device,
        )
        reference_acoustic_mask = torch.ones(
            1,
            reference_acoustic.size(1),
            dtype=torch.bool,
            device=device,
        )
        metadata = (batch.metadata[0],)
    return SemanticCodecBatch(
        semantic_codes=semantic,
        acoustic_codes=acoustic,
        mask=torch.ones(1, semantic.size(1), dtype=torch.bool, device=device),
        semantic_pad_id=batch.semantic_pad_id,
        acoustic_pad_ids=batch.acoustic_pad_ids,
        acoustic_mask=torch.ones(1, acoustic.size(1), dtype=torch.bool, device=device),
        acoustic_layout=batch.acoustic_layout,
        reference_semantic_codes=reference_semantic,
        reference_acoustic_codes=reference_acoustic,
        reference_mask=reference_mask,
        reference_acoustic_mask=reference_acoustic_mask,
        metadata=metadata,
    )


def _to(batch: SemanticCodecBatch, device: torch.device) -> SemanticCodecBatch:
    return SemanticCodecBatch(
        semantic_codes=batch.semantic_codes.to(device=device),
        acoustic_codes=batch.acoustic_codes.to(device=device),
        mask=batch.mask.to(device=device),
        semantic_pad_id=batch.semantic_pad_id,
        acoustic_pad_ids=batch.acoustic_pad_ids,
        acoustic_mask=_acoustic_mask(batch).to(device=device),
        acoustic_layout=batch.acoustic_layout,
        reference_semantic_codes=_optional_to(batch.reference_semantic_codes, device),
        reference_acoustic_codes=_optional_to(batch.reference_acoustic_codes, device),
        reference_mask=_optional_to(batch.reference_mask, device),
        reference_acoustic_mask=_optional_to(batch.reference_acoustic_mask, device),
        metadata=batch.metadata,
    )


def _generated_features(
    runtime: SemanticCodecRuntime,
    module: SemanticCodecModule,
    batch: SemanticCodecBatch,
    *,
    with_reference: bool,
    generator: torch.Generator,
) -> Tensor:
    reference_features: Tensor | None = None
    reference_mask: Tensor | None = None
    if with_reference:
        reference_features = _reference_features(module, batch)
        reference_mask = _reference_acoustic_mask(batch)
    return runtime.sample_features(
        batch.semantic_codes,
        mask=batch.mask,
        reference_features=reference_features,
        reference_mask=reference_mask,
        generator=generator,
    )


def _target_features(module: SemanticCodecModule, batch: SemanticCodecBatch) -> Tensor:
    return masked_acoustic_features(module.backend, batch.acoustic_codes, _acoustic_mask(batch))


def _reference_features(module: SemanticCodecModule, batch: SemanticCodecBatch) -> Tensor:
    acoustic = _reference_acoustic(batch)
    acoustic_mask = _reference_acoustic_mask(batch)
    return masked_acoustic_features(module.backend, acoustic, acoustic_mask)


def _feature_mse(generated: Tensor, target: Tensor, batch: SemanticCodecBatch) -> float:
    mask = _acoustic_mask(batch)
    if generated.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError(
            "sample feature tensors must align: "
            f"generated={tuple(generated.shape)}, target={tuple(target.shape)}, mask={tuple(mask.shape)}"
        )
    diff = (generated.float() - target.float()).pow(2)
    value = diff[mask].mean()
    if not bool(torch.isfinite(value).detach().cpu()):
        raise ValueError("sample feature_mse must be finite.")
    return float(value.detach().cpu())


def _reconstruct_target(module: SemanticCodecModule, batch: SemanticCodecBatch) -> Tensor:
    codes = SemanticAcousticCodes(semantic=batch.semantic_codes, acoustic=batch.acoustic_codes)
    return module.backend.detokenize(codes)


def _reconstruct_reference(module: SemanticCodecModule, batch: SemanticCodecBatch) -> Tensor:
    codes = SemanticAcousticCodes(
        semantic=_reference_semantic(batch),
        acoustic=_reference_acoustic(batch),
    )
    return module.backend.detokenize(codes)


def _reference_passthrough(module: SemanticCodecModule, batch: SemanticCodecBatch) -> Tensor:
    # Fixed acoustic slots can be paired with the target semantic sequence without axis alignment.
    codes = SemanticAcousticCodes(
        semantic=batch.semantic_codes,
        acoustic=_reference_acoustic(batch),
    )
    return module.backend.detokenize(codes)


def _audio_stats(value: Tensor) -> dict[str, Any]:
    audio = _audio_tensor(value)
    if not bool(torch.isfinite(audio).all().detach().cpu()):
        raise ValueError("sample audio must be finite.")
    rms = audio.float().pow(2).mean().sqrt()
    if not bool(torch.isfinite(rms).detach().cpu()):
        raise ValueError("sample audio RMS must be finite.")
    return {
        "shape": list(value.shape),
        "finite": True,
        "rms": float(rms.detach().cpu()),
    }


def _audio_tensor(value: Tensor) -> Tensor:
    audio = value.detach().float().cpu()
    if audio.dim() == 3:
        audio = audio[0]
    elif audio.dim() == 1:
        audio = audio[None, :]
    if audio.dim() != 2:
        raise ValueError("audio tensors must have shape [B, C, T], [C, T], or [T].")
    return audio


def _generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        generator = torch.Generator()
    return generator.manual_seed(seed)


def _add_audio(
    trainer: Trainer,
    audio: dict[str, Tensor],
    *,
    step: int,
    sample_rate: int,
) -> None:
    experiment = experiment_audio(trainer)
    if experiment is None:
        return
    for tag, value in audio.items():
        experiment.add_audio(tag, value, global_step=step, sample_rate=sample_rate)


def _add_scalars(trainer: Trainer, scalars: dict[str, float], *, step: int) -> None:
    experiment = experiment_scalar(trainer)
    if experiment is None:
        return
    for tag, value in scalars.items():
        experiment.add_scalar(tag, value, global_step=step)


def _append_json(path: Path, event: dict[str, Any]) -> None:
    events: list[dict[str, Any]]
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise TypeError(f"{path} must contain a JSON list.")
        events = [cast(dict[str, Any], item) for item in loaded]
    else:
        events = []
    events.append(event)
    path.write_text(json.dumps(events, indent=2, sort_keys=True), encoding="utf-8")


def _acoustic_mask(batch: SemanticCodecBatch) -> Tensor:
    if batch.acoustic_mask is None:
        raise RuntimeError("SemanticCodecBatch must expose acoustic_mask after validation.")
    return batch.acoustic_mask


def _reference_semantic(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_semantic_codes
    if value is None:
        raise RuntimeError("reference_semantic_codes are required for paired sampling.")
    return value


def _reference_acoustic(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_acoustic_codes
    if value is None:
        raise RuntimeError("reference_acoustic_codes are required for paired sampling.")
    return value


def _reference_mask(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_mask
    if value is None:
        raise RuntimeError("reference_mask is required for paired sampling.")
    return value


def _reference_acoustic_mask(batch: SemanticCodecBatch) -> Tensor:
    value = batch.reference_acoustic_mask
    if value is None:
        raise RuntimeError("reference_acoustic_mask is required for paired sampling.")
    return value


def _optional_to(value: Tensor | None, device: torch.device) -> Tensor | None:
    return None if value is None else value.to(device=device)


def _module(module: LightningModule) -> SemanticCodecModule:
    from semantic_acoustic_codec.pl_module.semantic import SemanticCodecModule

    if not isinstance(module, SemanticCodecModule):
        raise TypeError("SampleLogger requires a SemanticCodecModule.")
    return module


__all__ = ["SampleLogConfig", "SampleLogger"]
