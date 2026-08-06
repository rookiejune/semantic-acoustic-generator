from __future__ import annotations

import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from anytrain.codec import SemanticAcousticCodes, masked_acoustic_features
from torch import Tensor

from semantic_acoustic_generator.backend.adapter import (
    LongCatCodebookAdapter,
    LongCatFirstCodebookAdapter,
)
from semantic_acoustic_generator.model.feature import FMFeatureGenerator
from semantic_acoustic_generator.types import GeneratorBatch

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec

    from semantic_acoustic_generator.runtime.semantic import GeneratorRuntime


@dataclass(frozen=True)
class PairedFeatureEvaluation:
    without_reference: Tensor
    with_reference: Tensor
    mse_without_reference: float
    mse_with_reference: float

    @property
    def reference_gain(self) -> float:
        return self.mse_without_reference - self.mse_with_reference


@dataclass(frozen=True)
class FirstCodebookOracleEvaluation:
    audio: dict[str, Tensor]
    metrics: dict[str, object]


@dataclass(frozen=True)
class ArtifactSampleEvaluation:
    audio: dict[str, Tensor]
    metrics: dict[str, float]


@torch.no_grad()
def target_acoustic_features(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    validate: bool = True,
) -> Tensor:
    return masked_acoustic_features(
        backend,
        batch.acoustic_codes,
        batch.acoustic_mask,
        validate=validate,
    )


@torch.no_grad()
def reference_acoustic_condition(
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    validate: bool = True,
) -> tuple[Tensor, Tensor]:
    if not batch.has_reference:
        raise ValueError("reference acoustic condition requires reference codec units.")
    reference = batch.reference
    mask = reference.acoustic_mask.to(device=batch.semantic_codes.device)
    features = masked_acoustic_features(
        backend,
        reference.acoustic_codes,
        mask,
        validate=validate,
    )
    return features, mask


def masked_feature_mse(
    generated: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    name: str,
) -> float:
    if generated.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError(
            f"{name} feature tensors must align: "
            f"generated={tuple(generated.shape)}, target={tuple(target.shape)}, "
            f"mask={tuple(mask.shape)}"
        )
    value = (generated.float() - target.float()).pow(2)[mask].mean()
    if not bool(torch.isfinite(value).detach().cpu()):
        raise ValueError(f"{name} feature MSE must be finite.")
    return float(value.detach().cpu())


@torch.no_grad()
def evaluate_feature_pair(
    runtime: GeneratorRuntime,
    backend: SemanticAcousticCodec,
    batch: GeneratorBatch,
    *,
    seed: int,
    cfg_scale: float | None = None,
    name: str,
) -> PairedFeatureEvaluation:
    if not batch.has_reference:
        raise ValueError(f"{name} evaluation requires reference codec units.")
    target = target_acoustic_features(backend, batch)
    reference, reference_mask = reference_acoustic_condition(backend, batch)
    device = batch.semantic_codes.device
    without = runtime.sample_features(
        batch.semantic_codes,
        mask=batch.mask,
        reference_features=None,
        reference_mask=None,
        generator=seeded_generator(device, seed),
    )
    with_reference = runtime.sample_features(
        batch.semantic_codes,
        mask=batch.mask,
        reference_features=reference,
        reference_mask=reference_mask,
        cfg_scale=cfg_scale,
        generator=seeded_generator(device, seed),
    )
    return PairedFeatureEvaluation(
        without_reference=without,
        with_reference=with_reference,
        mse_without_reference=masked_feature_mse(
            without,
            target,
            batch.acoustic_mask,
            name=name,
        ),
        mse_with_reference=masked_feature_mse(
            with_reference,
            target,
            batch.acoustic_mask,
            name=name,
        ),
    )


@torch.no_grad()
def evaluate_artifact_sample(
    runtime: GeneratorRuntime,
    batch: GeneratorBatch,
    *,
    seed: int,
) -> ArtifactSampleEvaluation:
    """Evaluate one fixed artifact sample, including supported LongCat diagnostics."""
    backend = runtime.backend
    target = target_acoustic_features(backend, batch)
    generated = runtime.sample_features(
        batch.semantic_codes,
        mask=batch.mask,
        generator=seeded_generator(batch.semantic_codes.device, seed),
    )
    semantic = batch.semantic_codes.masked_fill(~batch.mask[..., None], 0)
    acoustic = batch.acoustic_codes.masked_fill(~batch.acoustic_mask[..., None], 0)
    audio = {
        "target_reconstruction": backend.detokenize(
            SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)
        ),
        "generated_without_reference_raw": runtime.decode_features(
            semantic,
            generated,
            mask=batch.mask,
        ),
    }
    metrics = {
        "raw_feature_mse": masked_feature_mse(
            generated,
            target,
            batch.acoustic_mask,
            name="artifact set raw",
        )
    }
    if not isinstance(backend, LongCatCodebookAdapter):
        return ArtifactSampleEvaluation(audio=audio, metrics=metrics)

    audio["selected_codebook_reconstruction"] = runtime.decode_features(
        semantic,
        target,
        mask=batch.mask,
    )
    snapped = backend.snap_features(generated)
    audio["generated_without_reference_snap"] = runtime.decode_features(
        semantic,
        snapped,
        mask=batch.mask,
    )
    metrics["snap_feature_mse"] = masked_feature_mse(
        snapped,
        target,
        batch.acoustic_mask,
        name="artifact set snap",
    )
    predicted = backend.features_to_factor_codes(generated)
    labels = backend.factor_codes(batch.acoustic_codes).to(predicted.device)
    valid = batch.acoustic_mask.to(predicted.device)
    metrics.update(factor_accuracy(predicted, labels, valid))

    feature_generator = runtime.support.generator
    if isinstance(feature_generator, FMFeatureGenerator) and feature_generator.factor_depth is not None:
        condition = runtime.support.condition(batch.semantic_codes, mask=batch.mask)
        teacher_logits = feature_generator.factor_logits(
            condition,
            batch.mask,
            factor_targets=labels,
        )
        teacher = torch.stack(
            tuple(value.argmax(dim=-1) for value in teacher_logits),
            dim=-1,
        )
        metrics.update(factor_accuracy(teacher, labels, valid, prefix="teacher_forced_"))

    if backend.feature_codebooks <= 1:
        return ArtifactSampleEvaluation(audio=audio, metrics=metrics)

    full_target = backend.backend.acoustic_codes_to_features(batch.acoustic_codes)
    audio["generated_stage0_only"] = backend.decode_features(
        semantic,
        generated,
        active_codebooks=1,
    )
    metrics["stage0_projected_mse"] = masked_feature_mse(
        backend.project_features(generated, active_codebooks=1),
        full_target,
        batch.acoustic_mask,
        name="artifact set stage0 projection",
    )
    retargeted = backend.retarget_factor_codes(batch.acoustic_codes, predicted)
    metrics.update(
        factor_accuracy(
            predicted[..., 2:],
            retargeted[..., 2:],
            valid,
            prefix="retargeted_",
            codebook_offset=1,
        )
    )
    retargeted_features = backend.factor_codes_to_features(retargeted)
    audio["generated_residual_oracle"] = runtime.decode_features(
        semantic,
        retargeted_features,
        mask=batch.mask,
    )
    metrics["generated_projected_mse"] = masked_feature_mse(
        backend.project_features(generated),
        full_target,
        batch.acoustic_mask,
        name="artifact set generated projection",
    )
    metrics["residual_oracle_projected_mse"] = masked_feature_mse(
        backend.project_features(retargeted_features),
        full_target,
        batch.acoustic_mask,
        name="artifact set residual oracle projection",
    )
    return ArtifactSampleEvaluation(audio=audio, metrics=metrics)


def factor_accuracy(
    predicted: Tensor,
    labels: Tensor,
    valid: Tensor,
    *,
    prefix: str = "",
    codebook_offset: int = 0,
) -> dict[str, float]:
    accuracy = predicted[valid].eq(labels[valid]).float().mean(dim=0)
    metrics: dict[str, float] = {}
    for factor, value in enumerate(accuracy):
        codebook, local = divmod(factor, 2)
        codebook += codebook_offset
        suffix = "a" if local == 0 else "b"
        name = (
            f"factor_{suffix}_accuracy"
            if codebook == 0
            else f"codebook_{codebook}_factor_{suffix}_accuracy"
        )
        metrics[f"{prefix}{name}"] = float(value.cpu())
    return metrics


def seeded_generator(device: torch.device | str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def waveform_summary(waveform: Tensor, *, sample_rate: int) -> dict[str, Any]:
    audio = waveform.detach().float().cpu()
    finite = bool(torch.isfinite(audio).all())
    if audio.dim() != 3:
        raise ValueError("decoded waveform must have shape [batch, channel, samples].")
    samples = int(audio.size(-1))
    return {
        "finite": finite,
        "seconds": samples / sample_rate,
        "waveform_max": float(audio.max()),
        "waveform_min": float(audio.min()),
        "waveform_rms": float(audio.square().mean().sqrt()),
        "waveform_shape": list(audio.shape),
    }


def write_pcm16_wav(path: Path, waveform: Tensor, *, sample_rate: int) -> None:
    audio = waveform.detach().float().cpu()[0]
    if audio.dim() != 2:
        raise ValueError("decoded waveform must have shape [batch, channel, samples].")
    pcm = (
        audio.clamp(-1, 1)
        .mul(32_767)
        .round()
        .to(torch.int16)
        .transpose(0, 1)
        .contiguous()
        .reshape(-1)
    )
    frames = array("h", pcm.tolist())
    if sys.byteorder == "big":
        frames.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(audio.size(0))
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames.tobytes())


@torch.no_grad()
def evaluate_first_codebook_oracle(
    backend: LongCatFirstCodebookAdapter,
    batch: GeneratorBatch,
    *,
    sigmas: tuple[float, ...],
    seed: int,
) -> FirstCodebookOracleEvaluation:
    if batch.semantic_codes.size(0) != 1:
        raise ValueError("first-codebook oracle requires one sample per batch.")
    if any(value <= 0 for value in sigmas):
        raise ValueError("first-codebook oracle sigmas must be positive.")
    mask = batch.acoustic_mask.to(device=batch.acoustic_codes.device)
    semantic = batch.semantic_codes.masked_fill(~batch.mask[..., None], 0)
    acoustic = batch.acoustic_codes.masked_fill(~mask[..., None], 0)
    target = backend.acoustic_codes_to_features(acoustic)
    native = backend.native_stage0_features(acoustic)
    projected = backend.project_features(target)
    target_factors = backend.factor_codes(acoustic)
    full_codes = SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)
    audio = {
        "full_reconstruction": backend.backend.detokenize(full_codes),
        "stage0_code_reconstruction": backend.backend.decode_features(semantic, native),
        "exact_16d_reconstruction": backend.decode_features(semantic, target),
    }
    metrics: dict[str, object] = {
        "native_projection_max_abs": float((native - projected).abs().max().cpu()),
        "native_projection_mse": float((native.float() - projected.float()).square().mean().cpu()),
        "exact_snap_max_abs": float((backend.snap_features(target) - target).abs().max().cpu()),
        "groups": {},
    }
    group_metrics = metrics["groups"]
    if not isinstance(group_metrics, dict):
        raise RuntimeError("oracle group metrics must be a dict.")
    scale = torch.cat(
        tuple(codebook.float().std(dim=0, correction=0) for codebook in backend.factor_codebooks)
    ).view(1, 1, -1)
    generator = seeded_generator(target.device, seed)
    for sigma in sigmas:
        noise = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        raw = target + noise * scale.to(device=target.device, dtype=target.dtype) * sigma
        raw = raw.masked_fill(~mask.to(device=raw.device)[..., None], 0)
        snapped = backend.snap_features(raw)
        key = _sigma_key(sigma)
        raw_name = f"raw_sigma_{key}"
        snap_name = f"snap_sigma_{key}"
        audio[raw_name] = backend.decode_features(semantic, raw)
        audio[snap_name] = backend.decode_features(semantic, snapped)
        predicted_factors = backend.features_to_factor_codes(raw)
        valid = mask.to(device=predicted_factors.device)
        factor_accuracy = predicted_factors[valid].eq(
            target_factors.to(device=predicted_factors.device)[valid]
        ).float().mean(dim=0)
        group_metrics[raw_name] = {
            "feature_mse": float((raw.float() - target.float())[valid].square().mean().cpu()),
            "factor_a_accuracy": float(factor_accuracy[0].cpu()),
            "factor_b_accuracy": float(factor_accuracy[1].cpu()),
        }
        group_metrics[snap_name] = {
            "feature_mse": float(
                (snapped.float() - target.float())[valid].square().mean().cpu()
            ),
            "factor_a_accuracy": float(factor_accuracy[0].cpu()),
            "factor_b_accuracy": float(factor_accuracy[1].cpu()),
        }
    return FirstCodebookOracleEvaluation(audio=audio, metrics=metrics)


def _sigma_key(value: float) -> str:
    return format(value, ".6g").replace("-", "m").replace(".", "p")


__all__ = [
    "ArtifactSampleEvaluation",
    "FirstCodebookOracleEvaluation",
    "PairedFeatureEvaluation",
    "evaluate_artifact_sample",
    "evaluate_first_codebook_oracle",
    "evaluate_feature_pair",
    "factor_accuracy",
    "masked_feature_mse",
    "reference_acoustic_condition",
    "seeded_generator",
    "target_acoustic_features",
    "waveform_summary",
    "write_pcm16_wav",
]
