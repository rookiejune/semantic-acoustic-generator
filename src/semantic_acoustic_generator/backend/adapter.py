from __future__ import annotations

from typing import Any, cast

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodec, SemanticAcousticCodes
from torch import Tensor, nn
from torch.nn import functional as F

from semantic_acoustic_generator.config import FeatureAdapter


class _ResidualFactorTargeter:
    def __init__(self, adapter: LongCatCodebookAdapter, target: Tensor) -> None:
        self.adapter = adapter
        self.target = target

    def __call__(self, stage: int, prefix: Tensor) -> Tensor:
        return self.adapter._retarget_factor_pair(self.target, prefix, stage=stage)


class LongCatCodebookAdapter(nn.Module):
    """Expose selected LongCat factor embeddings as frame-aligned acoustic features."""

    def __init__(self, backend: SemanticAcousticCodec, *, codebooks: int = 1) -> None:
        super().__init__()
        if backend.name != "longcat":
            raise ValueError("LongCat factor features require backend.name='longcat'.")
        if backend.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
            raise ValueError("LongCat factor features require frame-aligned units.")
        self.backend = backend
        self._feature_codebooks = _positive_int(codebooks, "feature codebook count")
        if len(backend.acoustic_codebook_sizes) < self._feature_codebooks:
            raise ValueError("LongCat feature codebooks exceed backend acoustic codebooks.")
        factor_sizes: list[tuple[int, int]] = []
        factor_dims: list[tuple[int, int]] = []
        for index, stage in enumerate(self._stages()):
            size_a = _positive_int(stage.codebook_size_a, f"stage {index} codebook_size_a")
            size_b = _positive_int(stage.codebook_size_b, f"stage {index} codebook_size_b")
            dim_a = _embedding_dim(stage.codebook_a, f"stage {index} codebook_a")
            dim_b = _embedding_dim(stage.codebook_b, f"stage {index} codebook_b")
            if size_a != stage.codebook_a.num_embeddings:
                raise ValueError(f"LongCat stage {index} codebook_size_a must match codebook_a.")
            if size_b != stage.codebook_b.num_embeddings:
                raise ValueError(f"LongCat stage {index} codebook_size_b must match codebook_b.")
            if backend.acoustic_codebook_sizes[index] != size_a * size_b:
                raise ValueError(
                    f"LongCat stage {index} factors must match the composite codebook size."
                )
            projected_dim = _projection_dim(stage.out_proj_a, dim_a) + _projection_dim(
                stage.out_proj_b,
                dim_b,
            )
            if projected_dim != backend.acoustic_feature_dim:
                raise ValueError(
                    f"LongCat stage {index} projections must produce the decoder feature dim."
                )
            factor_sizes.append((size_a, size_b))
            factor_dims.append((dim_a, dim_b))
        self._factor_sizes = tuple(factor_sizes)
        self._factor_dims = tuple(factor_dims)

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def sample_rate(self) -> int:
        return self.backend.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.backend.frame_rate

    @property
    def semantic_frame_rate(self) -> float:
        return self.backend.semantic_frame_rate

    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]:
        return self.backend.semantic_codebook_sizes

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]:
        return self.backend.acoustic_codebook_sizes

    @property
    def acoustic_layout(self) -> AcousticLayout:
        return self.backend.acoustic_layout

    @property
    def semantic_codebook(self) -> Tensor:
        return self.backend.semantic_codebook

    @property
    def acoustic_feature_dim(self) -> int:
        return sum(dim for pair in self._factor_dims for dim in pair)

    @property
    def feature_codebooks(self) -> int:
        return self._feature_codebooks

    @property
    def factor_codebooks(self) -> tuple[Tensor, ...]:
        result: list[Tensor] = []
        for stage in self._stages():
            result.extend((stage.codebook_a.weight, stage.codebook_b.weight))
        return tuple(result)

    @property
    def acoustic_unit_length(self) -> int | None:
        return self.backend.acoustic_unit_length

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticAcousticCodes:
        return self.backend.tokenize(audio, sample_rate)

    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor:
        return self.backend.detokenize(codes)

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        encode = getattr(self.backend, "encode", None)
        if not callable(encode):
            raise TypeError("adapted LongCat backend does not provide encode().")
        return cast(Tensor, encode(audio, sample_rate))

    def decode(self, codes: Tensor) -> Tensor:
        decode = getattr(self.backend, "decode", None)
        if not callable(decode):
            raise TypeError("adapted LongCat backend does not provide decode().")
        return cast(Tensor, decode(codes))

    @torch.no_grad()
    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        return self.factor_codes_to_features(self.factor_codes(acoustic_codes))

    @torch.no_grad()
    def factor_codes_to_features(self, factor_codes: Tensor) -> Tensor:
        self._validate_factor_codes(factor_codes, stages=self.feature_codebooks)
        return torch.cat(
            tuple(
                F.embedding(code.to(device=weight.device), weight)
                for code, weight in zip(
                    factor_codes.unbind(dim=-1),
                    self.factor_codebooks,
                    strict=True,
                )
            ),
            dim=-1,
        )

    @torch.no_grad()
    def retarget_factor_codes(
        self,
        acoustic_codes: Tensor,
        predicted_factors: Tensor,
    ) -> Tensor:
        """Requantize each later residual after the generated prefix.

        This is an oracle diagnostic because the residual is measured against the
        target utterance's full acoustic reconstruction.
        """
        _ = self.factor_codes(acoustic_codes)
        self._validate_factor_codes(predicted_factors, stages=self.feature_codebooks)
        if predicted_factors.shape[:2] != acoustic_codes.shape[:2]:
            raise ValueError("predicted factors and acoustic codes must align on batch and time.")
        if self.feature_codebooks == 1:
            return predicted_factors.clone()

        target = self.backend.acoustic_codes_to_features(acoustic_codes)
        output = predicted_factors.clone()
        for stage_index in range(1, self.feature_codebooks):
            prefix = output[..., : stage_index * 2]
            output[..., stage_index * 2 : stage_index * 2 + 2] = (
                self._retarget_factor_pair(target, prefix, stage=stage_index)
            )
        return output

    @torch.no_grad()
    def residual_factor_targeter(
        self,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> _ResidualFactorTargeter:
        if mask.shape != acoustic_codes.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("residual factor target mask must be boolean and frame-aligned.")
        if mask.device != acoustic_codes.device:
            raise ValueError("residual factor target mask and codes must share a device.")
        packed_codes = acoustic_codes[mask]
        target = self.backend.acoustic_codes_to_features(packed_codes[:, None])[:, 0]
        return _ResidualFactorTargeter(self, target)

    @torch.no_grad()
    def factor_codes(self, acoustic_codes: Tensor) -> Tensor:
        if acoustic_codes.dim() != 3:
            raise ValueError("acoustic_codes must have shape [batch, time, codebook].")
        if acoustic_codes.size(-1) != len(self.acoustic_codebook_sizes):
            raise ValueError(
                "acoustic_codes must contain the backend acoustic codebooks before adaptation."
            )
        if acoustic_codes.is_floating_point() or acoustic_codes.is_complex():
            raise TypeError("acoustic_codes must be an integer tensor.")
        factors: list[Tensor] = []
        for index, (size_a, size_b) in enumerate(self._factor_sizes):
            composite = acoustic_codes[..., index]
            if bool(((composite < 0) | (composite >= size_a * size_b)).any()):
                raise ValueError(f"LongCat stage {index} codes contain an ID outside the codebook.")
            factors.extend(
                (
                    torch.div(composite, size_b, rounding_mode="floor"),
                    composite.remainder(size_b),
                )
            )
        return torch.stack(factors, dim=-1)

    @torch.no_grad()
    def snap_features(self, acoustic_features: Tensor) -> Tensor:
        factors = self.features_to_factor_codes(acoustic_features)
        return torch.cat(
            tuple(
                F.embedding(code, codebook)
                for code, codebook in zip(
                    factors.unbind(dim=-1),
                    self.factor_codebooks,
                    strict=True,
                )
            ),
            dim=-1,
        )

    @torch.no_grad()
    def features_to_factor_codes(self, acoustic_features: Tensor) -> Tensor:
        if acoustic_features.dim() != 3:
            raise ValueError("acoustic_features must have shape [batch, time, dim].")
        if acoustic_features.size(-1) != self.acoustic_feature_dim:
            raise ValueError(
                f"adapted LongCat acoustic features must have dim {self.acoustic_feature_dim}."
            )
        dims = tuple(dim for pair in self._factor_dims for dim in pair)
        factors = acoustic_features.split(dims, dim=-1)
        indices = []
        for value, codebook in zip(factors, self.factor_codebooks, strict=True):
            normalized = F.normalize(value.float(), dim=-1)
            normalized_codebook = F.normalize(codebook.float(), dim=-1)
            indices.append(
                torch.matmul(normalized, normalized_codebook.transpose(0, 1)).argmax(-1)
            )
        return torch.stack(indices, dim=-1)

    @torch.no_grad()
    def project_features(
        self,
        acoustic_features: Tensor,
        *,
        active_codebooks: int | None = None,
    ) -> Tensor:
        if acoustic_features.dim() != 3:
            raise ValueError("acoustic_features must have shape [batch, time, dim].")
        if acoustic_features.size(-1) != self.acoustic_feature_dim:
            raise ValueError(
                f"adapted LongCat acoustic features must have dim {self.acoustic_feature_dim}."
            )
        count = (
            self.feature_codebooks
            if active_codebooks is None
            else _positive_int(active_codebooks, "active feature codebook count")
        )
        if count > self.feature_codebooks:
            raise ValueError("active LongCat feature codebooks exceed the adapted codebooks.")
        stages = self._stages()[:count]
        reference = stages[0].codebook_a.weight
        features = acoustic_features.to(device=reference.device, dtype=reference.dtype)
        dims = tuple(dim for pair in self._factor_dims for dim in pair)
        factors = features.split(dims, dim=-1)
        projected: Tensor | None = None
        for index, stage in enumerate(stages):
            features_a = factors[index * 2].transpose(1, 2).contiguous()
            features_b = factors[index * 2 + 1].transpose(1, 2).contiguous()
            value = torch.cat(
                (stage.out_proj_a(features_a), stage.out_proj_b(features_b)),
                dim=1,
            )
            projected = value if projected is None else projected + value
        if projected is None:
            raise RuntimeError("LongCat factor adapter must contain at least one stage.")
        return projected.transpose(1, 2).contiguous()

    def _project_factor_codes(self, factor_codes: Tensor, *, stages: int) -> Tensor:
        self._validate_factor_codes(factor_codes, stages=stages)
        selected = self._stages()[:stages]
        projected: Tensor | None = None
        for index, stage in enumerate(selected):
            code_a = factor_codes[..., index * 2].to(device=stage.codebook_a.weight.device)
            code_b = factor_codes[..., index * 2 + 1].to(device=stage.codebook_b.weight.device)
            features_a = F.embedding(code_a, stage.codebook_a.weight).transpose(1, 2).contiguous()
            features_b = F.embedding(code_b, stage.codebook_b.weight).transpose(1, 2).contiguous()
            value = torch.cat(
                (stage.out_proj_a(features_a), stage.out_proj_b(features_b)),
                dim=1,
            )
            projected = value if projected is None else projected + value
        if projected is None:
            raise RuntimeError("LongCat factor projection requires at least one stage.")
        return projected.transpose(1, 2).contiguous()

    def _retarget_factor_pair(
        self,
        target: Tensor,
        prefix: Tensor,
        *,
        stage: int,
    ) -> Tensor:
        if not 0 < stage < self.feature_codebooks:
            raise ValueError("residual factor stage must follow a non-empty generated prefix.")
        packed = prefix.dim() == 2
        prefix_frames = prefix[:, None] if packed else prefix
        target_frames = target[:, None] if packed else target
        if target_frames.dim() != 3 or target_frames.shape[:2] != prefix_frames.shape[:2]:
            raise ValueError("residual target and factor prefix must align on batch and time.")
        projected = self._project_factor_codes(prefix_frames, stages=stage)
        residual = (target_frames - projected).transpose(1, 2).contiguous()
        result = self._stages()[stage](residual)
        if (
            not isinstance(result, tuple)
            or len(result) < 4
            or not isinstance(result[3], Tensor)
        ):
            raise TypeError("LongCat quantizer stage must return code indices at position 3.")
        composite = result[3]
        if composite.shape != prefix_frames.shape[:2]:
            raise ValueError("LongCat residual code indices must align on batch and time.")
        size_b = self._factor_sizes[stage][1]
        pair = torch.stack(
            (
                torch.div(composite, size_b, rounding_mode="floor"),
                composite.remainder(size_b),
            ),
            dim=-1,
        ).to(device=prefix.device)
        return pair[:, 0] if packed else pair

    def _validate_factor_codes(self, factor_codes: Tensor, *, stages: int) -> None:
        if factor_codes.dim() != 3 or factor_codes.size(-1) != stages * 2:
            raise ValueError("factor codes must have shape [batch, time, 2 * stages].")
        if factor_codes.is_floating_point() or factor_codes.is_complex():
            raise TypeError("factor codes must be an integer tensor.")
        limits = torch.tensor(
            [size for pair in self._factor_sizes[:stages] for size in pair],
            device=factor_codes.device,
            dtype=torch.long,
        )
        if bool(((factor_codes < 0) | (factor_codes >= limits)).any()):
            raise ValueError("factor codes contain an ID outside the selected codebooks.")

    @torch.no_grad()
    def native_features(self, acoustic_codes: Tensor) -> Tensor:
        _ = self.factor_codes(acoustic_codes)
        model = self._decoder()
        quantizer = model.acoustic_quantizer
        convert = getattr(quantizer, "from_codes", None)
        if not callable(convert):
            raise TypeError("LongCat acoustic quantizer must provide from_codes().")
        output = convert(
            acoustic_codes[..., : self.feature_codebooks].transpose(1, 2).contiguous()
        )
        if not isinstance(output, tuple) or not output or not isinstance(output[0], Tensor):
            raise TypeError("LongCat acoustic quantizer from_codes() must return features.")
        features = output[0]
        if features.dim() != 3:
            raise ValueError("LongCat native features must have shape [batch, dim, time].")
        return features.transpose(1, 2).contiguous()

    @torch.no_grad()
    def native_stage0_features(self, acoustic_codes: Tensor) -> Tensor:
        if self.feature_codebooks != 1:
            raise RuntimeError("native_stage0_features requires exactly one selected codebook.")
        return self.native_features(acoustic_codes)

    @torch.no_grad()
    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
        *,
        active_codebooks: int | None = None,
    ) -> Tensor:
        if acoustic_features.dim() != 3:
            raise ValueError("acoustic_features must have shape [batch, time, dim].")
        if acoustic_features.size(-1) != self.acoustic_feature_dim:
            raise ValueError(
                f"adapted LongCat acoustic features must have dim {self.acoustic_feature_dim}."
            )
        if acoustic_features.shape[:2] != semantic_codes.shape[:2]:
            raise ValueError("semantic_codes and acoustic_features must align on batch and time.")
        if not acoustic_features.is_floating_point() or acoustic_features.is_complex():
            raise TypeError("acoustic_features must be real floating point tensors.")
        return self.backend.decode_features(
            semantic_codes,
            self.project_features(
                acoustic_features,
                active_codebooks=active_codebooks,
            ),
        )

    def _decoder(self) -> Any:
        decoder = getattr(self.backend, "_decoder", None)
        if not callable(decoder):
            raise TypeError("LongCat backend must provide its decoder to the feature adapter.")
        return decoder()

    def _stages(self) -> tuple[Any, ...]:
        model = self._decoder()
        quantizer = getattr(model, "acoustic_quantizer", None)
        quantizers = getattr(quantizer, "quantizers", None)
        if quantizers is None or len(quantizers) < self.feature_codebooks:
            raise TypeError("LongCat decoder does not expose the requested quantizer stages.")
        stages = tuple(quantizers[: self.feature_codebooks])
        for index, stage in enumerate(stages):
            for name in (
                "codebook_size_a",
                "codebook_size_b",
                "codebook_a",
                "codebook_b",
                "out_proj_a",
                "out_proj_b",
            ):
                if not hasattr(stage, name):
                    raise TypeError(f"LongCat stage {index} quantizer must expose {name}.")
        return stages


LongCatFirstCodebookAdapter = LongCatCodebookAdapter


def adapt_backend(
    backend: SemanticAcousticCodec,
    adapter: FeatureAdapter,
    *,
    codebooks: int = 1,
) -> SemanticAcousticCodec:
    if not isinstance(adapter, FeatureAdapter):
        raise TypeError("adapter must be a FeatureAdapter.")
    if adapter is FeatureAdapter.NONE:
        return backend
    if adapter in {
        FeatureAdapter.LONGCAT_FIRST_CODEBOOK,
        FeatureAdapter.LONGCAT_CODEBOOKS,
    }:
        selected = 1 if adapter is FeatureAdapter.LONGCAT_FIRST_CODEBOOK else codebooks
        if adapter is FeatureAdapter.LONGCAT_FIRST_CODEBOOK and codebooks != 1:
            raise ValueError("longcat_first_codebook requires codebooks=1.")
        if isinstance(backend, LongCatCodebookAdapter):
            if backend.feature_codebooks != selected:
                raise ValueError("LongCat backend is already adapted with a different codebook count.")
            return backend
        return cast(SemanticAcousticCodec, LongCatCodebookAdapter(backend, codebooks=selected))
    raise AssertionError(f"unsupported feature adapter: {adapter}")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"LongCat {name} must be a positive integer.")
    return value


def _embedding_dim(value: object, name: str) -> int:
    if not isinstance(value, nn.Embedding):
        raise TypeError(f"LongCat {name} must be an nn.Embedding.")
    return int(value.embedding_dim)


@torch.no_grad()
def _projection_dim(value: object, input_dim: int) -> int:
    if not isinstance(value, nn.Module):
        raise TypeError("LongCat stage-0 output projection must be an nn.Module.")
    parameter = next(value.parameters(), None)
    if parameter is None:
        raise TypeError("LongCat stage-0 output projection must expose parameters.")
    sample = parameter.new_zeros(1, input_dim, 1)
    output = value(sample)
    if output.dim() != 3 or output.shape[:1] != (1,) or output.size(-1) != 1:
        raise ValueError("LongCat stage-0 output projection must preserve [batch, channel, time].")
    return int(output.size(1))


__all__ = ["LongCatCodebookAdapter", "LongCatFirstCodebookAdapter", "adapt_backend"]
