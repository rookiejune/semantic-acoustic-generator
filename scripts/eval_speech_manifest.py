from __future__ import annotations

import argparse
import json
import sys
import wave
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio
from anytrain.evaluator.speech import SpeechEvaluator, UTMOSEvaluator, WhisperASREvaluator
from anytrain.evaluator.text import TextComparisonEvaluator


def main() -> None:
    args = _args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("speech manifest must contain a non-empty items list.")
    text_evaluator = TextComparisonEvaluator(lowercase=True, remove_punctuation=True)
    evaluator = SpeechEvaluator(
        asr=WhisperASREvaluator(
            text_evaluator=text_evaluator,
            model_name=args.whisper_model,
            device=args.device,
            download_root=args.whisper_root,
            decode_options={"temperature": 0.0, "language": args.language},
        ),
        utmos=UTMOSEvaluator(
            device=args.device,
            backend_load_options={"trust_repo": True},
            allow_remote_code=args.allow_utmos_remote_code,
        ),
    )
    private: list[dict[str, Any]] = []
    grouped_predictions: dict[str, list[str]] = defaultdict(list)
    grouped_targets: dict[str, list[str]] = defaultdict(list)
    grouped_utmos: dict[str, list[float]] = defaultdict(list)
    grouped_item_wer: dict[str, list[float]] = defaultdict(list)
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError("speech manifest items must be mappings.")
        group = item.get("group")
        text = item.get("target_text")
        wav = item.get("wav")
        if not isinstance(group, str) or not isinstance(text, str) or not isinstance(wav, str):
            raise TypeError("speech manifest group, target_text, and wav must be strings.")
        audio, sample_rate = _load_audio(Path(wav))
        prediction = evaluator.asr.transcribe(audio, sample_rate)
        if not isinstance(prediction, str):
            if len(prediction) != 1:
                raise ValueError("single WAV transcription must contain one string.")
            prediction = prediction[0]
        utmos = float(evaluator.utmos(audio, sample_rate)["utmos"])
        item_scores = text_evaluator.evaluate(prediction, text)
        grouped_predictions[group].append(prediction)
        grouped_targets[group].append(text)
        grouped_utmos[group].append(utmos)
        grouped_item_wer[group].append(float(item_scores["wer"]))
        private.append({**item, "prediction": prediction, "utmos": utmos, **item_scores})
        print(json.dumps({"index": index, "group": group, "utmos": utmos}, sort_keys=True))
    groups: dict[str, dict[str, float | int]] = {}
    for group in sorted(grouped_predictions):
        text_evaluator.reset()
        corpus = text_evaluator.evaluate(grouped_predictions[group], grouped_targets[group])
        utmos = grouped_utmos[group]
        item_wer = grouped_item_wer[group]
        groups[group] = {
            "count": len(utmos),
            "bleu": float(corpus["bleu"]),
            "wer": float(corpus["wer"]),
            "chrf": float(corpus["chrf"]),
            "item_wer_mean": sum(item_wer) / len(item_wer),
            "utmos_mean": sum(utmos) / len(utmos),
            "utmos_min": min(utmos),
            "utmos_max": max(utmos),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "private.json").write_text(
        json.dumps(private, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "config": {
            "whisper_model": args.whisper_model,
            "language": args.language,
            "temperature": 0.0,
            "device": args.device,
            "allow_utmos_remote_code": args.allow_utmos_remote_code,
        },
        "groups": groups,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, sort_keys=True))


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(path)
    except RuntimeError:
        try:
            return _load_pcm16_wav(path)
        except (OSError, ValueError, wave.Error) as fallback_error:
            raise RuntimeError(
                f"failed to load {path} with torchaudio or the PCM16 WAV fallback"
            ) from fallback_error


def _load_pcm16_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channels <= 0 or sample_width != 2:
        raise ValueError("PCM WAV fallback requires at least one channel and 16-bit samples.")
    pcm = array("h")
    pcm.frombytes(frames)
    if sys.byteorder == "big":
        pcm.byteswap()
    if len(pcm) % channels:
        raise ValueError("PCM WAV samples are not divisible by the channel count.")
    audio = torch.tensor(pcm, dtype=torch.float32).view(-1, channels).transpose(0, 1)
    return audio.div_(32_768), sample_rate


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grouped WAVs with Anytrain ASR/UTMOS.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-root", type=Path, default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--allow-utmos-remote-code",
        action="store_true",
        help="Allow the fixed TorchHub UTMOS repository to execute its model-loading code.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
