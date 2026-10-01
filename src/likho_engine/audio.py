"""Finding and loading recordings."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio

from likho_engine.config import SAMPLE_RATE

AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".aac", ".wma"})


def is_recording(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def find_recordings(inputs: Iterable[Path], default_dir: Path) -> list[Path]:
    """Resolve the recordings to transcribe.

    Each input may be a file or a folder (searched non-recursively). With no inputs,
    every recording in default_dir is used. The result is sorted and de-duplicated.
    """
    inputs = list(inputs) or [default_dir]
    found: dict[Path, None] = {}
    for item in inputs:
        item = item.expanduser().resolve()
        if item.is_dir():
            for child in sorted(item.iterdir()):
                if is_recording(child):
                    found[child] = None
        elif item.is_file():
            found[item] = None
        else:
            raise FileNotFoundError(f"no such file or folder: {item}")
    return list(found)


def load_audio(path: Path, limit_seconds: float = 0.0) -> np.ndarray:
    """Decode a recording to 16 kHz mono float32 samples, optionally keeping only the start."""
    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    if limit_seconds > 0:
        audio = audio[: int(limit_seconds * SAMPLE_RATE)]
    return audio


def duration_seconds(audio: np.ndarray) -> float:
    return audio.shape[0] / SAMPLE_RATE
