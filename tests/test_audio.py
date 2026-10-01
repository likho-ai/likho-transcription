"""Tests for finding and decoding recordings."""

import wave
from pathlib import Path

import pytest

from likho_engine.audio import duration_seconds, find_recordings, load_audio


def write_silent_wav(path: Path, seconds: float, rate: int = 8000) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def test_find_recordings_in_default_dir_sorted_and_filtered(tmp_path: Path) -> None:
    (tmp_path / "b.wav").touch()
    (tmp_path / "a.mp3").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.wav").touch()  # folders are not searched recursively

    found = find_recordings([], tmp_path)

    assert [p.name for p in found] == ["a.mp3", "b.wav"]


def test_find_recordings_accepts_files_and_folders_without_duplicates(tmp_path: Path) -> None:
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.wav").touch()

    found = find_recordings([tmp_path / "a.wav", tmp_path], tmp_path / "unused")

    assert [p.name for p in found] == ["a.wav", "b.wav"]


def test_find_recordings_rejects_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_recordings([tmp_path / "missing.wav"], tmp_path)


def test_load_audio_resamples_to_16khz_mono(tmp_path: Path) -> None:
    audio = load_audio(write_silent_wav(tmp_path / "call.wav", seconds=2.0))

    assert audio.ndim == 1
    assert audio.dtype.name == "float32"
    assert abs(duration_seconds(audio) - 2.0) < 0.05


def test_load_audio_limit_keeps_only_the_start(tmp_path: Path) -> None:
    audio = load_audio(write_silent_wav(tmp_path / "call.wav", seconds=3.0), limit_seconds=1.0)

    assert abs(duration_seconds(audio) - 1.0) < 0.01
