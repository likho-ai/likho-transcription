"""Stand-ins for the Whisper model and pipeline, so tests need no 1.6 GB of weights."""

import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from likho_engine import Engine, EngineSettings


class FakeModel:
    """Pretends Whisper detected the given language."""

    def __init__(self, language: str = "hi", probability: float = 0.9, ranked: list[tuple[str, float]] | None = None):
        self.info = SimpleNamespace(language=language, language_probability=probability)
        if ranked is not None:
            self.info.all_language_probs = ranked

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
        return iter([]), self.info


LINES = [
    (0.5, 4.0, " नमस्ते, आपका ऑर्डर कल तक पहुँच जाएगा "),
    (4.0, 9.5, " दवा दिन में दो बार लीजिए "),
]


class FakePipeline:
    """Returns Devanagari segments and records how it was called.

    gate: when given, the pipeline waits for it before handing over the second segment,
    so a test can act while a job is in the middle of its work.
    """

    def __init__(self, lines: list[tuple[float, float, str]] | None = None, gate: threading.Event | None = None):
        self.lines = LINES if lines is None else lines
        self.gate = gate
        self.calls: list[dict[str, Any]] = []
        self.first_line_out = threading.Event()

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(kwargs)

        def segments() -> Any:
            for number, (start, end, text) in enumerate(self.lines):
                if number == 1:
                    self.first_line_out.set()
                    if self.gate is not None:
                        self.gate.wait(timeout=30)
                yield SimpleNamespace(start=start, end=end, text=text, tokens=[1, 2, 3])

        return segments(), None


class TruncatingPipeline:
    """First call: one segment that used up the whole token budget. Later calls: one segment per clip."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(kwargs)
        clips = kwargs["clip_timestamps"]
        if len(self.calls) == 1:
            segments = [SimpleNamespace(start=clips[0]["start"], end=clips[0]["end"], text=" नमस्ते ", tokens=[1, 2, 3])]
        else:
            texts = [" नमस्ते ", " धन्यवाद "]
            segments = [
                SimpleNamespace(start=clip["start"], end=clip["end"], text=texts[i], tokens=[1])
                for i, clip in enumerate(clips)
            ]
        return iter(segments), None


def fake_engine(
    language: str = "hi",
    probability: float = 0.9,
    *,
    pipeline: Any = None,
    settings: EngineSettings | None = None,
    ranked: list[tuple[str, float]] | None = None,
) -> tuple[Engine, Any]:
    pipeline = pipeline or FakePipeline()
    engine = Engine(settings or EngineSettings(), model=FakeModel(language, probability, ranked), pipeline=pipeline)
    return engine, pipeline


def write_silent_wav(path: Path, seconds: float = 10.0, rate: int = 8000) -> Path:
    """A valid recording with nothing in it: the voice detector finds no speech, so it is decoded as one chunk."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return path
