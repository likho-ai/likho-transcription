"""From a recording to a two-layer transcript.

Layer 1 is the text in the script the model decoded in. Layer 2 is the same text in
Hinglish. The engine produces layer 1; a Language object decides which language to decode
as, which names to listen for, and how layer 1 is written in Roman letters. The command
line uses LocalLanguage (rules and files on this machine); the service asks likho-language.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from likho_hinglish import Transliterator

from likho_engine.audio import duration_seconds, load_audio
from likho_engine.cleanup import collapse_repeats
from likho_engine.engine import Detection, Engine

# English calls are left as they are when the model is at least this sure.
ENGLISH_THRESHOLD = 0.80


@dataclass(frozen=True)
class Decision:
    decode_as: str
    transliterate: bool


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    text_script: str  # layer 1
    text_roman: str  # layer 2


@dataclass(frozen=True)
class Transcript:
    source: str
    detection: Detection
    decoded_as: str
    policy: str  # "auto", or the language that was forced
    transliterated: bool
    audio_seconds: float
    elapsed_seconds: float
    chunks: int
    silence_skipped_seconds: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def realtime_factor(self) -> float:
        """Seconds of audio written per second of work."""
        return self.audio_seconds / self.elapsed_seconds if self.elapsed_seconds else 0.0


class Language(Protocol):
    """The language decisions a transcription needs."""

    def decide(self, detection: Detection) -> Decision: ...

    def hotwords(self, language: str) -> Sequence[str]: ...

    def to_roman(self, text: str) -> str: ...


def default_decision(detection: Detection) -> Decision:
    """Clearly English stays English; everything else, Urdu included, is decoded as Hindi.

    Decoded as Hindi the model writes Devanagari, which the Hinglish rules read. Hindi with
    many English words still counts as Hindi.
    """
    if detection.language == "en" and detection.probability >= ENGLISH_THRESHOLD:
        return Decision("en", transliterate=False)
    return Decision("hi", transliterate=True)


class LocalLanguage:
    """Language decisions without any service: built-in rules plus the caller's own names and spellings."""

    def __init__(self, glossary: Sequence[str] = (), spellings: Mapping[str, str] | None = None) -> None:
        self._glossary = tuple(glossary)
        self._transliterator = Transliterator(spellings)

    def decide(self, detection: Detection) -> Decision:
        return default_decision(detection)

    def hotwords(self, language: str) -> Sequence[str]:
        return self._glossary

    def to_roman(self, text: str) -> str:
        return self._transliterator.text(text)


def transcribe(
    engine: Engine,
    source: Path | str,
    language: Language,
    *,
    audio: np.ndarray | None = None,
    force_language: str | None = None,
    on_start: Callable[[float, Detection, Decision], None] | None = None,
    on_segment: Callable[[Segment], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe one recording into both layers.

    source           the file to read (or just a name, when audio is given)
    force_language   "hi", "en", ... to skip the automatic choice; None or "auto" to decide from detection
    on_start         called once the audio is loaded and the language is chosen: (seconds, detection, decision)
    on_segment       called for each line as soon as it is ready
    should_stop      asked before each line; True stops the work with engine.StopRequested
    """
    started = time.perf_counter()
    if audio is None:
        audio = load_audio(Path(source), engine.settings.limit_seconds)
    audio_seconds = duration_seconds(audio)

    detection = engine.detect(audio)
    forced = force_language if force_language and force_language != "auto" else None
    # A forced language is decoded as asked; only English is left untransliterated.
    decision = Decision(forced, transliterate=forced != "en") if forced else language.decide(detection)
    if on_start is not None:
        on_start(audio_seconds, detection, decision)

    plan = engine.plan(audio)
    hotwords = ", ".join(language.hotwords(decision.decode_as)) or None

    segments: list[Segment] = []
    for line in engine.decode(audio, plan, decision.decode_as, hotwords, should_stop):
        script = collapse_repeats(line.text)
        roman = language.to_roman(script) if decision.transliterate else script
        segment = Segment(len(segments), line.start, line.end, script, roman)
        segments.append(segment)
        if on_segment is not None:
            on_segment(segment)

    return Transcript(
        source=str(source),
        detection=detection,
        decoded_as=decision.decode_as,
        policy=forced or "auto",
        transliterated=decision.transliterate,
        audio_seconds=audio_seconds,
        elapsed_seconds=time.perf_counter() - started,
        chunks=len(plan.chunks),
        silence_skipped_seconds=plan.silence_skipped_seconds,
        segments=segments,
    )
