"""Model loading, language detection and decoding.

The engine writes what was said in the script the model decodes in (Devanagari for Hindi).
It knows nothing about Hinglish: turning that text into Roman letters is the caller's job
(see transcript.py), so the same engine serves the command line and the service.
"""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import ctranslate2
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.tokenizer import Tokenizer

from likho_engine.audio import duration_seconds
from likho_engine.chunking import Chunk, halve, loudness, plan_chunks
from likho_engine.config import LANGUAGE_CANDIDATES, SAMPLE_RATE, EngineSettings

log = logging.getLogger(__name__)

# Whisper detects the language from the first 30 s of audio.
DETECTION_SECONDS = 30
# CTranslate2 stops generating after 224 tokens (half the decoder's 448 positions, like
# OpenAI's sample_len) whatever max_length or max_new_tokens say. That is about 13 s of
# fast Hindi in Devanagari, and the audio after the cut-off is silently dropped.
MAX_OUTPUT_TOKENS = 224
# A chunk shorter than this that still overflows the decoder is a repetition loop, not speech.
MIN_SPLIT_SECONDS = 2.0


class StopRequested(Exception):  # noqa: N818 - a signal, not an error
    """Raised inside decode() when the caller asked to stop (a cancelled job)."""


@dataclass(frozen=True)
class Detection:
    """What the model thinks is spoken at the start of the recording."""

    language: str
    probability: float
    # The best guesses, most likely first: (("hi", 0.91), ("ur", 0.07), ...). Includes `language`.
    candidates: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Line:
    """One decoded stretch of speech, as the model wrote it."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ChunkPlan:
    chunks: tuple[Chunk, ...]
    audio_seconds: float

    @property
    def speech_seconds(self) -> float:
        return sum(chunk.seconds for chunk in self.chunks)

    @property
    def silence_skipped_seconds(self) -> float:
        return max(0.0, self.audio_seconds - self.speech_seconds)


class SpeechModel(Protocol):
    """The part of faster_whisper.WhisperModel this module uses (lets tests pass a fake)."""

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]: ...


class SpeechPipeline(Protocol):
    """The part of faster_whisper.BatchedInferencePipeline this module uses."""

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]: ...


def resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """Turn "auto" choices into concrete values.

    float16 is only supported on GPU; int8 is the fastest option on CPU.
    """
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def load_model(settings: EngineSettings) -> tuple[WhisperModel, str, str]:
    """Load the Whisper model described by settings; returns (model, device, compute_type)."""
    device, compute_type = resolve_device(settings.device, settings.compute_type)
    log.info("Loading '%s' on %s (%s)...", settings.model_size, device, compute_type)
    started = time.perf_counter()
    model = WhisperModel(
        settings.model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=settings.cpu_threads,
    )
    log.info("Model loaded in %.1fs", time.perf_counter() - started)
    return model, device, compute_type


def token_budget(model: Any, hotwords: str | None) -> int | None:
    """How many text tokens the decoder may emit per chunk; None for a stand-in model without a tokenizer.

    The decoder's 448 positions are shared with the prompt (hotwords and control tokens),
    and generation stops at MAX_OUTPUT_TOKENS anyway. Whisper normally stops by itself
    with an end token, so a chunk that fills the whole budget was cut off, and the audio
    after the cut was never written down.
    """
    hf_tokenizer = getattr(model, "hf_tokenizer", None)
    if hf_tokenizer is None:
        return None
    tokenizer = Tokenizer(hf_tokenizer, True, task="transcribe", language="hi")
    prompt = model.get_prompt(tokenizer, [], without_timestamps=True, hotwords=hotwords)
    return min(MAX_OUTPUT_TOKENS, int(model.max_length) - len(prompt))


class Engine:
    """Loads a Whisper model once and decodes recordings with it.

        engine = Engine(EngineSettings())
        audio = load_audio(Path("call.wav"))
        detection = engine.detect(audio)
        plan = engine.plan(audio)
        for line in engine.decode(audio, plan, language="hi"):
            ...

    A model and pipeline can be injected (tests do this to avoid loading 1.6 GB of weights).
    """

    def __init__(
        self,
        settings: EngineSettings,
        *,
        model: SpeechModel | None = None,
        pipeline: SpeechPipeline | None = None,
    ) -> None:
        self.settings = settings
        if model is None:
            model, self.device, self.compute_type = load_model(settings)
        else:
            self.device, self.compute_type = resolve_device(settings.device, settings.compute_type)
        self.model = model
        # The batched pipeline decodes each chunk on its own; chunking.py chooses the chunks.
        self.pipeline = pipeline if pipeline is not None else BatchedInferencePipeline(model=model)
        self._budgets: dict[str | None, int | None] = {}

    def detect(self, audio: np.ndarray) -> Detection:
        """The model's language guess for the start of the audio, with the runners-up.

        Nothing is decoded: transcribe() only runs the encoder until segments are read.
        """
        head = audio[: DETECTION_SECONDS * SAMPLE_RATE]
        _, info = self.model.transcribe(head, language=None)
        ranked = getattr(info, "all_language_probs", None) or [(info.language, info.language_probability)]
        candidates = tuple((str(code), float(prob)) for code, prob in ranked[:LANGUAGE_CANDIDATES])
        return Detection(str(info.language), float(info.language_probability), candidates)

    def plan(self, audio: np.ndarray) -> ChunkPlan:
        """Decide which stretches of the recording are decoded, and where they are cut."""
        chunks = plan_chunks(
            audio,
            self.settings.chunk_seconds,
            self.settings.speech_threshold,
            self.settings.skip_silence_seconds,
        )
        return ChunkPlan(tuple(chunks), duration_seconds(audio))

    def decode(
        self,
        audio: np.ndarray,
        plan: ChunkPlan,
        language: str,
        hotwords: str | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[Line]:
        """The lines of the recording, in order, as they are decoded.

        should_stop is asked before each line is handed over; when it answers True the
        generator raises StopRequested.
        """
        loud = loudness(audio)
        for raw in self._decode(audio, loud, list(plan.chunks), language, hotwords):
            if should_stop is not None and should_stop():
                raise StopRequested
            text = raw.text.strip()
            if text:
                yield Line(float(raw.start), float(raw.end), text)

    def _decode(
        self, audio: np.ndarray, loud: np.ndarray, chunks: list[Chunk], language: str, hotwords: str | None
    ) -> Iterator[Any]:
        """Raw faster-whisper segments for the chunks, in order.

        A chunk that fills the whole token budget was cut off by the decoder, so it is
        decoded again as two halves (and those again, if needed) until every word is in.
        """
        if not chunks:
            return
        raw_segments, _ = self.pipeline.transcribe(
            audio,
            language=language,
            beam_size=self.settings.beam_size,
            batch_size=self.settings.batch_size,
            hotwords=hotwords,
            clip_timestamps=[{"start": chunk.start, "end": chunk.end} for chunk in chunks],
        )
        for raw in raw_segments:
            if self.ran_out_of_tokens(raw, hotwords) and raw.end - raw.start >= MIN_SPLIT_SECONDS:
                log.warning(
                    "[%.1fs -> %.1fs] filled the decoder's token budget; decoding it in two parts", raw.start, raw.end
                )
                yield from self._decode(audio, loud, halve(Chunk(raw.start, raw.end), loud), language, hotwords)
            else:
                yield raw

    def budget(self, hotwords: str | None) -> int | None:
        """The token budget for one chunk with these hotwords (cached per hotword string)."""
        if hotwords not in self._budgets:
            self._budgets[hotwords] = token_budget(self.model, hotwords)
        return self._budgets[hotwords]

    def ran_out_of_tokens(self, raw: Any, hotwords: str | None = None) -> bool:
        """True when a chunk's text stopped at the token limit (measured: 224 tokens, 223 with a shorter max_length)."""
        budget = self.budget(hotwords)
        return budget is not None and len(raw.tokens) >= budget - 1
