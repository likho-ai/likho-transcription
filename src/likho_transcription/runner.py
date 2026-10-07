"""Runs one transcription job: fetch the audio, decode it, write both layers, store the result."""

import asyncio
import logging
import tempfile
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from likho_engine import Decision, Detection, Engine, EngineSettings, Segment, StopRequested, Transcript, transcribe
from likho_engine.audio import load_audio
from likho_engine.config import MODEL_CHOICES
from likho_transcription.errors import JobCancelled, JobError
from likho_transcription.gateways import LanguageGateway, MediaGateway
from likho_transcription.ids import new_id
from likho_transcription.settings import Settings
from likho_transcription.store import Document, TranscriptStore

log = logging.getLogger(__name__)

ENGINE_NAME = "faster-whisper"
# The script each decoded language is written in (layer 1).
SCRIPTS = {"hi": "devanagari", "mr": "devanagari", "ne": "devanagari", "ur": "arabic"}

EngineFactory = Callable[[str], Engine]
OnStarted = Callable[[float, Detection, Decision], Awaitable[None]]
OnSegment = Callable[[Segment, float], Awaitable[None]]


@dataclass(frozen=True)
class JobRequest:
    job_id: str
    recording_id: str
    media_id: str
    workspace_id: str
    model_registry_id: str = ""
    language_policy: str = "auto"
    force: bool = False
    # An evaluation (likho-ml): the transcript is made and returned but not stored.
    evaluation: bool = False


class EngineProvider:
    """Keeps one speech model in memory and swaps it when a job asks for another."""

    def __init__(self, settings: Settings, factory: EngineFactory | None = None) -> None:
        self._settings = settings
        self._factory = factory or self._load
        self._engine: Engine | None = None
        self._size = ""
        self._lock = asyncio.Lock()

    def _load(self, size: str) -> Engine:
        return Engine(
            EngineSettings(
                model_size=size,
                device=self._settings.device,
                compute_type=self._settings.compute_type,
                cpu_threads=self._settings.cpu_threads,
            )
        )

    @property
    def default_registry_id(self) -> str:
        return f"{ENGINE_NAME}/{self._settings.default_model}"

    def size_of(self, registry_id: str) -> str:
        """The model size a registry id names: "" is the default, "faster-whisper/turbo" is "turbo"."""
        if not registry_id:
            return self._settings.default_model
        engine, _, size = registry_id.rpartition("/")
        if engine not in ("", ENGINE_NAME) or size not in (*MODEL_CHOICES, self._settings.default_model):
            raise JobError("model_unavailable", f"The model {registry_id!r} is not available", retryable=False)
        return size

    def registry_ids(self) -> list[str]:
        sizes = dict.fromkeys((self._settings.default_model, *MODEL_CHOICES))
        return [f"{ENGINE_NAME}/{size}" for size in sizes]

    async def get(self, registry_id: str) -> tuple[Engine, str]:
        size = self.size_of(registry_id)
        async with self._lock:
            if self._engine is None or self._size != size:
                self._engine = None  # free the old model before loading the next one
                try:
                    self._engine = await asyncio.to_thread(self._factory, size)
                except Exception as error:
                    raise JobError(
                        "model_unavailable", f"The model {size!r} could not be loaded", retryable=True
                    ) from error
                self._size = size
            return self._engine, f"{ENGINE_NAME}/{size}"


class _BlockingLanguage:
    """The engine's Language, answered by likho-language. Called from the decoding thread."""

    def __init__(self, gateway: LanguageGateway, workspace_id: str, loop: asyncio.AbstractEventLoop, timeout: float):
        self._gateway = gateway
        self._workspace_id = workspace_id
        self._loop = loop
        self._timeout = timeout
        self._versions: list[int] = []

    def _wait(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(self._timeout)

    def decide(self, detection: Detection) -> Decision:
        return self._wait(self._gateway.decide(self._workspace_id, detection))

    def hotwords(self, language: str) -> Sequence[str]:
        return self._wait(self._gateway.hotwords(self._workspace_id, language))

    def to_roman(self, text: str) -> str:
        texts, version = self._wait(self._gateway.to_roman(self._workspace_id, [text]))
        self._versions.append(version)
        return texts[0]

    @property
    def vocabulary_version(self) -> int:
        """The vocabulary version behind layer 2; 0 when any line fell back to the built-in rules."""
        if not self._versions or min(self._versions) == 0:
            return 0
        return max(self._versions)


class JobRunner:
    def __init__(
        self,
        settings: Settings,
        store: TranscriptStore,
        language: LanguageGateway,
        media: MediaGateway,
        engines: EngineProvider,
    ) -> None:
        self._settings = settings
        self._store = store
        self._language = language
        self._media = media
        self._engines = engines
        self._lock = asyncio.Lock()  # one transcription at a time: the model keeps the CPU busy
        self._running: dict[str, threading.Event] = {}
        # Jobs cancelled before they were in hand: a job given up on by likho-api whose request
        # reaches this worker later (or a moment after it said "started") is not run at all.
        self._cancelled: deque[str] = deque(maxlen=1000)

    def cancel(self, job_id: str) -> bool:
        """Ask a running job to stop after its current line. False when the job is not running here.

        A job not running here is remembered: should its request arrive later, it is dropped.
        """
        stop = self._running.get(job_id)
        if stop is None:
            self._cancelled.append(job_id)
            return False
        stop.set()
        return True

    async def run(
        self, job: JobRequest, on_started: OnStarted | None = None, on_segment: OnSegment | None = None
    ) -> Document:
        """Transcribe and store (an evaluation is not stored). Raises JobError or JobCancelled."""
        stop = threading.Event()
        self._running[job.job_id] = stop  # in hand first, so a cancel from now on is not missed
        if job.job_id in self._cancelled:
            stop.set()
        try:
            async with self._lock:
                return await self._run(job, stop, on_started, on_segment)
        finally:
            self._running.pop(job.job_id, None)

    async def _run(
        self, job: JobRequest, stop: threading.Event, on_started: OnStarted | None, on_segment: OnSegment | None
    ) -> Document:
        def cancelled() -> None:
            if stop.is_set():
                raise JobCancelled

        cancelled()
        engine, registry_id = await self._engines.get(job.model_registry_id)
        cancelled()  # a cancel that came while the model loaded

        with tempfile.TemporaryDirectory(prefix="likho-") as folder:
            path = await self._media.download(job.media_id, Path(folder))
            try:
                audio = await asyncio.to_thread(load_audio, path)
            except Exception as error:
                raise JobError("audio_unreadable", "The audio file could not be read", retryable=False) from error
        if len(audio) == 0:
            raise JobError("audio_unreadable", "The recording is empty", retryable=False)
        cancelled()  # ... or while the audio was fetched; decoding asks again before every line

        loop = asyncio.get_running_loop()
        language = _BlockingLanguage(self._language, job.workspace_id, loop, self._settings.rpc_timeout_seconds + 5)
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        total = len(audio) / 16_000

        def started(seconds: float, detection: Detection, decision: Decision) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("started", (seconds, detection, decision)))

        def segment(line: Segment) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("segment", line))

        async def pump() -> None:
            """Hands what the decoding thread reports to the async callbacks, in order."""
            while True:
                kind, payload = await queue.get()
                if kind == "end":
                    return
                if kind == "started" and on_started is not None:
                    await on_started(*payload)
                elif kind == "segment" and on_segment is not None:
                    await on_segment(payload, total)

        pumping = asyncio.create_task(pump())
        try:
            transcript = await asyncio.to_thread(
                transcribe,
                engine,
                job.recording_id,
                language,
                audio=audio,
                force_language=job.language_policy,
                on_start=started,
                on_segment=segment,
                should_stop=stop.is_set,
            )
        except StopRequested:
            raise JobCancelled from None
        except JobError:
            raise
        except Exception as error:
            raise JobError("internal", f"Transcription failed: {error}", retryable=True) from error
        finally:
            queue.put_nowait(("end", None))
            await pumping

        document = self._document(job, engine, registry_id, transcript, language)
        if job.evaluation:
            # Scored against the gold set and thrown away: the recording's versions stay as they are.
            return {**document, "_id": new_id("trn"), "created_at": datetime.now(UTC), "version": 0}
        return await self._store.insert(document)

    @staticmethod
    def _document(
        job: JobRequest, engine: Engine, registry_id: str, transcript: Transcript, language: _BlockingLanguage
    ) -> Document:
        detection = transcript.detection
        return {
            "recording_id": job.recording_id,
            "job_id": job.job_id,
            "workspace_id": job.workspace_id,
            "model": {"registry_id": registry_id, "engine": ENGINE_NAME, "compute": engine.compute_type},
            "language": {
                "detected": detection.language,
                "probability": detection.probability,
                "candidates": [{"language": code, "probability": prob} for code, prob in detection.candidates],
                "decoded_as": transcript.decoded_as,
                "policy": transcript.policy,
            },
            "script": SCRIPTS.get(transcript.decoded_as, "latin"),
            "transliterated": transcript.transliterated,
            "vocabulary_version": language.vocabulary_version if transcript.transliterated else 0,
            "segments": [
                {
                    "index": s.index,
                    "start": s.start,
                    "end": s.end,
                    "text_script": s.text_script,
                    "text_roman": s.text_roman,
                }
                for s in transcript.segments
            ],
            "stats": {
                "audio_seconds": transcript.audio_seconds,
                "elapsed_seconds": transcript.elapsed_seconds,
                "realtime_factor": transcript.realtime_factor,
                "chunks": transcript.chunks,
                "silence_skipped_seconds": transcript.silence_skipped_seconds,
            },
        }
