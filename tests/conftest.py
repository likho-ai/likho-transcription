"""Fixtures for the service tests: the real service against the likho-infra stack (MongoDB and NATS),
with stand-ins for the speech model and for the two services it calls (likho-language, likho-media).

Start the stack first:  likho-infra> bash scripts/up.sh   (or .\\stack.ps1 up)
Without it these tests are skipped locally; with LIKHO_REQUIRE_STACK=1 (set in CI) they fail instead.
"""

import asyncio
import contextlib
import functools
import http.server
import json
import os
import socket
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
import nats
import pytest
import pytest_asyncio
from likho.language.v1 import language_pb2, language_pb2_grpc
from likho.media.v1 import media_pb2, media_pb2_grpc
from likho.transcription.v1 import transcription_pb2_grpc
from likho_hinglish import Transliterator
from nats.js.api import ConsumerConfig, DeliverPolicy

from likho_engine import Engine, EngineSettings
from likho_transcription.__main__ import serve
from likho_transcription.ids import new_id
from likho_transcription.settings import Settings
from likho_transcription.store import TranscriptStore
from tests.fakes import FakeModel, FakePipeline, write_silent_wav

WATCHED = (
    "likho.live.segment",
    "likho.transcription.completed",
    "likho.transcription.failed",
    "likho.dead",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class FakeLanguage(language_pb2_grpc.LanguageServiceServicer):
    """likho-language as the transcription service sees it. Tests change its vocabulary and switch it off."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.spellings: dict[str, str] = {}
        self.version = 3
        self.glossary = ["त्रिफला"]
        self.down = False

    async def _check(self, context: grpc.aio.ServicerContext) -> None:
        if self.down:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "switched off by the test")

    async def ResolveDecodePolicy(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        await self._check(context)
        english = request.detected == "en" and request.probability >= 0.8
        return language_pb2.ResolveDecodePolicyResponse(decode_as="en" if english else "hi", transliterate=not english)

    async def GetHotwords(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        await self._check(context)
        return language_pb2.GetHotwordsResponse(terms=self.glossary, vocabulary_version=self.version)

    async def TransliterateBatch(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        await self._check(context)
        rules = Transliterator(self.spellings)
        return language_pb2.TransliterateBatchResponse(
            texts_roman=[rules.text(text) for text in request.texts], vocabulary_version=self.version
        )


class FakeMedia(media_pb2_grpc.MediaServiceServicer):
    """likho-media as the transcription service sees it: media ids that lead to files, or to trouble."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.files = {"med_good": "call.wav", "med_broken": "broken.wav", "med_gone": "deleted.wav"}

    async def GetDownloadUrl(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        if request.id == "med_down":
            await context.abort(grpc.StatusCode.UNAVAILABLE, "media service is restarting")
        if request.id not in self.files:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"media {request.id} not found")
        return media_pb2.GetDownloadUrlResponse(url=f"{self.base_url}/{self.files[request.id]}")


class SwitchablePipeline:
    """The engine keeps one pipeline; tests swap what stands behind it."""

    def __init__(self) -> None:
        self.current = FakePipeline()

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
        return self.current.transcribe(audio, **kwargs)


@dataclass
class Platform:
    settings: Settings
    stub: transcription_pb2_grpc.TranscriptionServiceStub
    channel: grpc.aio.Channel
    language: FakeLanguage
    media: FakeMedia
    pipeline: SwitchablePipeline
    store: TranscriptStore
    js: Any
    subscriptions: dict[str, Any] = field(default_factory=dict)
    seen: list[tuple[str, dict[str, Any], dict[str, str] | None]] = field(default_factory=list)

    async def request(
        self, *, media_id: str = "med_good", force: bool = False, recording_id: str = "", **extra: Any
    ) -> dict[str, str]:
        """Publish a likho.transcription.requested event and return its ids."""
        ids = {
            "job_id": new_id("job"),
            "recording_id": recording_id or new_id("rec"),
            "media_id": media_id,
            "workspace_id": "wsp_01JB7Z5K3M9Q2W4X6Y8A0C1E3G",
        }
        event = {
            "specversion": "1.0",
            "id": new_id("evt"),
            "source": "likho-api",
            "type": "likho.transcription.requested.v1",
            "time": "2026-10-01T07:00:00Z",
            "subject": ids["recording_id"],
            "datacontenttype": "application/json",
            "data": {**ids, "language_policy": "auto", "force": force, **extra},
        }
        await self.js.publish("likho.transcription.requested", json.dumps(event).encode())
        return ids

    async def events(self, job_id: str, until: str, within: float = 20.0) -> list[tuple[str, dict[str, Any]]]:
        """Everything published about a job, in arrival order per subject, up to the first `until` event."""
        deadline = asyncio.get_running_loop().time() + within
        while True:
            mine = [(s, e) for s, e, _ in self.seen if self._job_of(e) == job_id]
            if any(subject == until for subject, _ in mine):
                # Subjects are read one after the other, so events on another subject that were
                # published earlier may still be waiting: read everything that is there first.
                await asyncio.sleep(0.2)
                await self._read(wait=0.02)
                return [(s, e) for s, e, _ in self.seen if self._job_of(e) == job_id]
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(f"no {until} event for {job_id} within {within}s; saw {[s for s, _ in mine]}")
            await self._read(wait=0.02)

    async def _read(self, wait: float) -> int:
        """Take every waiting message from every watched subject; returns how many arrived."""
        count = 0
        for subject, subscription in self.subscriptions.items():
            while True:
                try:
                    message = await subscription.next_msg(timeout=wait)
                except (TimeoutError, nats.errors.TimeoutError):
                    break
                self.seen.append((subject, json.loads(message.data), message.headers))
                count += 1
        return count

    @staticmethod
    def _job_of(event: dict[str, Any]) -> str:
        if "data" in event:
            return str(event["data"].get("job_id", ""))
        return str(event.get("event", {}).get("data", {}).get("job_id", ""))  # a dead-letter record

    def headers(self, event_id: str) -> dict[str, str] | None:
        return next((h for _, e, h in self.seen if e.get("id") == event_id), None)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def platform(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[Platform]:
    durable = "test-" + new_id("worker")[-12:].lower()
    settings = Settings(
        grpc_port=_free_port(),
        http_port=_free_port(),
        mongo_database="likho_transcription_test",
        language_grpc_addr=f"127.0.0.1:{_free_port()}",
        media_grpc_addr=f"127.0.0.1:{_free_port()}",
        rpc_timeout_seconds=3,
        job_durable=durable,
        job_start="new",
        job_max_deliver=2,
        job_retry_delay_seconds=0.2,
        job_heartbeat_seconds=0.5,
        log_level="WARNING",
    )
    mongo_host, mongo_port = settings.mongo_url.split("//", 1)[1].split("/")[0].split(":")
    nats_host, nats_port = settings.nats_url.split("//", 1)[1].split(":")
    missing = [
        name
        for name, host, port in (("MongoDB", mongo_host, mongo_port), ("NATS", nats_host, nats_port))
        if not _reachable(host, int(port))
    ]
    if missing:
        message = f"{' and '.join(missing)} not reachable; start the likho-infra stack"
        if os.environ.get("LIKHO_REQUIRE_STACK") == "1":
            pytest.fail(message)
        pytest.skip(message)

    # Audio files, served over HTTP the way presigned links are.
    files = tmp_path_factory.mktemp("media")
    write_silent_wav(files / "call.wav")
    (files / "broken.wav").write_bytes(b"this is not audio")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(files))
    handler.log_message = lambda *args: None  # type: ignore[attr-defined]
    web = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=web.serve_forever, daemon=True).start()

    language, media = FakeLanguage(), FakeMedia(f"http://127.0.0.1:{web.server_address[1]}")
    language_server, media_server = grpc.aio.server(), grpc.aio.server()
    language_pb2_grpc.add_LanguageServiceServicer_to_server(language, language_server)
    media_pb2_grpc.add_MediaServiceServicer_to_server(media, media_server)
    language_server.add_insecure_port(settings.language_grpc_addr)
    media_server.add_insecure_port(settings.media_grpc_addr)
    await language_server.start()
    await media_server.start()

    pipeline = SwitchablePipeline()

    def engine_factory(size: str) -> Engine:
        return Engine(
            EngineSettings(model_size=size),
            model=FakeModel("hi", 0.91, [("hi", 0.91), ("ur", 0.07)]),
            pipeline=pipeline,
        )

    connection = await nats.connect(settings.nats_url)
    js = connection.jetstream()
    subscriptions = {
        subject: await js.subscribe(subject, config=ConsumerConfig(deliver_policy=DeliverPolicy.NEW))
        for subject in WATCHED
    }

    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop, engine_factory))
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{settings.grpc_port}")
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=30)
    except TimeoutError:
        stop.set()
        await task  # surfaces the start-up error
        raise

    store = TranscriptStore(settings.mongo_url, settings.mongo_database)
    yield Platform(
        settings,
        transcription_pb2_grpc.TranscriptionServiceStub(channel),
        channel,
        language,
        media,
        pipeline,
        store,
        js,
        subscriptions,
    )

    await channel.close()
    stop.set()
    await asyncio.wait_for(task, timeout=10)  # an idle service stops at once
    with contextlib.suppress(Exception):
        await js.delete_consumer("LIKHO", durable)
    await connection.close()
    await store._client.drop_database(settings.mongo_database)
    await store.close()
    await language_server.stop(None)
    await media_server.stop(None)
    web.shutdown()


@pytest.fixture
def fresh(platform: Platform) -> Platform:
    """The platform with the stand-ins back in their normal state."""
    platform.language.reset()
    platform.pipeline.current = FakePipeline()
    return platform


def contract(name: str) -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "contracts" / name).read_text(encoding="utf-8"))
