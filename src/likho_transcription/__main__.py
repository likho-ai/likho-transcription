"""Starts the service:  python -m likho_transcription   (or the `likho-transcription` command)."""

import asyncio
import contextlib
import json
import logging
import signal
import sys
from datetime import UTC, datetime

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from likho.transcription.v1 import transcription_pb2, transcription_pb2_grpc

from likho_transcription import __version__
from likho_transcription.consumer import JobConsumer
from likho_transcription.events import EventBus
from likho_transcription.gateways import LanguageGateway, MediaGateway
from likho_transcription.grpc_server import TranscriptionServicer
from likho_transcription.health import start_health_server
from likho_transcription.metrics import shared
from likho_transcription.runner import EngineFactory, EngineProvider, JobRunner
from likho_transcription.settings import Settings
from likho_transcription.store import TranscriptStore

log = logging.getLogger("likho_transcription")

SERVICE_NAME = transcription_pb2.DESCRIPTOR.services_by_name["TranscriptionService"].full_name


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    for noisy in ("faster_whisper", "huggingface_hub", "httpx", "httpcore", "pymongo"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def serve(
    settings: Settings, stop: asyncio.Event | None = None, engine_factory: EngineFactory | None = None
) -> None:
    """Run until `stop` is set. `engine_factory` lets tests supply an engine with a stand-in model."""
    stop = stop or asyncio.Event()

    metrics = shared(__version__, settings.otel_exporter_otlp_endpoint)
    store = TranscriptStore(settings.mongo_url, settings.mongo_database)
    await store.prepare()
    bus = EventBus(settings.nats_url)
    await bus.connect(settings.nats_connect_timeout_seconds)

    language = LanguageGateway(settings.language_grpc_addr, settings.rpc_timeout_seconds)
    media = MediaGateway(settings.media_grpc_addr, settings.rpc_timeout_seconds, settings.download_timeout_seconds)
    engines = EngineProvider(settings, engine_factory)
    runner = JobRunner(settings, store, language, media, engines)

    server = grpc.aio.server()
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServicer(store, runner, engines, language, bus), server
    )
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    server.add_insecure_port(f"[::]:{settings.grpc_port}")
    await server.start()

    async def ready() -> bool:
        return bus.connected and await store.ping()

    http = await start_health_server(settings.http_port, ready, metrics.scrape)

    working: asyncio.Task[None] | None = None
    if settings.worker_enabled:
        consumer = JobConsumer(settings, bus, runner, store, metrics)
        await consumer.start()
        working = asyncio.create_task(consumer.run(stop))

    log.info(
        "likho-transcription %s: gRPC on %d, health on %d, worker %s",
        __version__,
        settings.grpc_port,
        settings.http_port,
        "on" if settings.worker_enabled else "off",
    )
    await stop.wait()

    log.info("stopping: finishing the job in hand")
    await health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.NOT_SERVING)
    if working is not None:
        await working
    await server.stop(grace=10)
    http.close()
    await http.wait_closed()
    await language.close()
    await media.close()
    await bus.close()
    await store.close()
    await asyncio.to_thread(metrics.flush)


async def _run() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        # Windows has no loop.add_signal_handler; signal.signal works for Ctrl+C there.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop.set)
            continue
        signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))
    await serve(settings, stop)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
