"""Takes transcription jobs from the event bus, one at a time.

The job queue is the JetStream subject likho.transcription.requested, read through one
durable pull consumer shared by every worker. A worker fetches one job and the next only
when it is done. While it works it tells the bus "in progress", so a long call is not
handed to another worker; if the worker dies, the bus gives the job to someone else.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any

from nats.errors import TimeoutError as NatsTimeoutError
from nats.js.api import AckPolicy, ConsumerConfig, DeliverPolicy

from likho_engine import Decision, Detection, Segment
from likho_transcription import events
from likho_transcription.errors import JobCancelled, JobError
from likho_transcription.events import EventBus
from likho_transcription.metrics import Metrics
from likho_transcription.runner import JobRequest, JobRunner
from likho_transcription.settings import Settings
from likho_transcription.store import TranscriptStore

log = logging.getLogger(__name__)


def parse_request(raw: bytes) -> JobRequest:
    """Read a likho.transcription.requested.v1 event. Raises ValueError when it is not one."""
    try:
        event = json.loads(raw)
        data = event["data"]
        if event.get("type") != "likho.transcription.requested.v1":
            raise ValueError(f"unexpected event type {event.get('type')!r}")
        return JobRequest(
            job_id=str(data["job_id"]),
            recording_id=str(data["recording_id"]),
            media_id=str(data["media_id"]),
            workspace_id=str(data["workspace_id"]),
            model_registry_id=str(data.get("model_registry_id", "")),
            language_policy=str(data.get("language_policy", "auto")) or "auto",
            force=bool(data.get("force", False)),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"not a transcription request: {error}") from error


class JobConsumer:
    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        runner: JobRunner,
        store: TranscriptStore,
        metrics: Metrics | None = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._runner = runner
        self._store = store
        self._metrics = metrics
        self._subscription: Any = None

    async def start(self) -> None:
        config = ConsumerConfig(
            durable_name=self._settings.job_durable,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=self._settings.job_ack_wait_seconds,
            max_deliver=self._settings.job_max_deliver,
            deliver_policy=DeliverPolicy.NEW if self._settings.job_start == "new" else DeliverPolicy.ALL,
            filter_subject=events.REQUESTED_SUBJECT,
        )
        self._subscription = await self._bus.js.pull_subscribe(
            events.REQUESTED_SUBJECT, durable=self._settings.job_durable, stream=events.JOB_STREAM, config=config
        )
        log.info("taking jobs from %s as %s", events.REQUESTED_SUBJECT, self._settings.job_durable)

    async def run(self, stop: asyncio.Event) -> None:
        """Fetch and handle jobs until stop is set. The job in hand is finished first."""
        while not stop.is_set():
            try:
                messages = await self._subscription.fetch(1, timeout=1)
            except (NatsTimeoutError, TimeoutError):
                continue
            except Exception:
                log.exception("could not fetch a job; trying again")
                await asyncio.sleep(1)
                continue
            for message in messages:
                await self.handle(message)
        # Stop listening. The durable consumer stays on the server for the other workers.
        with contextlib.suppress(Exception):
            await self._subscription.unsubscribe()

    async def handle(self, message: Any) -> None:
        attempt = message.metadata.num_delivered
        try:
            job = parse_request(message.data)
        except ValueError as error:
            log.error("dropping a message that is not a job: %s", error)
            await message.term()
            return

        heartbeat = asyncio.create_task(self._heartbeat(message))
        try:
            await self._handle(job, message, attempt)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self, message: Any) -> None:
        while True:
            await asyncio.sleep(self._settings.job_heartbeat_seconds)
            with contextlib.suppress(Exception):
                await message.in_progress()

    def _count(self, outcome: str, document: dict[str, Any] | None = None) -> None:
        if self._metrics is None:
            return
        self._metrics.jobs_finished.add(1, {"outcome": outcome})
        stats = (document or {}).get("stats") or {}
        if stats.get("elapsed_seconds"):
            self._metrics.job_seconds.record(float(stats["elapsed_seconds"]))
        if stats.get("realtime_factor"):
            self._metrics.realtime_factor.record(float(stats["realtime_factor"]))

    async def _handle(self, job: JobRequest, message: Any, attempt: int) -> None:
        log.info("job %s for recording %s, attempt %d", job.job_id, job.recording_id, attempt)
        if self._metrics is not None:
            self._metrics.jobs_running.add(1)
        try:
            # Said first, so likho-api knows the job is in hand while the model loads.
            await self._bus.publish(events.STARTED_SUBJECT, events.started_event(job, attempt))
            document = await self._existing(job)
            if document is None:

                async def on_started(seconds: float, detection: Detection, decision: Decision) -> None:
                    log.info(
                        "job %s: %.0fs of audio, detected %s (%.2f), decoding as %s",
                        job.job_id,
                        seconds,
                        detection.language,
                        detection.probability,
                        decision.decode_as,
                    )

                async def on_segment(segment: Segment, total_seconds: float) -> None:
                    await self._bus.publish(
                        events.SEGMENT_SUBJECT, events.segment_event(job, segment, total_seconds, attempt)
                    )
                    if self._metrics is not None:
                        self._metrics.segments.add(1)

                document = await self._runner.run(job, on_started, on_segment)
                self._count("done", document)
            else:
                self._count("reused")
            await self._bus.publish(events.COMPLETED_SUBJECT, events.completed_event(job, document))
            await message.ack()
            log.info("job %s done: transcript %s version %d", job.job_id, document["_id"], document["version"])
        except JobCancelled:
            self._count("cancelled")
            await self._fail(job, message, "cancelled", "The job was cancelled", attempt, dead=False)
        except JobError as error:
            if error.retryable and attempt < self._settings.job_max_deliver:
                log.warning("job %s attempt %d failed, will retry: %s", job.job_id, attempt, error.message)
                self._count("retry")
                await message.nak(delay=self._settings.job_retry_delay_seconds)
            else:
                self._count("failed")
                await self._fail(job, message, error.code, error.message, attempt, dead=True)
        except Exception as error:
            log.exception("job %s attempt %d failed unexpectedly", job.job_id, attempt)
            if attempt < self._settings.job_max_deliver:
                self._count("retry")
                await message.nak(delay=self._settings.job_retry_delay_seconds)
            else:
                self._count("failed")
                await self._fail(job, message, "internal", f"Transcription failed: {error}", attempt, dead=True)
        finally:
            if self._metrics is not None:
                self._metrics.jobs_running.add(-1)

    async def _existing(self, job: JobRequest) -> dict[str, Any] | None:
        """The transcript to report without transcribing again, if there is one.

        The same job delivered twice (the worker stopped after storing, before acknowledging)
        reports the transcript it already stored. A job for a recording that already has a
        transcript reports that one, unless the job says force.
        """
        document = await self._store.find_by_job(job.job_id)
        if document is None and not job.force:
            document = await self._store.latest_for_recording(job.recording_id)
        return document

    async def _fail(self, job: JobRequest, message: Any, code: str, text: str, attempt: int, *, dead: bool) -> None:
        log.error("job %s failed (%s): %s", job.job_id, code, text)
        await self._bus.publish(events.FAILED_SUBJECT, events.failed_event(job, code, text, attempt))
        if dead:
            # Keep the request that could not be done, with the reason, for a person to look at.
            await self._bus.publish(
                events.DEAD_SUBJECT,
                {
                    "id": f"evt_{job.job_id}_dead",
                    "reason": {"code": code, "message": text, "attempt": attempt},
                    "event": json.loads(message.data),
                },
            )
        await message.term()
