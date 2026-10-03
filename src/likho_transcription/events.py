"""This service's events: CloudEvents on NATS JetStream.

Contracts: likho-contracts/events/likho.transcription.{segment,completed,failed}.v1.schema.json
and streams.yaml for the subjects. Event ids are derived from the job, so publishing the
same event again (a retry after a crash) is stored once.
"""

import contextlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

import nats
from nats.js.errors import NotFoundError

from likho_engine import Segment
from likho_transcription.runner import JobRequest

log = logging.getLogger(__name__)

SOURCE = "likho-transcription"
REQUESTED_SUBJECT = "likho.transcription.requested"
SEGMENT_SUBJECT = "likho.live.segment"
COMPLETED_SUBJECT = "likho.transcription.completed"
FAILED_SUBJECT = "likho.transcription.failed"
DEAD_SUBJECT = "likho.dead"
CORRECTED_SUBJECT = "likho.transcript.corrected"
JOB_STREAM = "LIKHO"

Event = dict[str, Any]


def _envelope(event_id: str, event_type: str, subject: str, data: dict[str, Any]) -> Event:
    return {
        "specversion": "1.0",
        "id": event_id,
        "source": SOURCE,
        "type": event_type,
        "time": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "subject": subject,
        "datacontenttype": "application/json",
        "data": data,
    }


def segment_event(job: JobRequest, segment: Segment, total_seconds: float, attempt: int) -> Event:
    return _envelope(
        f"evt_{job.job_id}_segment_{segment.index}_{attempt}",
        "likho.transcription.segment.v1",
        job.recording_id,
        {
            "job_id": job.job_id,
            "recording_id": job.recording_id,
            "total_seconds": total_seconds,
            "segment": {
                "index": segment.index,
                "start_seconds": segment.start,
                "end_seconds": segment.end,
                "text_script": segment.text_script,
                "text_roman": segment.text_roman,
            },
        },
    )


def completed_event(job: JobRequest, document: dict[str, Any]) -> Event:
    stats = document["stats"]
    return _envelope(
        f"evt_{job.job_id}_completed",
        "likho.transcription.completed.v1",
        job.recording_id,
        {
            "job_id": job.job_id,
            "recording_id": job.recording_id,
            "transcript_id": document["_id"],
            "workspace_id": job.workspace_id,
            "version": document["version"],
            "language": document["language"],
            "stats": {
                "audio_seconds": stats["audio_seconds"],
                "elapsed_seconds": stats["elapsed_seconds"],
                "segments": len(document.get("segments", [])) or stats.get("segments", 0),
                "chunks": stats["chunks"],
                "silence_skipped_seconds": stats["silence_skipped_seconds"],
            },
        },
    )


def corrected_event(correction: dict[str, Any], workspace_id: str) -> Event:
    """likho.transcript.corrected.v1: one line a person changed; the corrected version is what to index."""
    return _envelope(
        "evt_" + correction["_id"][4:],
        "likho.transcript.corrected.v1",
        correction["corrected_transcript_id"],
        {
            "transcript_id": correction["corrected_transcript_id"],
            "recording_id": correction["recording_id"],
            "workspace_id": workspace_id,
            "user_id": correction["user_id"],
            "segment_index": correction["segment_index"],
            "layer": correction["layer"],
            "before": correction["before"],
            "after": correction["after"],
        },
    )


def failed_event(job: JobRequest, code: str, message: str, attempt: int) -> Event:
    return _envelope(
        f"evt_{job.job_id}_failed_{attempt}",
        "likho.transcription.failed.v1",
        job.recording_id,
        {
            "job_id": job.job_id,
            "recording_id": job.recording_id,
            "workspace_id": job.workspace_id,
            "code": code,
            "message": message,
            "attempt": attempt,
        },
    )


class EventBus:
    def __init__(self, url: str) -> None:
        self._url = url
        self.nc: Any = None
        self.js: Any = None

    async def connect(self) -> None:
        self.nc = await nats.connect(self._url, name=SOURCE, max_reconnect_attempts=-1)
        self.js = self.nc.jetstream()
        try:
            await self.js.stream_info(JOB_STREAM)
        except NotFoundError as error:
            raise RuntimeError(
                f"stream {JOB_STREAM} does not exist on {self._url}; "
                "create the streams first (likho-infra: scripts/up.sh)"
            ) from error

    @property
    def connected(self) -> bool:
        return self.nc is not None and self.nc.is_connected

    async def close(self) -> None:
        if self.nc is not None:
            # Every publish was already confirmed by the server, so there is nothing to drain.
            # drain() would also wait for the job subscription, which never ends by itself.
            with contextlib.suppress(Exception):
                await self.nc.flush(timeout=2)
            await self.nc.close()
            self.nc = None

    async def publish(self, subject: str, event: Event) -> None:
        await self.js.publish(
            subject,
            json.dumps(event, ensure_ascii=False).encode(),
            headers={"Nats-Msg-Id": event["id"]},
            timeout=5,
        )
