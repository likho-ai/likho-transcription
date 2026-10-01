"""likho.transcription.v1.TranscriptionService over gRPC."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import grpc
from google.protobuf.timestamp_pb2 import Timestamp
from likho.common.v1 import common_pb2
from likho.transcription.v1 import transcription_pb2 as pb
from likho.transcription.v1 import transcription_pb2_grpc

from likho_engine import Decision, Detection, Segment
from likho_transcription.errors import JobCancelled, JobError
from likho_transcription.gateways import LanguageGateway
from likho_transcription.ids import new_id
from likho_transcription.runner import EngineProvider, JobRequest, JobRunner
from likho_transcription.store import Document, TranscriptStore

log = logging.getLogger(__name__)

_SCRIPT = {
    "devanagari": common_pb2.SCRIPT_DEVANAGARI,
    "latin": common_pb2.SCRIPT_LATIN,
    "arabic": common_pb2.SCRIPT_ARABIC,
}
_STATUS = {
    "audio_unreadable": grpc.StatusCode.FAILED_PRECONDITION,
    "model_unavailable": grpc.StatusCode.FAILED_PRECONDITION,
    "internal": grpc.StatusCode.INTERNAL,
}


def _language(detected: dict[str, Any]) -> common_pb2.LanguageDetection:
    return common_pb2.LanguageDetection(
        detected=detected["detected"],
        probability=detected["probability"],
        candidates=[
            common_pb2.LanguageCandidate(language=c["language"], probability=c["probability"])
            for c in detected.get("candidates", [])
        ],
        decoded_as=detected["decoded_as"],
        policy=detected["policy"],
    )


def _segment(line: dict[str, Any]) -> common_pb2.Segment:
    return common_pb2.Segment(
        index=line["index"],
        start_seconds=line["start"],
        end_seconds=line["end"],
        text_script=line["text_script"],
        text_roman=line["text_roman"],
    )


def to_proto(document: Document) -> pb.Transcript:
    created = Timestamp()
    created.FromDatetime(document["created_at"])
    stats = document["stats"]
    return pb.Transcript(
        id=document["_id"],
        recording_id=document["recording_id"],
        job_id=document["job_id"],
        version=document["version"],
        model=pb.ModelRef(**document["model"]),
        language=_language(document["language"]),
        script=_SCRIPT.get(document["script"], common_pb2.SCRIPT_UNSPECIFIED),
        segments=[_segment(line) for line in document.get("segments", [])],
        stats=pb.TranscriptStats(
            audio_seconds=stats["audio_seconds"],
            elapsed_seconds=stats["elapsed_seconds"],
            realtime_factor=stats["realtime_factor"],
            chunks=stats["chunks"],
            silence_skipped_seconds=stats["silence_skipped_seconds"],
        ),
        created_at=created,
    )


class TranscriptionServicer(transcription_pb2_grpc.TranscriptionServiceServicer):
    def __init__(
        self, store: TranscriptStore, runner: JobRunner, engines: EngineProvider, language: LanguageGateway
    ) -> None:
        self._store = store
        self._runner = runner
        self._engines = engines
        self._language = language

    async def GetTranscript(
        self, request: pb.GetTranscriptRequest, context: grpc.aio.ServicerContext
    ) -> pb.GetTranscriptResponse:
        document = await self._store.get(request.id)
        if document is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"transcript {request.id!r} not found")
        assert document is not None  # abort raises
        return pb.GetTranscriptResponse(transcript=to_proto(document))

    async def ListTranscripts(
        self, request: pb.ListTranscriptsRequest, context: grpc.aio.ServicerContext
    ) -> pb.ListTranscriptsResponse:
        if not request.recording_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "recording_id is required")
        documents = await self._store.list_for_recording(request.recording_id)
        return pb.ListTranscriptsResponse(transcripts=[to_proto(document) for document in documents])

    async def Transcribe(
        self, request: pb.TranscribeRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[pb.TranscribeResponse]:
        """Transcribes now and streams what happens: started, each line, the stored transcript."""
        for name in ("recording_id", "media_id", "workspace_id"):
            if not getattr(request, name):
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{name} is required")
        job = JobRequest(
            job_id=new_id("job"),
            recording_id=request.recording_id,
            media_id=request.media_id,
            workspace_id=request.workspace_id,
            model_registry_id=request.model_registry_id,
            language_policy=request.language_policy or "auto",
            force=True,
        )
        replies: asyncio.Queue[pb.TranscribeResponse | BaseException | None] = asyncio.Queue()

        async def on_started(seconds: float, detection: Detection, decision: Decision) -> None:
            language = common_pb2.LanguageDetection(
                detected=detection.language,
                probability=detection.probability,
                candidates=[common_pb2.LanguageCandidate(language=c, probability=p) for c, p in detection.candidates],
                decoded_as=decision.decode_as,
                policy=job.language_policy,
            )
            await replies.put(
                pb.TranscribeResponse(started=pb.TranscribeStarted(audio_seconds=seconds, language=language))
            )

        async def on_segment(segment: Segment, _total: float) -> None:
            line = common_pb2.Segment(
                index=segment.index,
                start_seconds=segment.start,
                end_seconds=segment.end,
                text_script=segment.text_script,
                text_roman=segment.text_roman,
            )
            await replies.put(pb.TranscribeResponse(segment=line))

        async def work() -> None:
            try:
                document = await self._runner.run(job, on_started, on_segment)
                await replies.put(pb.TranscribeResponse(completed=to_proto(document)))
                await replies.put(None)
            except BaseException as error:
                await replies.put(error)

        task = asyncio.create_task(work())
        try:
            while True:
                reply = await replies.get()
                if reply is None:
                    return
                if isinstance(reply, JobCancelled):
                    await context.abort(grpc.StatusCode.CANCELLED, "The job was cancelled")
                if isinstance(reply, JobError):
                    await context.abort(_STATUS.get(reply.code, grpc.StatusCode.INTERNAL), reply.message)
                if isinstance(reply, BaseException):
                    raise reply
                yield reply
        finally:
            if not task.done():  # the caller went away: stop decoding
                self._runner.cancel(job.job_id)

    async def Retransliterate(
        self, request: pb.RetransliterateRequest, context: grpc.aio.ServicerContext
    ) -> pb.RetransliterateResponse:
        """A new version whose Hinglish is rebuilt from the saved script layer. The speech model does not run."""
        document = await self._store.get(request.transcript_id)
        if document is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"transcript {request.transcript_id!r} not found")
        assert document is not None  # abort raises
        if not document.get("transliterated", True):
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "this transcript has no Hinglish layer to rebuild")
        lines = document["segments"]
        texts, version = await self._language.to_roman(
            document["workspace_id"], [line["text_script"] for line in lines]
        )
        rebuilt = {key: value for key, value in document.items() if key not in ("_id", "version", "created_at")}
        rebuilt["job_id"] = ""
        rebuilt["vocabulary_version"] = version
        rebuilt["segments"] = [{**line, "text_roman": text} for line, text in zip(lines, texts, strict=True)]
        stored = await self._store.insert(rebuilt)
        return pb.RetransliterateResponse(transcript=to_proto(stored))

    async def ListEngines(
        self, request: pb.ListEnginesRequest, context: grpc.aio.ServicerContext
    ) -> pb.ListEnginesResponse:
        default = self._engines.default_registry_id
        return pb.ListEnginesResponse(
            engines=[
                pb.Engine(
                    registry_id=registry_id,
                    engine=registry_id.split("/")[0],
                    output_script=common_pb2.SCRIPT_DEVANAGARI,
                    available=True,  # models are downloaded on first use
                    is_default=registry_id == default,
                )
                for registry_id in self._engines.registry_ids()
            ]
        )

    async def CancelJob(self, request: pb.CancelJobRequest, context: grpc.aio.ServicerContext) -> pb.CancelJobResponse:
        return pb.CancelJobResponse(cancelled=self._runner.cancel(request.job_id))
