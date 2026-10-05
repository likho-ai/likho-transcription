"""The running service: jobs from the event bus, the gRPC API, and what happens when things go wrong."""

import asyncio
import threading
import urllib.error
import urllib.request

import grpc
import jsonschema
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from likho.common.v1 import common_pb2
from likho.transcription.v1 import transcription_pb2 as pb

from likho_transcription.ids import new_id
from tests.conftest import Platform, contract
from tests.fakes import FakePipeline

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]

STARTED, SEGMENT, COMPLETED, FAILED, DEAD = (
    "likho.transcription.started",
    "likho.live.segment",
    "likho.transcription.completed",
    "likho.transcription.failed",
    "likho.dead",
)
HINGLISH = ["namaste, aapka order kal tak pahunch jayega", "dawa din mein do baar lijiye"]
SCRIPT = ["नमस्ते, आपका ऑर्डर कल तक पहुँच जाएगा", "दवा दिन में दो बार लीजिए"]


def valid(event: dict, schema_file: str) -> None:
    checker = jsonschema.FormatChecker()
    jsonschema.Draft202012Validator(contract("cloudevent.schema.json"), format_checker=checker).validate(event)
    jsonschema.Draft202012Validator(contract(schema_file), format_checker=checker).validate(event["data"])


async def _http_status(port: int, path: str) -> int:
    def get() -> int:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
                return int(response.status)
        except urllib.error.HTTPError as error:
            return error.code

    return await asyncio.to_thread(get)


async def test_health(fresh: Platform) -> None:
    health = health_pb2_grpc.HealthStub(fresh.channel)
    reply = await health.Check(health_pb2.HealthCheckRequest(service="likho.transcription.v1.TranscriptionService"))
    assert reply.status == health_pb2.HealthCheckResponse.SERVING
    assert await _http_status(fresh.settings.http_port, "/healthz") == 200
    assert await _http_status(fresh.settings.http_port, "/readyz") == 200
    assert await _http_status(fresh.settings.http_port, "/metrics") == 200


async def test_a_job_from_the_bus_becomes_live_lines_and_a_stored_transcript(fresh: Platform) -> None:
    ids = await fresh.request()
    events = await fresh.events(ids["job_id"], until=COMPLETED)

    # Said first, before any line: the job is in hand.
    started = next(event for subject, event in events if subject == STARTED)
    valid(started, "likho.transcription.started.v1.schema.json")
    assert started["data"]["job_id"] == ids["job_id"] and started["data"]["attempt"] == 1
    assert [subject for subject, _ in events].index(STARTED) < [subject for subject, _ in events].index(SEGMENT)

    segments = [event for subject, event in events if subject == SEGMENT]
    assert [e["data"]["segment"]["text_roman"] for e in segments] == HINGLISH
    assert [e["data"]["segment"]["text_script"] for e in segments] == SCRIPT
    assert [e["data"]["segment"]["index"] for e in segments] == [0, 1]
    assert all(e["data"]["total_seconds"] == 10.0 for e in segments)
    for event in segments:
        valid(event, "likho.transcription.segment.v1.schema.json")
        assert fresh.headers(event["id"])["Nats-Msg-Id"] == event["id"]  # type: ignore[index]

    completed = next(event for subject, event in events if subject == COMPLETED)
    valid(completed, "likho.transcription.completed.v1.schema.json")
    data = completed["data"]
    assert (data["recording_id"], data["workspace_id"], data["version"]) == (
        ids["recording_id"],
        ids["workspace_id"],
        1,
    )
    assert data["language"] == {
        "detected": "hi",
        "probability": 0.91,
        "candidates": [{"language": "hi", "probability": 0.91}, {"language": "ur", "probability": 0.07}],
        "decoded_as": "hi",
        "policy": "auto",
    }
    assert (data["stats"]["segments"], data["stats"]["audio_seconds"], data["stats"]["chunks"]) == (2, 10.0, 1)

    transcript = (await fresh.stub.GetTranscript(pb.GetTranscriptRequest(id=data["transcript_id"]))).transcript
    assert [(s.text_script, s.text_roman) for s in transcript.segments] == list(zip(SCRIPT, HINGLISH, strict=True))
    assert (transcript.job_id, transcript.version, transcript.script) == (
        ids["job_id"],
        1,
        common_pb2.SCRIPT_DEVANAGARI,
    )
    assert (transcript.model.registry_id, transcript.model.engine) == ("faster-whisper/turbo", "faster-whisper")
    assert transcript.language.candidates[1].language == "ur"
    assert transcript.created_at.seconds > 0
    # the workspace's glossary reached the speech model as hotwords
    assert fresh.pipeline.current.calls[0]["hotwords"] == "त्रिफला"
    stored = await fresh.store.get(data["transcript_id"])
    assert stored is not None and stored["vocabulary_version"] == 3


async def test_a_recording_is_not_transcribed_twice_unless_forced(fresh: Platform) -> None:
    first = await fresh.request()
    done = next(e for s, e in await fresh.events(first["job_id"], until=COMPLETED) if s == COMPLETED)
    assert len(fresh.pipeline.current.calls) == 1

    again = await fresh.request(recording_id=first["recording_id"])
    repeat = next(e for s, e in await fresh.events(again["job_id"], until=COMPLETED) if s == COMPLETED)
    assert repeat["data"]["transcript_id"] == done["data"]["transcript_id"]  # the existing one is reported
    assert repeat["data"]["job_id"] == again["job_id"]
    assert len(fresh.pipeline.current.calls) == 1  # the model did not run again

    forced = await fresh.request(recording_id=first["recording_id"], force=True)
    redo = next(e for s, e in await fresh.events(forced["job_id"], until=COMPLETED) if s == COMPLETED)
    assert (redo["data"]["version"], len(fresh.pipeline.current.calls)) == (2, 2)

    versions = (
        await fresh.stub.ListTranscripts(pb.ListTranscriptsRequest(recording_id=first["recording_id"]))
    ).transcripts
    assert [t.version for t in versions] == [2, 1]  # newest first
    assert all(len(t.segments) == 0 for t in versions)  # the list carries no lines


@pytest.mark.parametrize(
    ("media_id", "message"),
    [("med_unknown", "not found"), ("med_gone", "could not be downloaded"), ("med_broken", "could not be read")],
)
async def test_audio_that_cannot_be_used_fails_at_once(fresh: Platform, media_id: str, message: str) -> None:
    ids = await fresh.request(media_id=media_id)
    events = await fresh.events(ids["job_id"], until=DEAD)

    failed = next(event for subject, event in events if subject == FAILED)
    valid(failed, "likho.transcription.failed.v1.schema.json")
    assert (failed["data"]["code"], failed["data"]["attempt"]) == ("audio_unreadable", 1)  # trying again cannot help
    assert message in failed["data"]["message"]
    dead = next(event for subject, event in events if subject == DEAD)
    assert dead["reason"]["code"] == "audio_unreadable"
    assert dead["event"]["data"]["job_id"] == ids["job_id"]  # the request is kept for a person to look at
    assert await fresh.store.find_by_job(ids["job_id"]) is None


async def test_a_service_that_is_down_is_tried_again_before_giving_up(fresh: Platform) -> None:
    ids = await fresh.request(media_id="med_down")
    events = await fresh.events(ids["job_id"], until=FAILED)
    failed = next(event for subject, event in events if subject == FAILED)
    assert failed["data"]["code"] == "internal"
    assert failed["data"]["attempt"] == 2  # job_max_deliver in these tests


async def test_a_running_job_can_be_cancelled(fresh: Platform) -> None:
    gate = threading.Event()
    fresh.pipeline.current = FakePipeline(gate=gate)
    ids = await fresh.request()
    try:
        await asyncio.wait_for(asyncio.to_thread(fresh.pipeline.current.first_line_out.wait, 15), timeout=20)
        assert (await fresh.stub.CancelJob(pb.CancelJobRequest(job_id=ids["job_id"]))).cancelled is True
    finally:
        gate.set()

    events = await fresh.events(ids["job_id"], until=FAILED)
    failed = next(event for subject, event in events if subject == FAILED)
    assert failed["data"]["code"] == "cancelled"
    assert not any(subject == DEAD for subject, _ in events)  # a cancelled job is not an error to investigate
    assert await fresh.store.find_by_job(ids["job_id"]) is None
    assert (await fresh.stub.CancelJob(pb.CancelJobRequest(job_id=ids["job_id"]))).cancelled is False


async def test_a_job_cancelled_before_the_worker_has_it_is_dropped(fresh: Platform) -> None:
    # likho-api gives up on a job (it stalled on a worker that died, or a person cancelled it while
    # it waited) and asks this worker to stop it before its request gets here: when the request
    # comes, delivered once more, it is dropped instead of transcribed.
    job_id = new_id("job")
    assert (await fresh.stub.CancelJob(pb.CancelJobRequest(job_id=job_id))).cancelled is False
    await fresh.request(job_id=job_id)

    events = await fresh.events(job_id, until=FAILED)
    failed = next(event for subject, event in events if subject == FAILED)
    assert failed["data"]["code"] == "cancelled"
    assert not any(subject in (SEGMENT, COMPLETED, DEAD) for subject, _ in events)
    assert fresh.pipeline.current.calls == []  # the model did not run
    assert await fresh.store.find_by_job(job_id) is None


async def test_without_the_language_service_the_built_in_rules_are_used(fresh: Platform) -> None:
    fresh.language.down = True
    ids = await fresh.request()
    completed = next(e for s, e in await fresh.events(ids["job_id"], until=COMPLETED) if s == COMPLETED)
    stored = await fresh.store.get(completed["data"]["transcript_id"])
    assert stored is not None
    assert [line["text_roman"] for line in stored["segments"]] == HINGLISH
    assert stored["vocabulary_version"] == 0  # marks the transcript as written without the workspace's vocabulary
    assert fresh.pipeline.current.calls[0]["hotwords"] is None


async def test_retransliterate_rebuilds_hinglish_without_running_the_model(fresh: Platform) -> None:
    ids = await fresh.request()
    completed = next(e for s, e in await fresh.events(ids["job_id"], until=COMPLETED) if s == COMPLETED)

    fresh.language.spellings = {"दवा": "medicine"}
    fresh.language.version = 4
    rebuilt = (
        await fresh.stub.Retransliterate(pb.RetransliterateRequest(transcript_id=completed["data"]["transcript_id"]))
    ).transcript

    assert (rebuilt.version, rebuilt.job_id, rebuilt.recording_id) == (2, "", ids["recording_id"])
    assert [s.text_script for s in rebuilt.segments] == SCRIPT  # layer 1 is untouched
    assert [s.text_roman for s in rebuilt.segments] == [HINGLISH[0], "medicine din mein do baar lijiye"]
    assert len(fresh.pipeline.current.calls) == 1  # only the first transcription used the model
    stored = await fresh.store.get(rebuilt.id)
    assert stored is not None and stored["vocabulary_version"] == 4


async def test_a_correction_makes_the_next_version_and_is_announced(fresh: Platform) -> None:
    ids = await fresh.request()
    completed = next(e for s, e in await fresh.events(ids["job_id"], until=COMPLETED) if s == COMPLETED)
    first = completed["data"]["transcript_id"]

    # The Hinglish of a line, as a person wrote it: it stands as written.
    reply = await fresh.stub.CorrectSegment(
        pb.CorrectSegmentRequest(
            transcript_id=first,
            segment_index=1,
            layer=pb.LAYER_ROMAN,
            text="dawai din mein do baar lijiye",
            user_id="usr_01JB7Z5K3M9Q2W4X6Y8A0C1E3G",
            workspace_id=ids["workspace_id"],
        )
    )
    assert (reply.transcript.version, reply.transcript.job_id) == (2, "")
    assert reply.transcript.segments[1].text_roman == "dawai din mein do baar lijiye"
    assert reply.transcript.segments[1].text_script == SCRIPT[1]  # the other layer is untouched
    assert reply.transcript.segments[0].text_roman == HINGLISH[0]
    assert (reply.correction.layer, reply.correction.before, reply.correction.after) == (
        pb.LAYER_ROMAN,
        HINGLISH[1],
        "dawai din mein do baar lijiye",
    )
    assert reply.correction.corrected_transcript_id == reply.transcript.id

    # The script layer corrected: the Hinglish of that line is derived again (through the language service).
    fresh.language.spellings = {"गोली": "tablet"}
    second = await fresh.stub.CorrectSegment(
        pb.CorrectSegmentRequest(
            transcript_id=reply.transcript.id,
            segment_index=0,
            layer=pb.LAYER_SCRIPT,
            text="गोली सुबह लीजिए",
            user_id="usr_01JB7Z5K3M9Q2W4X6Y8A0C1E3G",
            workspace_id=ids["workspace_id"],
        )
    )
    assert second.transcript.version == 3
    assert second.transcript.segments[0].text_script == "गोली सुबह लीजिए"
    assert second.transcript.segments[0].text_roman == "tablet subah lijiye"
    assert second.transcript.segments[1].text_roman == "dawai din mein do baar lijiye"  # the earlier correction stays

    # Both corrections are kept, newest first, and were announced in the shape of the contract.
    listed = (await fresh.stub.ListCorrections(pb.ListCorrectionsRequest(recording_id=ids["recording_id"]))).corrections
    assert [c.segment_index for c in listed] == [0, 1]
    await fresh._read(wait=0.5)
    announced = [
        e
        for s, e, _ in fresh.seen
        if s == "likho.transcript.corrected" and e["data"]["recording_id"] == ids["recording_id"]
    ]
    assert len(announced) == 2
    assert announced[-1]["data"] == {
        "transcript_id": second.transcript.id,
        "recording_id": ids["recording_id"],
        "workspace_id": ids["workspace_id"],
        "user_id": "usr_01JB7Z5K3M9Q2W4X6Y8A0C1E3G",
        "segment_index": 0,
        "layer": "script",
        "before": SCRIPT[0],
        "after": "गोली सुबह लीजिए",
    }
    valid(announced[-1], "likho.transcript.corrected.v1.schema.json")

    # Refusals: an older version, a line that is not there, the same text.
    with pytest.raises(grpc.aio.AioRpcError) as refused:
        await fresh.stub.CorrectSegment(
            pb.CorrectSegmentRequest(transcript_id=first, segment_index=0, layer=pb.LAYER_ROMAN, text="x", user_id="u")
        )
    assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    with pytest.raises(grpc.aio.AioRpcError) as missing:
        await fresh.stub.CorrectSegment(
            pb.CorrectSegmentRequest(
                transcript_id=second.transcript.id, segment_index=9, layer=pb.LAYER_ROMAN, text="x", user_id="u"
            )
        )
    assert missing.value.code() == grpc.StatusCode.NOT_FOUND


async def test_transcribe_streams_started_lines_and_the_result(fresh: Platform) -> None:
    request = pb.TranscribeRequest(
        recording_id="rec_direct", media_id="med_good", workspace_id="wsp_1", language_policy="auto"
    )
    replies = [reply async for reply in fresh.stub.Transcribe(request)]

    assert [reply.WhichOneof("event") for reply in replies] == ["started", "segment", "segment", "completed"]
    assert replies[0].started.audio_seconds == 10.0
    assert (replies[0].started.language.detected, replies[0].started.language.decoded_as) == ("hi", "hi")
    assert [replies[1].segment.text_roman, replies[2].segment.text_roman] == HINGLISH
    assert replies[3].completed.id.startswith("trn_")
    assert len(replies[3].completed.segments) == 2
    await fresh.store.delete_recording("rec_direct")


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (lambda s: s.GetTranscript(pb.GetTranscriptRequest(id="trn_missing")), grpc.StatusCode.NOT_FOUND),
        (lambda s: s.ListTranscripts(pb.ListTranscriptsRequest()), grpc.StatusCode.INVALID_ARGUMENT),
        (
            lambda s: s.Retransliterate(pb.RetransliterateRequest(transcript_id="trn_missing")),
            grpc.StatusCode.NOT_FOUND,
        ),
    ],
    ids=["unknown-transcript", "no-recording-id", "retransliterate-unknown"],
)
async def test_bad_requests_get_a_clear_status(fresh: Platform, call: object, code: grpc.StatusCode) -> None:
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await call(fresh.stub)  # type: ignore[operator]
    assert error.value.code() == code
    assert error.value.details()


async def test_transcribe_refuses_incomplete_requests_and_reports_failures(fresh: Platform) -> None:
    with pytest.raises(grpc.aio.AioRpcError) as error:
        _ = [r async for r in fresh.stub.Transcribe(pb.TranscribeRequest(recording_id="rec_x"))]
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "media_id is required" in (error.value.details() or "")

    with pytest.raises(grpc.aio.AioRpcError) as error:
        _ = [
            r
            async for r in fresh.stub.Transcribe(
                pb.TranscribeRequest(recording_id="rec_x", media_id="med_broken", workspace_id="wsp_1")
            )
        ]
    assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "could not be read" in (error.value.details() or "")


async def test_engines_are_listed_with_the_default_marked(fresh: Platform) -> None:
    engines = (await fresh.stub.ListEngines(pb.ListEnginesRequest())).engines
    assert [e.registry_id for e in engines][:2] == ["faster-whisper/turbo", "faster-whisper/large-v3"]
    assert [e.registry_id for e in engines if e.is_default] == ["faster-whisper/turbo"]


async def test_a_message_that_is_not_a_job_is_dropped_and_work_goes_on(fresh: Platform) -> None:
    await fresh.js.publish("likho.transcription.requested", b"not json at all")
    ids = await fresh.request()
    events = await fresh.events(ids["job_id"], until=COMPLETED)
    assert any(subject == COMPLETED for subject, _ in events)
