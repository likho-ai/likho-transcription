"""Parts of the service that need no database or event bus."""

from pathlib import Path

import jsonschema
import pytest

from likho_engine import Segment
from likho_transcription import events
from likho_transcription.consumer import parse_request
from likho_transcription.errors import JobError
from likho_transcription.runner import EngineProvider, JobRequest
from likho_transcription.settings import Settings
from tests.conftest import contract

JOB = JobRequest(
    "job_01JB7Z5K3M9Q2W4X6Y8A0C1E3G", "rec_01JB7Z5K3M9Q2W4X6Y8A0C1E3G", "med_1", "wsp_01JB7Z5K3M9Q2W4X6Y8A0C1E3G"
)


def valid(event: dict, schema_file: str) -> None:
    checker = jsonschema.FormatChecker()
    jsonschema.Draft202012Validator(contract("cloudevent.schema.json"), format_checker=checker).validate(event)
    jsonschema.Draft202012Validator(contract(schema_file), format_checker=checker).validate(event["data"])


def test_parse_request_reads_the_contract_example() -> None:
    example = (Path(__file__).parent / "contracts" / "example.requested.json").read_bytes()
    job = parse_request(example)
    assert (job.model_registry_id, job.language_policy, job.force) == ("faster-whisper/turbo", "auto", False)
    assert job.job_id.startswith("job_") and job.media_id.startswith("med_")


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"{}",
        b'{"type": "other.v1", "data": {}}',
        b'{"type": "likho.transcription.requested.v1", "data": {"job_id": "j"}}',
        b'{"type": "likho.transcription.requested.v1", "data": "text"}',
    ],
)
def test_parse_request_rejects_anything_else(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_request(raw)


def test_model_names() -> None:
    provider = EngineProvider(Settings(default_model="turbo"), lambda size: None)  # type: ignore[arg-type, return-value]
    assert provider.default_registry_id == "faster-whisper/turbo"
    assert provider.size_of("") == "turbo"
    assert provider.size_of("faster-whisper/large-v3") == "large-v3"
    assert provider.size_of("small") == "small"
    for unknown in ("faster-whisper/huge", "other-engine/turbo"):
        with pytest.raises(JobError) as error:
            provider.size_of(unknown)
        assert (error.value.code, error.value.retryable) == ("model_unavailable", False)


def test_events_match_the_contracts_and_have_stable_ids() -> None:
    segment = events.segment_event(JOB, Segment(3, 1.5, 4.0, "ठीक है", "theek hai"), 60.0, attempt=1)
    valid(segment, "likho.transcription.segment.v1.schema.json")
    failed = events.failed_event(JOB, "audio_unreadable", "The audio file could not be read", attempt=2)
    valid(failed, "likho.transcription.failed.v1.schema.json")
    document = {
        "_id": "trn_01JB7Z5K3M9Q2W4X6Y8A0C1E3G",
        "version": 1,
        "language": {"detected": "hi", "probability": 0.9, "candidates": [], "decoded_as": "hi", "policy": "auto"},
        "segments": [{}, {}],
        "stats": {"audio_seconds": 60.0, "elapsed_seconds": 40.0, "chunks": 5, "silence_skipped_seconds": 2.0},
    }
    completed = events.completed_event(JOB, document)
    valid(completed, "likho.transcription.completed.v1.schema.json")
    assert completed["data"]["stats"]["segments"] == 2

    # the same event built again has the same id, so a retried publish is stored once
    assert events.completed_event(JOB, document)["id"] == completed["id"]
    again = events.segment_event(JOB, Segment(3, 1.5, 4.0, "ठीक है", "theek hai"), 60.0, attempt=1)
    assert again["id"] == segment["id"]
    assert events.segment_event(JOB, Segment(3, 1.5, 4.0, "x", "x"), 60.0, attempt=2)["id"] != segment["id"]


def test_job_error_only_accepts_contract_codes() -> None:
    with pytest.raises(AssertionError):
        JobError("made_up", "x", retryable=False)
