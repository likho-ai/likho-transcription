"""Clients of the other services: likho-language (how text is written) and likho-media (where the audio is)."""

import logging
from pathlib import Path

import grpc
import httpx
from likho.common.v1 import common_pb2
from likho.language.v1 import language_pb2, language_pb2_grpc
from likho.media.v1 import media_pb2, media_pb2_grpc
from likho_hinglish import Transliterator

from likho_engine import Decision, Detection
from likho_engine.transcript import default_decision
from likho_transcription.errors import JobError

log = logging.getLogger(__name__)

_DOWN = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)


class LanguageGateway:
    """Asks likho-language; when it cannot be reached, falls back to the built-in rules.

    The fallback keeps transcription running: the lines are still written in Hinglish with
    the built-in table, only the workspace's own names and spellings are missing. Such a
    transcript carries vocabulary_version 0 and can be re-transliterated later.
    """

    def __init__(self, address: str, timeout: float) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = language_pb2_grpc.LanguageServiceStub(self._channel)
        self._timeout = timeout
        self._address = address
        self._fallback = Transliterator()
        self._reported_down = False

    async def close(self) -> None:
        await self._channel.close()

    def _down(self, error: grpc.aio.AioRpcError) -> bool:
        if error.code() not in _DOWN:
            return False
        if not self._reported_down:
            log.warning("likho-language at %s is not reachable; using the built-in rules", self._address)
            self._reported_down = True
        return True

    def _up(self) -> None:
        if self._reported_down:
            log.info("likho-language at %s is reachable again", self._address)
            self._reported_down = False

    async def decide(self, workspace_id: str, detection: Detection) -> Decision:
        request = language_pb2.ResolveDecodePolicyRequest(
            workspace_id=workspace_id,
            detected=detection.language,
            probability=detection.probability,
            candidates=[common_pb2.LanguageCandidate(language=c, probability=p) for c, p in detection.candidates],
        )
        try:
            reply = await self._stub.ResolveDecodePolicy(request, timeout=self._timeout)
        except grpc.aio.AioRpcError as error:
            if self._down(error):
                return default_decision(detection)
            raise
        self._up()
        return Decision(reply.decode_as, reply.transliterate)

    async def hotwords(self, workspace_id: str, language: str) -> list[str]:
        request = language_pb2.GetHotwordsRequest(workspace_id=workspace_id, language=language)
        try:
            reply = await self._stub.GetHotwords(request, timeout=self._timeout)
        except grpc.aio.AioRpcError as error:
            if self._down(error):
                return []
            raise
        self._up()
        return list(reply.terms)

    async def to_roman(self, workspace_id: str, texts: list[str]) -> tuple[list[str], int]:
        """The Hinglish for each text, and the version of the vocabulary that was applied (0 = built-in rules only)."""
        if not texts:
            return [], 0
        request = language_pb2.TransliterateBatchRequest(
            workspace_id=workspace_id, texts=texts, source_script=common_pb2.SCRIPT_DEVANAGARI
        )
        try:
            reply = await self._stub.TransliterateBatch(request, timeout=self._timeout)
        except grpc.aio.AioRpcError as error:
            if self._down(error):
                return [self._fallback.text(text) for text in texts], 0
            raise
        self._up()
        return list(reply.texts_roman), int(reply.vocabulary_version)


class MediaGateway:
    """Finds a recording's audio through likho-media and downloads it."""

    def __init__(self, address: str, rpc_timeout: float, download_timeout: float) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = media_pb2_grpc.MediaServiceStub(self._channel)
        self._rpc_timeout = rpc_timeout
        self._download_timeout = download_timeout

    async def close(self) -> None:
        await self._channel.close()

    async def download(self, media_id: str, directory: Path) -> Path:
        """Save the audio in directory and return its path."""
        # The file exactly as it was uploaded. A converted copy is not the same audio to the model:
        # on real calls it changed about one word in seven.
        request = media_pb2.GetDownloadUrlRequest(id=media_id, kind=media_pb2.MEDIA_KIND_ORIGINAL)
        try:
            reply = await self._stub.GetDownloadUrl(request, timeout=self._rpc_timeout)
        except grpc.aio.AioRpcError as error:
            if error.code() == grpc.StatusCode.NOT_FOUND:
                raise JobError("audio_unreadable", "The recording was not found", retryable=False) from error
            raise JobError(
                "internal", f"The media service did not answer ({error.code().name})", retryable=True
            ) from error

        target = directory / "audio"
        try:
            async with (
                httpx.AsyncClient(timeout=self._download_timeout, follow_redirects=True) as client,
                client.stream("GET", reply.url) as response,
            ):
                if response.status_code in (403, 404):
                    raise JobError("audio_unreadable", "The recording could not be downloaded", retryable=False)
                response.raise_for_status()
                with target.open("wb") as file:
                    async for chunk in response.aiter_bytes():
                        file.write(chunk)
        except httpx.HTTPError as error:
            raise JobError("internal", f"Downloading the recording failed: {error}", retryable=True) from error
        return target
