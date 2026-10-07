"""Which model transcribes when a job names none: the model registry's default (likho-ml).

The answer is kept for ml_default_ttl_seconds, and forgotten at once when likho-ml publishes
likho.model.chosen. Whatever goes wrong - likho-ml unreachable, no default yet, a default this
worker cannot load (a fine-tuned model without its weights here) - the worker's own
default_model is used and the reason is logged once, so the registry can never stop a job.
"""

import asyncio
import logging
import time
from collections.abc import Callable

import grpc
from likho.ml.v1 import ml_pb2, ml_pb2_grpc

log = logging.getLogger(__name__)


class RegistryDefault:
    def __init__(
        self,
        address: str,
        fallback: str,
        loadable: Callable[[str], bool],
        ttl_seconds: float = 60.0,
        timeout_seconds: float = 2.0,
    ) -> None:
        """`fallback` and the answer are registry ids ("faster-whisper/turbo"); `loadable` says
        whether this worker can load one."""
        self._fallback = fallback
        self._loadable = loadable
        self._ttl = ttl_seconds
        self._timeout = timeout_seconds
        self._channel = grpc.aio.insecure_channel(address) if address else None
        self._stub = ml_pb2_grpc.MlServiceStub(self._channel) if self._channel else None
        self._value = fallback
        self._until = 0.0
        self._lock = asyncio.Lock()
        self._last_reason = ""

    def forget(self) -> None:
        """The default changed (likho.model.chosen): ask again on the next job."""
        self._until = 0.0

    async def registry_id(self) -> str:
        if self._stub is None:
            return self._fallback
        if time.monotonic() < self._until:
            return self._value
        async with self._lock:
            if time.monotonic() < self._until:
                return self._value
            self._value = await self._ask()
            # A failed answer is kept for the same time, so a down likho-ml costs one timeout per period.
            self._until = time.monotonic() + self._ttl
            return self._value

    async def _ask(self) -> str:
        assert self._stub is not None
        try:
            reply = await self._stub.GetDefault(ml_pb2.GetDefaultRequest(), timeout=self._timeout)
        except grpc.aio.AioRpcError as error:
            return self._fall_back(f"likho-ml did not answer ({error.code().name})")
        chosen = reply.model.registry_id
        if not chosen:
            return self._fall_back("the registry has no default yet")
        if not self._loadable(chosen):
            return self._fall_back(f"the registry's default {chosen} cannot be loaded by this worker")
        if chosen != self._value or self._last_reason:
            log.info("default model from the registry: %s", chosen)
        self._last_reason = ""
        return chosen

    def _fall_back(self, reason: str) -> str:
        if reason != self._last_reason:
            log.warning("%s; transcribing with %s", reason, self._fallback)
            self._last_reason = reason
        return self._fallback

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
