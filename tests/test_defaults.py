"""The default model from the registry (likho-ml), with a stand-in MlService."""

from typing import Any

import grpc
import pytest
from likho.ml.v1 import ml_pb2, ml_pb2_grpc

from likho_transcription.defaults import RegistryDefault
from likho_transcription.runner import EngineProvider
from likho_transcription.settings import Settings


class FakeRegistry(ml_pb2_grpc.MlServiceServicer):
    def __init__(self) -> None:
        self.default = "faster-whisper/large-v3"
        self.asked = 0

    async def GetDefault(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        self.asked += 1
        return ml_pb2.GetDefaultResponse(model=ml_pb2.Model(registry_id=self.default))


@pytest.fixture
async def registry() -> Any:
    fake = FakeRegistry()
    server = grpc.aio.server()
    ml_pb2_grpc.add_MlServiceServicer_to_server(fake, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    yield fake, f"127.0.0.1:{port}"
    await server.stop(grace=0)


def provider(address: str, ttl: float = 60.0) -> EngineProvider:
    settings = Settings(default_model="turbo")
    holder: dict[str, EngineProvider] = {}
    defaults = RegistryDefault(
        address, fallback="faster-whisper/turbo", loadable=lambda r: holder["p"].can_load(r), ttl_seconds=ttl
    )
    holder["p"] = EngineProvider(settings, lambda size: size, defaults)  # type: ignore[arg-type, return-value]
    return holder["p"]


async def test_the_registry_names_the_default_and_a_job_without_a_model_gets_it(registry: Any) -> None:
    _, address = registry
    engines = provider(address)
    assert await engines.current_default() == "faster-whisper/large-v3"
    _, used = await engines.get("")
    assert used == "faster-whisper/large-v3"
    # A job that names its model is not changed.
    _, named = await engines.get("faster-whisper/small")
    assert named == "faster-whisper/small"


async def test_the_answer_is_kept_until_the_default_changes(registry: Any) -> None:
    fake, address = registry
    engines = provider(address)
    await engines.current_default()
    await engines.current_default()
    assert fake.asked == 1, "kept for the TTL"
    fake.default = "faster-whisper/medium"
    assert await engines.current_default() == "faster-whisper/large-v3", "still the kept answer"
    engines._defaults.forget()  # type: ignore[union-attr]  # what likho.model.chosen does
    assert await engines.current_default() == "faster-whisper/medium"


async def test_a_default_this_worker_cannot_load_falls_back(registry: Any) -> None:
    fake, address = registry
    fake.default = "faster-whisper/likho-2026-10"  # fine-tuned: no weights on this worker yet
    assert await provider(address).current_default() == "faster-whisper/turbo"


async def test_a_registry_that_does_not_answer_falls_back() -> None:
    engines = provider("127.0.0.1:1")  # nothing listens there
    assert await engines.current_default() == "faster-whisper/turbo"
    _, used = await engines.get("")
    assert used == "faster-whisper/turbo"


async def test_without_a_registry_address_the_settings_decide() -> None:
    engines = EngineProvider(Settings(default_model="medium"), lambda size: size)  # type: ignore[arg-type, return-value]
    assert await engines.current_default() == "faster-whisper/medium"
