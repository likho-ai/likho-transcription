"""GET /healthz (the process is alive) and GET /readyz (its dependencies answer), for the gateway and Kubernetes."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

ReadyCheck = Callable[[], Awaitable[bool]]


def _response(status: str, body: str) -> bytes:
    payload = body.encode()
    head = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    )
    return head.encode() + payload


async def start_health_server(port: int, ready: ReadyCheck) -> asyncio.Server:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = (await asyncio.wait_for(reader.readline(), timeout=5)).decode("latin-1")
            path = request_line.split(" ")[1] if request_line.count(" ") >= 2 else ""
            if path == "/healthz":
                writer.write(_response("200 OK", "ok\n"))
            elif path == "/readyz":
                try:
                    is_ready = await asyncio.wait_for(ready(), timeout=3)
                except Exception:
                    is_ready = False
                writer.write(
                    _response("200 OK", "ready\n") if is_ready else _response("503 Service Unavailable", "not ready\n")
                )
            else:
                writer.write(_response("404 Not Found", "not found\n"))
            await writer.drain()
        except (TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, host="0.0.0.0", port=port)
