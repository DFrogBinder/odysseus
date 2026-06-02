import json

import pytest
from fastapi import HTTPException

from routes.harness_routes import setup_harness_routes
from src.codex_harness.types import HarnessEvent, HarnessRequest


class FakeRunner:
    def __init__(self, request):
        self.request = request

    async def run(self):
        yield HarnessEvent("message_delta", {"text": "hello"})
        yield HarnessEvent("final", {"text": "hello"})


def test_harness_route_emits_sse_events(tmp_path):
    endpoint = _endpoint()
    response = _run(
        endpoint(
            HarnessRequest(
                prompt="hi",
                workspace=str(tmp_path),
                endpoint_url="http://example/v1/chat/completions",
                model="fake-model",
            )
        )
    )
    text = _run(_body(response))

    assert "event: message_delta" in text
    assert "event: final" in text
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    assert {"text": "hello"} in payloads


def test_harness_route_rejects_bad_workspace(tmp_path):
    endpoint = _endpoint()

    with pytest.raises(HTTPException) as excinfo:
        _run(
            endpoint(
                HarnessRequest(
                    prompt="hi",
                    workspace=str(tmp_path / "missing"),
                    endpoint_url="http://example/v1/chat/completions",
                    model="fake-model",
                )
            )
        )

    assert excinfo.value.status_code == 400
    assert "workspace" in excinfo.value.detail.lower()


def _endpoint():
    router = setup_harness_routes(runner_factory=FakeRunner)
    return next(route.endpoint for route in router.routes if getattr(route, "path", "") == "/api/harness/run")


async def _body(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def _run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
