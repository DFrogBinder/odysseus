from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.codex_harness.runner import HarnessRunner
from src.codex_harness.types import HarnessEvent, HarnessRequest
from src.codex_harness.workspace import Workspace, WorkspaceError


def setup_harness_routes(runner_factory=None) -> APIRouter:
    router = APIRouter()
    make_runner = runner_factory or HarnessRunner

    @router.post("/api/harness/run")
    async def run_harness(request: HarnessRequest) -> StreamingResponse:
        try:
            Workspace.from_path(request.workspace)
        except WorkspaceError as exc:
            raise HTTPException(400, str(exc)) from exc

        async def _stream():
            try:
                runner = make_runner(request)
                async for event in runner.run():
                    yield event.to_sse()
            except Exception as exc:
                yield HarnessEvent("error", {"error": str(exc)}).to_sse()

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return router
