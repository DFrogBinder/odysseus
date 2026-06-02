import asyncio
import json

from src.codex_harness.model_adapter import ModelAdapter


def _data(payload):
    return "data: " + json.dumps(payload) + "\n\n"


def test_adapter_normalizes_streamed_text_and_native_tool_calls(monkeypatch):
    async def fake_stream_llm(*args, **kwargs):
        yield _data({"delta": "Let me check."})
        yield _data(
            {
                "type": "tool_calls",
                "calls": [
                    {
                        "id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path": "notes.txt"}',
                    }
                ],
            }
        )
        yield "data: [DONE]\n\n"

    monkeypatch.setattr("src.codex_harness.model_adapter.stream_llm", fake_stream_llm)
    adapter = ModelAdapter("http://example/v1/chat/completions", "model-a", headers={"Authorization": "Bearer k"})

    events = asyncio.run(_collect(adapter.stream([{"role": "user", "content": "hi"}], tools=[])))

    assert [event.event for event in events] == ["message_delta", "tool_call"]
    assert events[0].data["text"] == "Let me check."
    assert events[1].data["tool_call"]["name"] == "read_file"
    assert events[1].data["tool_call"]["arguments"] == {"path": "notes.txt"}


def test_adapter_extracts_text_fallback_tool_calls(monkeypatch):
    async def fake_stream_llm(*args, **kwargs):
        yield _data(
            {
                "delta": '[TOOL_CALL] {tool: "read_file", path: "notes.txt"} [/TOOL_CALL]',
            }
        )
        yield "data: [DONE]\n\n"

    monkeypatch.setattr("src.codex_harness.model_adapter.stream_llm", fake_stream_llm)
    adapter = ModelAdapter("http://example/v1/chat/completions", "model-a")

    events = asyncio.run(_collect(adapter.stream([{"role": "user", "content": "hi"}], tools=[])))

    assert [event.event for event in events] == ["tool_call"]
    assert events[0].data["tool_call"]["name"] == "read_file"
    assert events[0].data["tool_call"]["arguments"] == {"path": "notes.txt"}


async def _collect(aiter):
    return [event async for event in aiter]
