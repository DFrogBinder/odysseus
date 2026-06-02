from __future__ import annotations

import json
from typing import Any, AsyncIterator

from src.llm_core import stream_llm
from src.agent_tools import parse_tool_blocks, strip_tool_blocks

from .types import HarnessEvent


TOOL_MARKERS = ("[TOOL_CALL]", "<invoke", "<tool_call", "<tool_code>", "```bash", "```shell")


class ModelAdapter:
    def __init__(
        self,
        endpoint_url: str,
        model: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: int = 60,
    ):
        self.endpoint_url = endpoint_url
        self.model = model
        self.headers = headers or {}
        self.timeout = timeout

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[HarnessEvent]:
        fallback_buffer = ""
        buffering_for_tool_text = False
        native_tool_seen = False

        async for chunk in stream_llm(
            self.endpoint_url,
            self.model,
            messages,
            headers=self.headers,
            timeout=self.timeout,
            tools=tools,
        ):
            for event_name, payload in _parse_sse_chunk(chunk):
                if event_name == "error":
                    yield HarnessEvent("error", payload if isinstance(payload, dict) else {"error": str(payload)})
                    continue
                if payload == "[DONE]":
                    continue
                if not isinstance(payload, dict):
                    continue

                if payload.get("type") == "tool_calls":
                    native_tool_seen = True
                    for call in payload.get("calls") or []:
                        yield HarnessEvent("tool_call", {"tool_call": _normalize_native_tool_call(call)})
                    continue
                if payload.get("type") == "usage":
                    yield HarnessEvent("usage", payload.get("data") or {})
                    continue
                if payload.get("type") == "tool_call_delta":
                    continue

                text = payload.get("delta")
                if not text:
                    continue
                if payload.get("thinking"):
                    yield HarnessEvent("message_metadata", {"thinking": True})
                    continue
                if buffering_for_tool_text or any(marker in text for marker in TOOL_MARKERS):
                    buffering_for_tool_text = True
                    fallback_buffer += text
                else:
                    yield HarnessEvent("message_delta", {"text": text})

        if not native_tool_seen and fallback_buffer:
            blocks = parse_tool_blocks(fallback_buffer)
            if blocks:
                visible = strip_tool_blocks(fallback_buffer)
                if visible:
                    yield HarnessEvent("message_delta", {"text": visible})
                for block in blocks:
                    call = _tool_block_to_call(block)
                    if call:
                        yield HarnessEvent("tool_call", {"tool_call": call})
            else:
                yield HarnessEvent("message_delta", {"text": fallback_buffer})


def _parse_sse_chunk(chunk: str):
    event_name = "message"
    data_lines: list[str] = []
    for line in chunk.splitlines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if not data_lines:
        return
    raw = "\n".join(data_lines)
    if raw == "[DONE]":
        yield event_name, "[DONE]"
        return
    try:
        yield event_name, json.loads(raw)
    except json.JSONDecodeError:
        yield event_name, raw


def _normalize_native_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    name = call.get("name") or call.get("function", {}).get("name") or ""
    raw_args = call.get("arguments")
    if raw_args is None and isinstance(call.get("function"), dict):
        raw_args = call["function"].get("arguments")
    args = _coerce_arguments(raw_args)
    normalized = {
        "id": call.get("id") or "call_0",
        "name": _normalize_tool_name(name),
        "arguments": _normalize_tool_arguments(name, args),
    }
    if call.get("extra_content"):
        normalized["extra_content"] = call["extra_content"]
    return normalized


def _tool_block_to_call(block) -> dict[str, Any] | None:
    name = _normalize_tool_name(block.tool_type)
    if not name:
        return None
    return {
        "id": f"text_call_{abs(hash((name, block.content))) % 1_000_000}",
        "name": name,
        "arguments": _normalize_tool_arguments(name, {"content": block.content}),
    }


def _coerce_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"content": raw_args}
    if raw_args is None:
        return {}
    return {"value": raw_args}


def _normalize_tool_name(name: str) -> str:
    mapping = {
        "bash": "shell_exec",
        "shell": "shell_exec",
        "terminal": "shell_exec",
        "read": "read_file",
    }
    return mapping.get((name or "").lower(), name or "")


def _normalize_tool_arguments(name: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized_name = _normalize_tool_name(name)
    if normalized_name == "shell_exec":
        if "command" in args:
            return args
        return {"command": args.get("content") or args.get("value") or ""}
    if normalized_name == "read_file":
        if "path" in args:
            return args
        return {"path": args.get("content") or args.get("file") or args.get("value") or ""}
    if normalized_name == "write_file":
        if "path" in args and "content" in args:
            return args
        return {"path": args.get("path") or args.get("file") or "", "content": args.get("content") or ""}
    if normalized_name == "apply_patch":
        if "patch" in args:
            return args
        return {"patch": args.get("content") or args.get("value") or ""}
    if normalized_name == "list_files":
        return {"query": args.get("query") or args.get("content") or ""}
    return args
