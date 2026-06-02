from __future__ import annotations

import json
from typing import Any, AsyncIterator

from .approvals import ApprovalPolicy, policy_from_name
from .model_adapter import ModelAdapter
from .tools import HARNESS_TOOL_SCHEMAS, HarnessToolExecutor
from .transcript import TranscriptWriter
from .types import HarnessEvent, HarnessRequest, ToolExecutionResult
from .vision import image_path_to_content_block
from .workspace import Workspace


SYSTEM_PROMPT = """You are Odysseus Harness, a minimal coding agent.
Use tools only when needed. All tool paths are relative to the selected workspace.
Return a final answer in normal assistant text when you are done."""


class HarnessRunner:
    def __init__(
        self,
        request: HarnessRequest,
        *,
        adapter: Any | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ):
        self.request = request
        self.workspace = Workspace.from_path(request.workspace)
        self.adapter = adapter or ModelAdapter(
            request.endpoint_url,
            request.model,
            headers=request.headers,
            timeout=request.timeout,
        )
        self.approval_policy = approval_policy or policy_from_name(request.approval_policy)
        self.tools = HarnessToolExecutor(
            self.workspace,
            self.approval_policy,
            default_timeout=request.timeout,
        )
        self.transcript = TranscriptWriter(self.workspace.root)

    async def run(self) -> AsyncIterator[HarnessEvent]:
        messages = self._initial_messages()
        total_tool_calls = 0
        final_text_parts: list[str] = []

        self.transcript.write("request", self.request.model_dump())
        for round_index in range(self.request.max_rounds):
            round_text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            async for event in self.adapter.stream(messages, HARNESS_TOOL_SCHEMAS):
                self.transcript.write(event.event, event.data)
                if event.event == "message_delta":
                    text = str(event.data.get("text") or "")
                    round_text_parts.append(text)
                    final_text_parts.append(text)
                    yield event
                elif event.event == "tool_call":
                    tool_call = event.data.get("tool_call") or {}
                    tool_calls.append(tool_call)
                    yield event
                elif event.event in {"usage", "message_metadata"}:
                    if event.event == "usage":
                        yield event
                elif event.event == "error":
                    yield event
                    return

            if not tool_calls:
                final = HarnessEvent("final", {"text": "".join(final_text_parts)})
                self.transcript.write(final.event, final.data)
                yield final
                usage = HarnessEvent(
                    "usage",
                    {
                        "rounds": round_index + 1,
                        "tool_calls": total_tool_calls,
                        "transcript": str(self.transcript.path),
                    },
                )
                self.transcript.write(usage.event, usage.data)
                yield usage
                return

            total_tool_calls += len(tool_calls)
            if total_tool_calls > self.request.max_tool_calls:
                error = HarnessEvent("error", {"error": "max tool call budget exceeded"})
                self.transcript.write(error.event, error.data)
                yield error
                return

            messages.append(self._assistant_tool_call_message(round_text_parts, tool_calls))
            for tool_call in tool_calls:
                approval_need = self.tools.requires_approval(tool_call.get("name", ""), tool_call.get("arguments") or {})
                if approval_need.required:
                    approval_event = HarnessEvent(
                        "approval_required",
                        {
                            "tool_call": tool_call,
                            "reason": approval_need.reason,
                        },
                    )
                    self.transcript.write(approval_event.event, approval_event.data)
                    yield approval_event

                result = await self.tools.run(tool_call.get("name", ""), tool_call.get("arguments") or {})
                result_event = HarnessEvent(
                    "tool_result",
                    {
                        "tool_call_id": tool_call.get("id"),
                        "tool_name": tool_call.get("name"),
                        "result": result.to_dict(),
                    },
                )
                self.transcript.write(result_event.event, result_event.data)
                yield result_event
                messages.append(self._tool_result_message(tool_call, result))

        error = HarnessEvent("error", {"error": "max rounds exceeded"})
        self.transcript.write(error.event, error.data)
        yield error

    def _initial_messages(self) -> list[dict[str, Any]]:
        user_content: str | list[dict[str, Any]]
        if self.request.images:
            blocks: list[dict[str, Any]] = [{"type": "text", "text": self.request.prompt}]
            for image in self.request.images:
                image_path = self.workspace.resolve(image)
                blocks.append(image_path_to_content_block(image_path))
            user_content = blocks
        else:
            user_content = self.request.prompt

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _assistant_tool_call_message(self, text_parts: list[str], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        content = "".join(text_parts).strip() or None
        shaped_calls = []
        for index, tool_call in enumerate(tool_calls):
            shaped = {
                "id": tool_call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": tool_call.get("name") or "",
                    "arguments": json.dumps(tool_call.get("arguments") or {}),
                },
            }
            if tool_call.get("extra_content"):
                shaped["extra_content"] = tool_call["extra_content"]
            shaped_calls.append(shaped)
        return {"role": "assistant", "content": content, "tool_calls": shaped_calls}

    def _tool_result_message(self, tool_call: dict[str, Any], result: ToolExecutionResult) -> dict[str, Any]:
        content = result.output if result.ok else f"ERROR: {result.error}"
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id") or "",
            "content": content,
        }
