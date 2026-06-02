import asyncio

from src.codex_harness.approvals import StaticApprovalPolicy
from src.codex_harness.runner import HarnessRunner
from src.codex_harness.types import HarnessEvent, HarnessRequest


class FakeAdapter:
    def __init__(self):
        self.messages_by_round = []

    async def stream(self, messages, tools):
        self.messages_by_round.append(messages)
        if len(self.messages_by_round) == 1:
            yield HarnessEvent(
                "tool_call",
                {
                    "tool_call": {
                        "id": "call_1",
                        "name": "read_file",
                        "arguments": {"path": "answer.txt"},
                    }
                },
            )
        else:
            yield HarnessEvent("message_delta", {"text": "The file says: 42."})


def test_runner_executes_tool_result_then_returns_final_answer(tmp_path):
    async def run():
        (tmp_path / "answer.txt").write_text("42", encoding="utf-8")
        adapter = FakeAdapter()
        request = HarnessRequest(
            prompt="Read answer.txt",
            workspace=str(tmp_path),
            endpoint_url="http://example/v1/chat/completions",
            model="fake-model",
            max_rounds=3,
        )
        runner = HarnessRunner(request, adapter=adapter, approval_policy=StaticApprovalPolicy("allow"))

        events = [event async for event in runner.run()]

        assert [event.event for event in events] == [
            "tool_call",
            "tool_result",
            "message_delta",
            "final",
            "usage",
        ]
        assert events[1].data["result"]["ok"] is True
        assert events[1].data["result"]["output"] == "42"
        assert events[3].data["text"] == "The file says: 42."
        assert any(m.get("role") == "tool" and m.get("content") == "42" for m in adapter.messages_by_round[1])
        assert (tmp_path / ".odysseus_harness" / "runs").is_dir()

    asyncio.run(run())
