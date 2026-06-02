from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


ApprovalPolicyName = Literal["ask", "always", "never"]


class HarnessRequest(BaseModel):
    prompt: str
    workspace: str
    endpoint_url: str
    model: str
    headers: dict[str, str] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)
    approval_policy: ApprovalPolicyName = "ask"
    max_rounds: int = 12
    max_tool_calls: int = 30
    timeout: int = 60


@dataclass
class HarnessEvent:
    event: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        import json

        return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


@dataclass
class ToolExecutionResult:
    ok: bool
    output: str = ""
    error: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    approval_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
        }
        if self.exit_code is not None:
            data["exit_code"] = self.exit_code
        if self.timed_out:
            data["timed_out"] = True
        if self.approval_required:
            data["approval_required"] = True
        return data
