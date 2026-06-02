from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ApprovalRequest:
    tool_name: str
    arguments: dict[str, Any]
    reason: str


@dataclass
class ApprovalDecision:
    approved: bool
    reason: str = ""


class ApprovalPolicy(Protocol):
    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        ...


class StaticApprovalPolicy:
    def __init__(self, decision: str = "deny"):
        normalized = decision.lower()
        self.approved = normalized in {"allow", "always", "approve", "approved", "yes"}
        self.requests: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        if self.approved:
            return ApprovalDecision(True, "approved by policy")
        return ApprovalDecision(False, "approval denied by policy")


class CLIApprovalPolicy:
    def __init__(self, mode: str = "ask", *, stdin=None, stderr=None):
        self.mode = mode
        self.stdin = stdin or sys.stdin
        self.stderr = stderr or sys.stderr
        self.requests: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        if self.mode == "always":
            return ApprovalDecision(True, "approved by CLI policy")
        if self.mode == "never":
            return ApprovalDecision(False, "approval denied by CLI policy")

        prompt = (
            f"\nApproval required for {request.tool_name}: {request.reason}\n"
            f"Arguments: {json.dumps(request.arguments, ensure_ascii=False)}\n"
            "Allow? [y/N] "
        )
        self.stderr.write(prompt)
        self.stderr.flush()
        answer = await asyncio.to_thread(self.stdin.readline)
        if answer.strip().lower() in {"y", "yes"}:
            return ApprovalDecision(True, "approved by user")
        return ApprovalDecision(False, "approval denied by user")


def policy_from_name(name: str) -> StaticApprovalPolicy:
    if name == "always":
        return StaticApprovalPolicy("allow")
    return StaticApprovalPolicy("deny")
