from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .approvals import ApprovalPolicy, ApprovalRequest, StaticApprovalPolicy
from .types import ToolExecutionResult
from .workspace import Workspace, WorkspaceError


MAX_OUTPUT_CHARS = 200_000
SAFE_SHELL_COMMANDS = {"ls", "pwd", "find", "rg", "grep", "cat", "sed", "head", "tail", "wc", "du", "tree", "file", "stat"}
DANGEROUS_SHELL_COMMANDS = {
    "rm",
    "mv",
    "cp",
    "chmod",
    "chown",
    "git",
    "pip",
    "pip3",
    "npm",
    "pnpm",
    "yarn",
    "curl",
    "wget",
    "ssh",
    "scp",
    "python",
    "python3",
    "bash",
    "sh",
    "zsh",
    "apt",
    "apt-get",
    "brew",
}
SHELL_CONTROL_MARKERS = (";", "&&", "||", "|", ">", "<", "$(", "`")


HARNESS_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Run a non-interactive shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 text file in the workspace after approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff patch in the workspace after approval.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the workspace, optionally filtered by glob or substring query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    },
]


@dataclass
class ApprovalNeed:
    required: bool
    reason: str = ""


class HarnessToolExecutor:
    def __init__(
        self,
        workspace: Workspace,
        approval_policy: ApprovalPolicy | None = None,
        *,
        default_timeout: int = 60,
    ):
        self.workspace = workspace
        self.approval_policy = approval_policy or StaticApprovalPolicy("deny")
        self.default_timeout = default_timeout

    def requires_approval(self, tool_name: str, arguments: dict[str, Any]) -> ApprovalNeed:
        if tool_name in {"write_file", "apply_patch"}:
            return ApprovalNeed(True, f"{tool_name} changes files in the workspace")
        if tool_name == "shell_exec":
            command = str(arguments.get("command") or "")
            if not _is_readonly_shell_command(command):
                return ApprovalNeed(True, "command is not in the safe read-only allowlist")
        return ApprovalNeed(False)

    async def run(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        arguments = arguments or {}
        try:
            approval_need = self.requires_approval(tool_name, arguments)
            if approval_need.required:
                decision = await self.approval_policy.approve(
                    ApprovalRequest(tool_name, dict(arguments), approval_need.reason)
                )
                if not decision.approved:
                    return ToolExecutionResult(
                        ok=False,
                        error=decision.reason or "approval denied",
                        approval_required=True,
                    )

            if tool_name == "read_file":
                return await self._read_file(arguments)
            if tool_name == "write_file":
                return await self._write_file(arguments)
            if tool_name == "apply_patch":
                return await self._apply_patch(arguments)
            if tool_name == "list_files":
                return await self._list_files(arguments)
            if tool_name == "shell_exec":
                return await self._shell_exec(arguments)
            return ToolExecutionResult(ok=False, error=f"unknown harness tool: {tool_name}")
        except WorkspaceError as exc:
            return ToolExecutionResult(ok=False, error=str(exc))
        except Exception as exc:
            return ToolExecutionResult(ok=False, error=str(exc))

    async def _read_file(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        path = self.workspace.resolve(arguments.get("path", ""))
        data = path.read_bytes()
        if len(data) > MAX_OUTPUT_CHARS:
            return ToolExecutionResult(ok=False, error=f"file exceeds {MAX_OUTPUT_CHARS} bytes")
        return ToolExecutionResult(ok=True, output=data.decode("utf-8"))

    async def _write_file(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        path = self.workspace.resolve(arguments.get("path", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments.get("content", "")), encoding="utf-8")
        return ToolExecutionResult(ok=True, output=f"wrote {self.workspace.relative(path)}")

    async def _apply_patch(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        patch = str(arguments.get("patch") or "")
        if not patch.strip():
            return ToolExecutionResult(ok=False, error="patch is required")
        strip_level = self._validate_patch_paths(patch)
        proc = await asyncio.to_thread(
            subprocess.run,
            ["patch", f"-p{strip_level}", "--batch", "--forward"],
            input=patch,
            text=True,
            cwd=str(self.workspace.root),
            capture_output=True,
            timeout=self.default_timeout,
        )
        output = _truncate((proc.stdout or "") + (proc.stderr or ""))
        return ToolExecutionResult(ok=proc.returncode == 0, output=output, error="" if proc.returncode == 0 else output, exit_code=proc.returncode)

    async def _list_files(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        query = str(arguments.get("query") or "").strip()
        files = self._collect_files(query)
        return ToolExecutionResult(ok=True, output="\n".join(files))

    async def _shell_exec(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        command = str(arguments.get("command") or "")
        timeout = int(arguments.get("timeout") or self.default_timeout)
        try:
            proc = self._run_command_sync(command, timeout)
        except subprocess.TimeoutExpired:
            return ToolExecutionResult(
                ok=False,
                error=f"command timed out after {timeout}s",
                exit_code=-1,
                timed_out=True,
            )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        return ToolExecutionResult(
            ok=proc.returncode == 0,
            output=_truncate(stdout),
            error="" if proc.returncode == 0 else _truncate(stderr),
            exit_code=proc.returncode,
        )

    def _run_command_sync(self, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
        if not any(marker in command for marker in SHELL_CONTROL_MARKERS):
            try:
                argv = shlex.split(command)
            except ValueError:
                argv = []
            if argv:
                argv = _normalize_readonly_argv_for_platform(argv)
                return subprocess.run(
                    argv,
                    cwd=str(self.workspace.root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                )
        shell_kwargs = {}
        if os.name != "nt":
            shell_kwargs["executable"] = "/bin/sh"
        return subprocess.run(
            command,
            shell=True,
            cwd=str(self.workspace.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **shell_kwargs,
        )

    def _collect_files(self, query: str) -> list[str]:
        files: list[str] = []
        candidates = self._rg_candidates(query)
        if candidates is None:
            candidates = self._rglob_candidates()

        for candidate in candidates:
            if query and not any(ch in query for ch in "*?[]") and query not in candidate:
                continue
            if query and any(ch in query for ch in "*?[]") and not fnmatch.fnmatch(candidate, query):
                continue
            try:
                self.workspace.resolve(candidate)
            except WorkspaceError:
                continue
            files.append(candidate)
        return sorted(dict.fromkeys(files))

    def _rglob_candidates(self) -> list[str]:
        return [
            p.relative_to(self.workspace.root).as_posix()
            for p in self.workspace.root.rglob("*")
            if p.is_file()
        ]

    def _rg_candidates(self, query: str) -> list[str] | None:
        rg = shutil.which("rg")
        if not rg:
            return None
        cmd = [rg, "--files"]
        if query and any(ch in query for ch in "*?[]"):
            cmd.extend(["-g", query])
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workspace.root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(self.default_timeout, 2),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode not in {0, 1}:
            return None
        return proc.stdout.splitlines()

    def _validate_patch_paths(self, patch: str) -> int:
        paths: list[str] = []
        strip_level = 0
        for line in patch.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                raw = line[4:].split("\t", 1)[0].strip()
                if raw == "/dev/null":
                    continue
                if raw.startswith(("a/", "b/")):
                    strip_level = 1
                normalized = raw[2:] if raw.startswith(("a/", "b/")) else raw
                paths.append(normalized)
        if not paths:
            raise WorkspaceError("patch does not include file paths")
        for path in paths:
            self.workspace.resolve(path)
        return strip_level


def _is_readonly_shell_command(command: str) -> bool:
    if not command.strip():
        return False
    if any(marker in command for marker in SHELL_CONTROL_MARKERS):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    exe = Path(argv[0]).name
    if exe in DANGEROUS_SHELL_COMMANDS:
        return False
    if exe not in SAFE_SHELL_COMMANDS:
        return False
    if exe == "sed" and any(arg.startswith("-i") for arg in argv[1:]):
        return False
    if exe == "find" and "-delete" in argv[1:]:
        return False
    return True


def _normalize_readonly_argv_for_platform(argv: list[str]) -> list[str]:
    if os.name != "nt" or not argv:
        return argv
    exe = Path(argv[0]).name.lower()
    if exe == "ls":
        # Common POSIX listing requests from models should work on Windows.
        dir_args = ["cmd", "/c", "dir"]
        if any(arg.startswith("-") and "a" in arg for arg in argv[1:]):
            dir_args.append("/a")
        return dir_args
    if exe == "pwd":
        return ["cmd", "/c", "cd"]
    return argv


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"
