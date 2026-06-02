import asyncio

from src.codex_harness.approvals import StaticApprovalPolicy
from src.codex_harness.tools import HarnessToolExecutor
from src.codex_harness.workspace import Workspace


def test_read_and_list_files_do_not_require_approval(tmp_path):
    async def run():
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.py").write_text("print('b')", encoding="utf-8")
        executor = HarnessToolExecutor(Workspace.from_path(tmp_path), StaticApprovalPolicy("deny"))

        read_result = await executor.run("read_file", {"path": "a.txt"})
        list_result = await executor.run("list_files", {"query": "*.txt"})

        assert read_result.ok is True
        assert read_result.output == "alpha"
        assert list_result.ok is True
        assert "a.txt" in list_result.output

    asyncio.run(run())


def test_write_file_requires_approval_and_denial_returns_tool_error(tmp_path):
    async def run():
        policy = StaticApprovalPolicy("deny")
        executor = HarnessToolExecutor(Workspace.from_path(tmp_path), policy)

        result = await executor.run("write_file", {"path": "created.txt", "content": "new"})

        assert result.ok is False
        assert "approval denied" in result.error.lower()
        assert not (tmp_path / "created.txt").exists()
        assert policy.requests[0].tool_name == "write_file"

    asyncio.run(run())


def test_apply_patch_requires_approval(tmp_path):
    async def run():
        (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
        patch = """--- a.txt
+++ a.txt
@@ -1 +1 @@
-old
+new
"""
        executor = HarnessToolExecutor(Workspace.from_path(tmp_path), StaticApprovalPolicy("deny"))

        result = await executor.run("apply_patch", {"patch": patch})

        assert result.ok is False
        assert "approval denied" in result.error.lower()
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "old\n"

    asyncio.run(run())


def test_safe_shell_exec_runs_without_approval(tmp_path):
    async def run():
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        executor = HarnessToolExecutor(Workspace.from_path(tmp_path), StaticApprovalPolicy("deny"))

        result = await executor.run("shell_exec", {"command": "ls", "timeout": 5})

        assert result.ok is True
        assert "a.txt" in result.output

    asyncio.run(run())


def test_destructive_shell_exec_requires_approval(tmp_path):
    async def run():
        target = tmp_path / "a.txt"
        target.write_text("alpha", encoding="utf-8")
        policy = StaticApprovalPolicy("deny")
        executor = HarnessToolExecutor(Workspace.from_path(tmp_path), policy)

        result = await executor.run("shell_exec", {"command": "rm -f a.txt", "timeout": 5})

        assert result.ok is False
        assert "approval denied" in result.error.lower()
        assert target.exists()
        assert policy.requests[0].tool_name == "shell_exec"

    asyncio.run(run())
