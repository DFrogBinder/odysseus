from pathlib import Path

import pytest

from src.codex_harness.workspace import Workspace, WorkspaceError


def test_workspace_resolves_relative_paths_inside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.txt").write_text("hello", encoding="utf-8")

    ws = Workspace.from_path(root)

    assert ws.resolve("notes.txt") == root / "notes.txt"
    assert ws.relative(root / "notes.txt") == "notes.txt"


def test_workspace_rejects_traversal_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    ws = Workspace.from_path(root)

    with pytest.raises(WorkspaceError):
        ws.resolve("../outside.txt")


def test_workspace_rejects_symlinks_that_escape_root(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside / "secret.txt")

    ws = Workspace.from_path(root)

    with pytest.raises(WorkspaceError):
        ws.resolve("link.txt")


@pytest.mark.parametrize(
    "candidate",
    [
        ".env",
        ".ssh/id_rsa",
        ".gnupg/private.key",
        "nested/private.key",
        ".bashrc",
    ],
)
def test_workspace_blocks_sensitive_names_even_inside_root(tmp_path, candidate):
    root = tmp_path / "project"
    root.mkdir()
    ws = Workspace.from_path(root)

    with pytest.raises(WorkspaceError):
        ws.resolve(candidate)
