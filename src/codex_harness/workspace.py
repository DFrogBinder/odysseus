from __future__ import annotations

import os
from pathlib import Path


class WorkspaceError(ValueError):
    """Raised when a requested path is not safe for the harness workspace."""


SENSITIVE_PARTS = {
    ".ssh",
    ".gnupg",
    ".env",
    ".env.local",
    ".envrc",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".netrc",
}

SENSITIVE_NAMES = {
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "authorized_keys",
    "known_hosts",
}

SENSITIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}


class Workspace:
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "Workspace":
        root = Path(path).expanduser()
        try:
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise WorkspaceError(f"workspace does not exist: {path}") from exc
        if not resolved.is_dir():
            raise WorkspaceError(f"workspace is not a directory: {path}")
        return cls(resolved)

    def resolve(self, candidate: str | os.PathLike[str]) -> Path:
        if candidate is None:
            raise WorkspaceError("path is required")
        raw_s = str(candidate)
        if not raw_s.strip():
            raise WorkspaceError("path is required")
        if "\x00" in raw_s:
            raise WorkspaceError("path contains a NUL byte")

        raw = Path(raw_s).expanduser()
        self._reject_sensitive_parts(raw)
        combined = raw if raw.is_absolute() else self.root / raw

        try:
            resolved = combined.resolve(strict=False)
        except OSError as exc:
            raise WorkspaceError(f"cannot resolve path: {candidate}") from exc

        self._ensure_inside_root(resolved)
        self._reject_symlink_components(combined)
        self._reject_sensitive_parts(resolved)
        return resolved

    def relative(self, path: str | os.PathLike[str]) -> str:
        resolved = self.resolve(path)
        return resolved.relative_to(self.root).as_posix()

    def _ensure_inside_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace: {path}") from exc

    def _reject_symlink_components(self, path: Path) -> None:
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            rel = path
        current = self.root
        for part in rel.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise WorkspaceError(f"symlinks are not allowed in harness paths: {current}")

    def _reject_sensitive_parts(self, path: Path) -> None:
        for part in path.parts:
            lower = part.lower()
            if lower in SENSITIVE_PARTS or lower in SENSITIVE_NAMES:
                raise WorkspaceError(f"sensitive path is blocked: {part}")
            if lower.endswith(tuple(SENSITIVE_SUFFIXES)):
                raise WorkspaceError(f"sensitive file type is blocked: {part}")
