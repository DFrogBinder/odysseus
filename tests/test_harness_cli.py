import importlib.machinery
import importlib.util
from pathlib import Path

from src.codex_harness.types import HarnessEvent


ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    path = ROOT / "scripts" / "odysseus-harness"
    loader = importlib.machinery.SourceFileLoader("odysseus_harness_cli", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class FakeRunner:
    def __init__(self, request, approval_policy=None):
        self.request = request
        self.approval_policy = approval_policy

    async def run(self):
        yield HarnessEvent("message_delta", {"text": "done"})
        yield HarnessEvent("final", {"text": "done"})


def test_cli_run_prints_final_answer(tmp_path, monkeypatch, capsys):
    cli = _load_cli()
    monkeypatch.setattr(cli, "HarnessRunner", FakeRunner)

    rc = cli.main(
        [
            "run",
            "--workspace",
            str(tmp_path),
            "--endpoint",
            "http://example/v1/chat/completions",
            "--model",
            "fake-model",
            "say hi",
        ]
    )

    assert rc == 0
    assert "done" in capsys.readouterr().out


def test_cli_run_exits_nonzero_on_denied_workspace(tmp_path, capsys):
    cli = _load_cli()

    rc = cli.main(
        [
            "run",
            "--workspace",
            str(tmp_path / "missing"),
            "--endpoint",
            "http://example/v1/chat/completions",
            "--model",
            "fake-model",
            "say hi",
        ]
    )

    assert rc == 1
    assert "workspace" in capsys.readouterr().err.lower()
