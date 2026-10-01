import json
from pathlib import Path

import pytest

from swegen.harbor_pi_agent import PiOAuthAgent, resolve_pi_auth_file


def test_resolve_pi_auth_file_accepts_provider_map(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"openai-codex": {"type": "oauth"}}))

    assert resolve_pi_auth_file(auth_file) == auth_file.resolve()


def test_resolve_pi_auth_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="use /login"):
        resolve_pi_auth_file(tmp_path / "missing.json")


def test_resolve_pi_auth_file_rejects_empty_provider_map(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}")

    with pytest.raises(ValueError, match="contains no providers"):
        resolve_pi_auth_file(auth_file)


def test_harbor_agent_accepts_explicit_auth_file(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"openai-codex": {"type": "oauth"}}))

    agent = PiOAuthAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai-codex/gpt-5.5",
        auth_file=auth_file,
        thinking="high",
    )

    assert agent.model_name == "openai-codex/gpt-5.5"
    assert agent._host_auth_file == auth_file.resolve()
