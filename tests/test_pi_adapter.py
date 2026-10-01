from __future__ import annotations

import json
from pathlib import Path

import pytest

from swegen.create import task_instruction
from swegen.create.pi_cli import PiRunResult, build_pi_command, run_pi
from swegen.create.pi_runner import _completion_prompt, run_completion_agent


def test_build_pi_command_isolated_and_tool_limited() -> None:
    command = build_pi_command(
        executable="pi",
        model="openai-codex/gpt-5.5",
        thinking="high",
        tools=["read", "bash"],
        system_prompt="system",
        prompt="prompt",
    )

    assert command[:3] == ["pi", "--mode", "json"]
    assert "--no-session" in command
    assert "--no-extensions" in command
    assert command[command.index("--tools") + 1] == "read,bash"
    assert command[-2:] == ["--", "prompt"]


def test_run_pi_parses_final_assistant_message(tmp_path: Path) -> None:
    fake_pi = tmp_path / "fake-pi"
    message = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stopReason": "stop",
        },
    }
    fake_pi.write_text(f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(message)}'\n")
    fake_pi.chmod(0o755)

    result = run_pi(prompt="work", cwd=tmp_path, timeout=5, executable=str(fake_pi))

    assert result.returncode == 0
    assert result.text == "done"
    assert result.error is None


def test_pi_evaluator_uses_subscription_runner_without_openai_key(monkeypatch) -> None:
    payload = {
        "is_substantial": True,
        "reason": "Meaningful behavior across modules",
        "instruction": "A" * 120,
        "difficulty": "medium",
        "category": "bugfix",
        "tags": ["typescript", "frontend", "react"],
        "task_name": None,
    }

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        task_instruction,
        "run_pi",
        lambda **_kwargs: PiRunResult(0, json.dumps(payload)),
    )

    result = task_instruction.evaluate_and_generate_task(
        metadata={"title": "Fix behavior", "body": "Detailed problem"},
        files=[{"filename": "src/a.ts"}, {"filename": "src/b.ts"}],
        repo="owner/repo",
        backend="pi",
        pi_model="openai-codex/gpt-5.5",
    )

    assert result.is_substantial is True
    assert result.tags == ["typescript", "frontend", "react"]


def test_completion_prompt_names_pi_and_preserves_paths(tmp_path: Path) -> None:
    prompt = _completion_prompt(
        repo="owner/repo",
        pr_number=1,
        repo_path=tmp_path / "repo",
        task_dir=tmp_path / "tasks" / "owner__repo-1",
        task_id="owner__repo-1",
        dataset_path=tmp_path / "tasks",
        jobs_dir=tmp_path / "jobs",
        test_files=["tests/example.test.ts"],
        reference_task_id=None,
        reference_pr=None,
        head_sha="abc",
        environment="docker",
        enforce_offline_tests=True,
    )

    assert "Claude Code" not in prompt
    assert "tests/example.test.ts" in prompt
    assert "tests/test.sh must be fully offline" in prompt


def test_completion_dispatch_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown completion agent"):
        run_completion_agent("other")
