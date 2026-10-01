from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PiRunResult:
    """Result of one non-interactive Pi invocation."""

    returncode: int
    text: str
    error: str | None = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""


def build_pi_command(
    *,
    executable: str = "pi",
    model: str | None = None,
    thinking: str = "high",
    tools: list[str] | None = None,
    system_prompt: str | None = None,
    prompt: str,
) -> list[str]:
    """Build a deterministic one-shot Pi JSON-mode command."""
    command = shlex.split(executable)
    if not command:
        raise ValueError("Pi executable cannot be empty")

    args = [
        *command,
        "--mode",
        "json",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
    ]
    if model:
        args.extend(["--model", model])
    if thinking:
        args.extend(["--thinking", thinking])
    if tools:
        args.extend(["--tools", ",".join(tools)])
    else:
        args.append("--no-tools")
    if system_prompt:
        args.extend(["--system-prompt", system_prompt])
    args.extend(["--", prompt])
    return args


def _assistant_text(stdout: str) -> tuple[str, str | None]:
    """Extract final assistant text and provider error from Pi's JSONL stream."""
    text = ""
    error = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        parts = [
            block.get("text", "")
            for block in message.get("content", [])
            if block.get("type") == "text"
        ]
        if parts:
            text = "".join(parts)
        if message.get("stopReason") == "error":
            error = message.get("errorMessage") or text or "Pi model request failed"
    return text, error


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass


def run_pi(
    *,
    prompt: str,
    cwd: Path,
    timeout: int,
    executable: str = "pi",
    model: str | None = None,
    thinking: str = "high",
    tools: list[str] | None = None,
    system_prompt: str | None = None,
    verbose: bool = False,
) -> PiRunResult:
    """Run Pi once and return its authoritative final assistant message."""
    logger = logging.getLogger("swegen")
    command = build_pi_command(
        executable=executable,
        model=model,
        thinking=thinking,
        tools=tools,
        system_prompt=system_prompt,
        prompt=prompt,
    )
    binary = command[0]
    if not (Path(binary).exists() or shutil.which(binary)):
        return PiRunResult(
            returncode=127,
            text="",
            error=f"Pi executable not found: {binary}. Install pi or pass --pi-command.",
        )

    logger.info("Invoking Pi with %ds timeout%s", timeout, f" ({model})" if model else "")
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        return PiRunResult(returncode=127, text="", error=f"Could not start Pi: {exc}")

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        text, provider_error = _assistant_text(stdout)
        if verbose and stderr:
            print(stderr, end="", flush=True)
        return PiRunResult(
            returncode=proc.returncode or 124,
            text=text,
            error=provider_error or f"Pi timed out after {timeout}s",
            timed_out=True,
            stdout=stdout,
            stderr=stderr,
        )

    text, provider_error = _assistant_text(stdout)
    if verbose:
        if text:
            print(text, flush=True)
        if stderr:
            print(stderr, end="", flush=True)

    error = provider_error
    if proc.returncode != 0 and not error:
        error = stderr.strip() or text.strip() or f"Pi exited with status {proc.returncode}"
    return PiRunResult(
        returncode=proc.returncode,
        text=text,
        error=error,
        stdout=stdout,
        stderr=stderr,
    )
