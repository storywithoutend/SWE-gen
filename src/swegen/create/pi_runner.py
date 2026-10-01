from __future__ import annotations

import logging
from pathlib import Path

from .claude_code_runner import (
    CC_PROMPT,
    CC_REFERENCE_PROMPT,
    TEST_OFFLINE_CONSTRAINT,
    ClaudeCodeResult,
    _check_validation_state,
)
from .pi_cli import run_pi


def _completion_prompt(
    *,
    repo: str,
    pr_number: int,
    repo_path: Path,
    task_dir: Path,
    task_id: str,
    dataset_path: Path,
    jobs_dir: Path,
    test_files: list[str],
    reference_task_id: str | None,
    reference_pr: int | None,
    head_sha: str | None,
    environment: str,
    enforce_offline_tests: bool,
) -> str:
    test_files_list = "\n".join(f"  - {path}" for path in test_files) or "  (none)"
    if reference_task_id and reference_pr:
        prompt = CC_REFERENCE_PROMPT.format(
            repo=repo,
            pr_number=pr_number,
            reference_pr=reference_pr,
            reference_task_id=reference_task_id,
            reference_task_dir=(dataset_path / reference_task_id).resolve(),
            repo_path=repo_path,
            task_dir=task_dir,
            task_id=task_id,
            dataset_path=dataset_path,
            jobs_dir=jobs_dir,
            test_files_list=test_files_list,
            head_sha=head_sha or "(check metadata)",
            environment=environment,
        )
    else:
        prompt = CC_PROMPT.format(
            repo=repo,
            pr_number=pr_number,
            repo_path=repo_path,
            task_dir=task_dir,
            task_id=task_id,
            dataset_path=dataset_path,
            jobs_dir=jobs_dir,
            test_files_list=test_files_list,
            environment=environment,
        )
    # The original prompt is backend-neutral except for its name. Avoid telling Pi that it
    # is Claude Code while retaining one source of truth for the detailed task contract.
    prompt = prompt.replace("Claude Code", "Pi coding agent").replace("CC session", "Pi session")
    if enforce_offline_tests:
        prompt += "\n" + TEST_OFFLINE_CONSTRAINT
    return prompt


def run_pi_session(
    repo: str,
    pr_number: int,
    repo_path: Path,
    task_dir: Path,
    task_id: str,
    dataset_path: Path,
    test_files: list[str],
    timeout: int = 900,
    verbose: bool = False,
    reference_task_id: str | None = None,
    reference_pr: int | None = None,
    head_sha: str | None = None,
    environment: str = "docker",
    enforce_offline_tests: bool = True,
    pi_model: str | None = None,
    pi_thinking: str = "high",
    pi_command: str = "pi",
) -> ClaudeCodeResult:
    """Use Pi to complete a task skeleton and drive Harbor NOP/Oracle validation."""
    logger = logging.getLogger("swegen")
    dataset_path = Path(dataset_path).resolve()
    task_dir = Path(task_dir).resolve()
    repo_path = Path(repo_path).resolve()
    jobs_dir = (dataset_path.parent / ".swegen" / "harbor-jobs").resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    prompt = _completion_prompt(
        repo=repo,
        pr_number=pr_number,
        repo_path=repo_path,
        task_dir=task_dir,
        task_id=task_id,
        dataset_path=dataset_path,
        jobs_dir=jobs_dir,
        test_files=test_files,
        reference_task_id=reference_task_id,
        reference_pr=reference_pr,
        head_sha=head_sha,
        environment=environment,
        enforce_offline_tests=enforce_offline_tests,
    )
    result = run_pi(
        prompt=prompt,
        cwd=Path.cwd(),
        timeout=timeout,
        executable=pi_command,
        model=pi_model,
        thinking=pi_thinking,
        tools=["read", "bash", "edit", "write", "grep", "find", "ls"],
        verbose=verbose,
    )

    validation = _check_validation_state(jobs_dir, task_id, logger, timed_out=result.timed_out)
    validation.cc_output = result.text
    if validation.success:
        return validation

    details = [validation.error_message, result.error]
    validation.error_message = "; ".join(dict.fromkeys(item for item in details if item)) or None
    if result.returncode != 0:
        logger.error("Pi completion failed: %s", validation.error_message)
    return validation


def run_completion_agent(
    backend: str,
    **kwargs,
) -> ClaudeCodeResult:
    """Dispatch task completion to the configured coding agent."""
    if backend == "pi":
        return run_pi_session(**kwargs)
    if backend == "claude":
        from .claude_code_runner import run_claude_code_session

        # Pi-only configuration is accepted by the common call site, not Claude's runner.
        for key in ("pi_model", "pi_thinking", "pi_command"):
            kwargs.pop(key, None)
        return run_claude_code_session(**kwargs)
    raise ValueError(f"Unknown completion agent: {backend!r}; expected 'claude' or 'pi'")
