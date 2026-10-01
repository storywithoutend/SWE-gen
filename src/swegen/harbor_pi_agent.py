"""Harbor Pi agent that carries the host's Pi OAuth session into the sandbox."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import override

from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_DEFAULT_AUTH_FILE = Path.home() / ".pi" / "agent" / "auth.json"


def resolve_pi_auth_file(auth_file: str | Path | None = None) -> Path:
    """Resolve and validate the host Pi authentication file."""
    path = Path(auth_file).expanduser() if auth_file else _DEFAULT_AUTH_FILE
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Pi authentication file not found: {path}. Run Pi and use /login first."
        )

    try:
        credentials = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Pi authentication file is not valid JSON: {path}") from exc
    if not isinstance(credentials, dict) or not credentials:
        raise ValueError(f"Pi authentication file contains no providers: {path}")
    return path


class PiOAuthAgent(Pi):
    """Run Harbor's Pi agent with a temporary copy of host OAuth credentials."""

    def __init__(
        self,
        *args,
        auth_file: str | Path | None = None,
        **kwargs,
    ) -> None:
        self._host_auth_file = resolve_pi_auth_file(auth_file)
        self._container_auth_file: str | None = None
        super().__init__(*args, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)

        identity = await environment.exec(
            command='printf \'%s\\n%s\\n%s\\n\' "$HOME" "$(id -u)" "$(id -g)"'
        )
        lines = (identity.stdout or "").splitlines()
        if len(lines) != 3 or not lines[0].startswith("/"):
            raise RuntimeError("Could not determine the Harbor agent user's home directory")

        home, uid, gid = lines
        auth_dir = f"{home}/.pi/agent"
        auth_file = f"{auth_dir}/auth.json"
        quoted_dir = shlex.quote(auth_dir)
        quoted_file = shlex.quote(auth_file)

        await self.exec_as_root(environment, command=f"mkdir -p {quoted_dir}")
        await environment.upload_file(self._host_auth_file, auth_file)
        await self.exec_as_root(
            environment,
            command=f"chown {uid}:{gid} {quoted_file} && chmod 600 {quoted_file}",
        )
        self._container_auth_file = auth_file

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        try:
            await super().run(instruction, environment, context)
        finally:
            if self._container_auth_file:
                try:
                    await self.exec_as_root(
                        environment,
                        command=f"rm -f {shlex.quote(self._container_auth_file)}",
                    )
                except Exception:
                    self.logger.warning("Could not remove the sandbox Pi credential copy")
