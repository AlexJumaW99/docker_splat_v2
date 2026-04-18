"""
Safe Docker wrapper.

This module deliberately exposes a narrow allow-list of subcommands. Mutating
subcommands — build, create, rm, rmi, commit, run, tag, push, cp — are NOT
available as methods. The GUI physically cannot call them.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from app.config import CONTAINER_NAME, IMAGE_NAME


class DockerError(Exception):
    pass


@dataclass
class ContainerState:
    status: str  # "missing" | "running" | "exited" | "created" | "paused" | "restarting" | "dead"
    id: str | None = None


def _docker_bin() -> str:
    path = shutil.which("docker")
    if path is None:
        raise DockerError("docker CLI not found on PATH")
    return path


def _run(args: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_docker_bin(), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def daemon_version() -> str | None:
    """Returns the Docker server version, or None if daemon is unreachable."""
    try:
        r = _run(["version", "--format", "{{.Server.Version}}"], timeout=2.0)
    except (subprocess.TimeoutExpired, DockerError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def image_exists(name: str = IMAGE_NAME) -> bool:
    try:
        r = _run(["image", "inspect", name], timeout=2.0)
    except (subprocess.TimeoutExpired, DockerError):
        return False
    return r.returncode == 0


def container_state(name: str = CONTAINER_NAME) -> ContainerState:
    try:
        r = _run(
            ["inspect", "--type=container",
             "--format", "{{.State.Status}}\t{{.Id}}", name],
            timeout=2.0,
        )
    except (subprocess.TimeoutExpired, DockerError):
        return ContainerState(status="missing")
    if r.returncode != 0:
        return ContainerState(status="missing")
    parts = r.stdout.strip().split("\t")
    return ContainerState(
        status=parts[0] if parts else "missing",
        id=parts[1] if len(parts) > 1 else None,
    )


def start_container(name: str = CONTAINER_NAME) -> None:
    """Starts an EXISTING container. Raises if the container doesn't exist —
    this wrapper never creates containers."""
    state = container_state(name)
    if state.status == "missing":
        raise DockerError(
            f"Container '{name}' does not exist. This GUI will not create it — "
            f"see USER_GUIDE.md."
        )
    if state.status == "running":
        return
    try:
        r = _run(["start", name], timeout=15.0)
    except subprocess.TimeoutExpired as e:
        raise DockerError(f"Timeout starting container: {e}") from e
    if r.returncode != 0:
        raise DockerError(f"docker start failed: {r.stderr.strip()}")


def start_daemon_via_polkit() -> tuple[bool, str]:
    """Prompts for authorization via pkexec, then runs `systemctl start docker`.
    Returns (ok, message). Does nothing destructive on failure."""
    pkexec = shutil.which("pkexec")
    if pkexec is None:
        return False, "pkexec not found on PATH"
    try:
        r = subprocess.run(
            [pkexec, "systemctl", "start", "docker"],
            capture_output=True, text=True, timeout=30.0,
        )
    except subprocess.TimeoutExpired:
        return False, "Timed out waiting for authorization."
    if r.returncode != 0:
        return False, (r.stderr.strip() or f"pkexec exited with code {r.returncode}")
    return True, "Docker daemon started."
