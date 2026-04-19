"""
Safe Docker wrapper.

This module deliberately exposes a narrow allow-list of subcommands. Mutating
subcommands — build, create, rm, rmi, commit, run, tag, push, cp — are NOT
available as methods. The GUI physically cannot call them. `exec` IS allowed,
but only through narrow helpers here (port probe, pkill cleanup); long-running
exec streaming goes through JobManager, not this module.
"""
from __future__ import annotations

import shutil
import subprocess
import time
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


def container_port_in_use(
    port: int, name: str = CONTAINER_NAME
) -> bool | None:
    """Is anything INSIDE the container bound to `port`?

    This is the question you actually want to ask on a Docker setup — the
    host-side port is always held by docker-proxy while the container is
    running, so a host-side probe tells you nothing useful.

    Probe uses bash's built-in /dev/tcp so it works without iproute2/netstat
    being installed in the image. Exit 0 from bash means the connect to
    localhost:<port> inside the container succeeded, i.e. something is bound.

    Returns:
        True  — something inside the container is listening on `port`
        False — nothing is bound
        None  — container not running or we couldn't determine
    """
    state = container_state(name)
    if state.status != "running":
        return None
    try:
        r = _run(
            ["exec", name, "bash", "-c", f"exec 3<>/dev/tcp/127.0.0.1/{port}"],
            timeout=3.0,
        )
    except (subprocess.TimeoutExpired, DockerError):
        return None
    return r.returncode == 0


def free_container_port(
    port: int,
    name: str = CONTAINER_NAME,
    owners: tuple[str, ...] = ("ns-train", "ns-viewer"),
    grace: float = 1.5,
) -> bool:
    """Free an in-container port by pkill'ing the likely owners.

    Scope is deliberately narrow: only processes matching `owners` (the two
    things in this stack that bind to viser's port). SIGTERM first, re-probe
    until `grace` seconds elapse, then SIGKILL stragglers. Returns True iff
    the port is free when we return.
    """
    state = container_state(name)
    if state.status != "running":
        return False

    for patt in owners:
        try:
            _run(["exec", name, "pkill", "-TERM", "-f", patt], timeout=3.0)
        except (subprocess.TimeoutExpired, DockerError):
            pass

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if container_port_in_use(port, name) is False:
            return True
        time.sleep(0.15)

    for patt in owners:
        try:
            _run(["exec", name, "pkill", "-KILL", "-f", patt], timeout=3.0)
        except (subprocess.TimeoutExpired, DockerError):
            pass

    # Give the kernel a beat to tear down the socket, then re-check.
    time.sleep(0.3)
    return container_port_in_use(port, name) is False


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
