"""
Host-side port utilities.

- is_port_listening(port): quick TCP probe on localhost.
- listeners_on(port): PIDs bound to that port (via `ss`).
- kill_port(port): SIGTERM (then SIGKILL) anything listening on that port.
  Scope is deliberately narrow — only user-owned processes; never sudo.
"""
from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import time


def is_port_listening(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def listeners_on(port: int) -> list[int]:
    """Returns PIDs that own a listening socket on `port`. Requires `ss`."""
    try:
        r = subprocess.run(
            ["ss", "-tlnpH"], capture_output=True, text=True, timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []

    pids: set[int] = set()
    pat_port = re.compile(rf":{port}\s")
    pat_pid = re.compile(r'pid=(\d+)')
    for line in r.stdout.splitlines():
        if not pat_port.search(line):
            continue
        for m in pat_pid.finditer(line):
            pids.add(int(m.group(1)))
    return sorted(pids)


def kill_port(port: int, grace: float = 1.5) -> list[int]:
    """SIGTERM, wait, then SIGKILL anything still listening on `port`.
    Returns the list of PIDs we tried to kill. Silently skips PIDs we
    don't own (no sudo escalation)."""
    pids = listeners_on(port)
    if not pids:
        return []

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not is_port_listening(port):
            return pids
        time.sleep(0.1)

    # Escalate
    for pid in listeners_on(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return pids
