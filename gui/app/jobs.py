"""
JobManager: runs ONE `docker exec` streaming job at a time.

It's a thin QProcess wrapper that:
- Serializes jobs (refuses to start a second one while one is running)
- Emits signals for stdout/stderr lines, started, finished, failed
- Provides cancel() that pkills the inner process inside the container
  (docker exec doesn't reliably forward signals).

Design constraint: this module only ever invokes `docker exec`. It does not
call build/create/rm/commit/run, and it doesn't know how.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from app.config import CONTAINER_NAME


ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


@dataclass
class JobSpec:
    kind: str          # "process" | "train" | "viewer" | "export" | "generic"
    display_cmd: str   # human-readable command shown at the top of the log
    container_argv: list[str]   # argv run INSIDE the container
    # A regex run against command lines that match this job's inner process.
    # Used by cancel() to pkill the right thing. Narrow by design.
    kill_pattern: str
    # Optional host-side file that receives every output line (ANSI-stripped)
    # plus a header (command, start time) and a footer (exit status). Views
    # point this at the run's own warehouse directory (`<run>/job.log`) so a
    # failed or cancelled run leaves evidence on disk — before this existed,
    # six of seven unfinalized splatfacto-big runs had nothing to diagnose.
    log_path: Path | None = None


class JobManager(QObject):
    started = Signal(object)     # JobSpec
    line = Signal(str)           # stdout OR stderr line (ANSI stripped)
    finished = Signal(int)       # exit code (0 = ok)
    cancelled = Signal()
    failed = Signal(str)         # message (start failure)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._spec: JobSpec | None = None
        self._exit_handled = False
        self._log = None  # open text file while a logged job runs

    # ---- state ----
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    @property
    def current(self) -> JobSpec | None:
        return self._spec

    # ---- start ----
    def start(self, spec: JobSpec) -> None:
        if self.is_running():
            self.failed.emit("Another job is already running.")
            return

        argv = ["exec", CONTAINER_NAME, *spec.container_argv]

        self._spec = spec
        self._exit_handled = False
        self._open_log(spec)
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._drain_stdout)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)

        self.started.emit(spec)
        self._proc.start("docker", argv)

        if not self._proc.waitForStarted(3000):
            self._exit_handled = True
            err = self._proc.errorString() or "Failed to start docker exec."
            self._close_log(f"failed to start: {err}")
            self._proc = None
            self._spec = None
            self.failed.emit(err)

    # ---- cancel ----
    def cancel(self) -> None:
        """Best-effort cancel: pkill the inner process via a second
        `docker exec`, then terminate our QProcess."""
        if not self.is_running() or self._spec is None:
            return

        kill_argv = [
            "docker", "exec", CONTAINER_NAME,
            "pkill", "-TERM", "-f", self._spec.kill_pattern,
        ]
        try:
            QProcess.startDetached("docker", kill_argv[1:])
        except Exception:
            pass

        if self._proc is not None:
            self._proc.terminate()
            if not self._proc.waitForFinished(2000):
                self._proc.kill()

    # ---- internals ----
    def _drain_stdout(self) -> None:
        if self._proc is None:
            return
        data = self._proc.readAllStandardOutput().data().decode("utf-8", "replace")
        for raw_line in data.splitlines():
            clean = strip_ansi(raw_line)
            self._log_line(clean)
            self.line.emit(clean)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self._exit_handled:
            return
        self._exit_handled = True

        # flush any trailing bytes
        self._drain_stdout()

        crashed = (exit_status == QProcess.CrashExit)
        self._close_log("cancelled / crashed" if crashed else f"exit code {exit_code}")
        self._proc = None
        self._spec = None

        if crashed:
            self.cancelled.emit()
        else:
            self.finished.emit(exit_code)

    def _on_error(self, err: QProcess.ProcessError) -> None:
        if self._exit_handled:
            return
        self._exit_handled = True
        msg = self._proc.errorString() if self._proc else str(err)
        self._close_log(f"docker exec error: {msg}")
        self._proc = None
        self._spec = None
        self.failed.emit(msg or "docker exec error")

    # ---- log sink ----
    def _open_log(self, spec: JobSpec) -> None:
        """Best effort: a log that cannot be opened must never block a job."""
        self._log = None
        if spec.log_path is None:
            return
        try:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(spec.log_path, "a", encoding="utf-8")
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._log.write(f"# {stamp}  kind={spec.kind}\n# {spec.display_cmd}\n")
            self._log.flush()
        except OSError:
            self._log = None

    def _log_line(self, line: str) -> None:
        if self._log is None:
            return
        try:
            self._log.write(line + "\n")
        except OSError:
            self._log = None

    def _close_log(self, status: str) -> None:
        if self._log is None:
            return
        try:
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._log.write(f"# {stamp}  {status}\n")
            self._log.close()
        except OSError:
            pass
        self._log = None
