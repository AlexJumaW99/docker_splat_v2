"""ffprobe wrapper — returns best-effort video metadata; degrades to None."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    width: int | None
    height: int | None
    duration_s: float | None
    codec: str | None
    size_bytes: int | None

    def pretty(self) -> str:
        parts = []
        if self.width and self.height:
            parts.append(f"{self.width}×{self.height}")
        if self.duration_s is not None:
            parts.append(_fmt_duration(self.duration_s))
        if self.size_bytes is not None:
            parts.append(_fmt_size(self.size_bytes))
        if self.codec:
            parts.append(self.codec.upper())
        return " · ".join(parts) if parts else "no metadata"


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def probe(path: Path) -> VideoInfo | None:
    """Returns VideoInfo or None if ffprobe is missing / the call fails."""
    if not path.exists():
        return None

    size = path.stat().st_size

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return VideoInfo(None, None, None, None, size)

    try:
        r = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=5.0,
        )
    except subprocess.TimeoutExpired:
        return VideoInfo(None, None, None, None, size)
    if r.returncode != 0:
        return VideoInfo(None, None, None, None, size)

    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return VideoInfo(None, None, None, None, size)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    fmt = data.get("format", {})

    width = video_stream.get("width") if video_stream else None
    height = video_stream.get("height") if video_stream else None
    codec = video_stream.get("codec_name") if video_stream else None
    duration_s = None
    d = fmt.get("duration")
    if d is not None:
        try:
            duration_s = float(d)
        except (TypeError, ValueError):
            duration_s = None

    return VideoInfo(
        width=width, height=height,
        duration_s=duration_s, codec=codec,
        size_bytes=size,
    )
