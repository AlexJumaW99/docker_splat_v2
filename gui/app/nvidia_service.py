from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class GpuStats:
    name: str
    temp_c: int
    mem_used_mb: int
    mem_total_mb: int
    util_gpu: int
    util_mem: int

    @property
    def mem_frac(self) -> float:
        return self.mem_used_mb / self.mem_total_mb if self.mem_total_mb else 0.0


def query_gpu() -> GpuStats | None:
    """Polls nvidia-smi once. Returns None if nvidia-smi is absent or fails."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        r = subprocess.run(
            [smi,
             "--query-gpu=name,temperature.gpu,memory.used,memory.total,"
             "utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    line = r.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        return None
    try:
        return GpuStats(
            name=parts[0],
            temp_c=int(parts[1]),
            mem_used_mb=int(parts[2]),
            mem_total_mb=int(parts[3]),
            util_gpu=int(parts[4]),
            util_mem=int(parts[5]),
        )
    except ValueError:
        return None
