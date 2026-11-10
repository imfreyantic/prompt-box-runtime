"""Host inspection helpers: GPU, CPU/memory limits for the build, disk space, git commit."""

from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

GPU_QUERY = "name,memory.total,power.limit,power.max_limit,driver_version"
_CUDA_RE = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)")


async def run_capture(argv: list[str], timeout: float = 30.0, cwd: Optional[Path] = None) -> tuple[int, str]:
    """Run a short command and capture combined output. Returns (rc, output); rc=127 if missing."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
        )
    except (FileNotFoundError, PermissionError) as exc:
        return 127, str(exc)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "timeout"
    return proc.returncode or 0, out.decode("utf-8", "replace")


def _float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def _int(value: str) -> Optional[int]:
    f = _float(value)
    return int(f) if f is not None else None


def parse_gpu_query(csv_text: str, smi_text: str = "") -> Optional[dict[str, Any]]:
    """Parse the first GPU line of ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``."""
    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    if not lines:
        return None
    fields = [f.strip() for f in lines[0].split(",")]
    if len(fields) < 5:
        return None
    name, mem, plimit, pmax, driver = fields[:5]
    cuda = _CUDA_RE.search(smi_text)
    return {
        "name": name,
        "vramMiB": _int(mem),
        "powerLimitW": _float(plimit),
        "powerMaxW": _float(pmax),
        "driver": driver,
        "cuda": cuda.group(1) if cuda else None,
    }


async def gpu_info() -> Optional[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return None
    rc, csv_text = await run_capture(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"], timeout=30
    )
    if rc != 0:
        return None
    _, smi_text = await run_capture(["nvidia-smi"], timeout=30)
    return parse_gpu_query(csv_text, smi_text)


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_cpu_limit(cgroup_root: Path = Path("/sys/fs/cgroup")) -> Optional[float]:
    v2 = _read(cgroup_root / "cpu.max")
    if v2:
        quota, _, period = v2.partition(" ")
        if quota != "max" and period:
            try:
                return int(quota) / int(period)
            except ValueError:
                return None
        return None
    quota = _read(cgroup_root / "cpu" / "cpu.cfs_quota_us") or _read(cgroup_root / "cpu.cfs_quota_us")
    period = _read(cgroup_root / "cpu" / "cpu.cfs_period_us") or _read(cgroup_root / "cpu.cfs_period_us")
    try:
        if quota and period and int(quota) > 0:
            return int(quota) / int(period)
    except ValueError:
        pass
    return None


def memory_limit_bytes(cgroup_root: Path = Path("/sys/fs/cgroup"), meminfo: Path = Path("/proc/meminfo")) -> Optional[int]:
    limits: list[int] = []
    for candidate in (cgroup_root / "memory.max", cgroup_root / "memory" / "memory.limit_in_bytes"):
        raw = _read(candidate)
        if raw and raw != "max":
            try:
                value = int(raw)
            except ValueError:
                continue
            if 0 < value < (1 << 60):
                limits.append(value)
    info = _read(meminfo)
    if info:
        m = re.search(r"^MemTotal:\s+(\d+)\s+kB", info, re.M)
        if m:
            limits.append(int(m.group(1)) * 1024)
    return min(limits) if limits else None


def effective_build_jobs(
    override: Optional[int] = None,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    meminfo: Path = Path("/proc/meminfo"),
    mem_per_job: int = int(2.5 * 1024**3),
) -> int:
    """nproc, capped by the container CPU quota and by memory (nvcc jobs are memory hungry)."""
    if override:
        return max(1, override)
    if hasattr(os, "sched_getaffinity"):
        jobs = len(os.sched_getaffinity(0))
    else:  # pragma: no cover - macOS
        jobs = os.cpu_count() or 1
    quota = cgroup_cpu_limit(cgroup_root)
    if quota:
        jobs = min(jobs, max(1, math.ceil(quota)))
    mem = memory_limit_bytes(cgroup_root, meminfo)
    if mem:
        jobs = min(jobs, max(1, mem // mem_per_job))
    return max(1, int(jobs))


def allocated_bytes(path: Path) -> int:
    try:
        st = path.stat()
    except FileNotFoundError:
        return 0
    blocks = getattr(st, "st_blocks", None)
    return blocks * 512 if blocks is not None else st.st_size


async def git_head(repo: Path) -> Optional[str]:
    if shutil.which("git") is None or not (repo / ".git").exists():
        return None
    rc, out = await run_capture(["git", "-c", "safe.directory=*", "-C", str(repo), "rev-parse", "HEAD"], timeout=15)
    out = out.strip()
    return out if rc == 0 and re.fullmatch(r"[0-9a-f]{40}", out) else None
