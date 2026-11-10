"""Subprocess helpers that stream output into log files."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


class StepError(Exception):
    """A setup step failed; the message is shown in /health."""


def _kill_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


async def terminate(proc: asyncio.subprocess.Process, grace: float = 20.0) -> None:
    """SIGTERM the process group, then SIGKILL after ``grace`` seconds."""
    if proc.returncode is not None:
        return
    _kill_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        _kill_group(proc, signal.SIGKILL)
        await proc.wait()


async def run_logged(
    argv: Sequence[str],
    log_path: Path,
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    on_line: Optional[Callable[[str], None]] = None,
) -> int:
    """Run ``argv`` appending stdout+stderr to ``log_path``; returns the exit code.

    ``on_line`` receives each output line (``\\r`` also ends a line) so callers can
    parse progress. A missing executable returns 127 instead of raising.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as logf:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logf.write(f"[{stamp}] $ {' '.join(shlex.quote(str(a)) for a in argv)}\n".encode())
        try:
            proc = await asyncio.create_subprocess_exec(
                *[str(a) for a in argv],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=str(cwd) if cwd else None,
                env=dict(env) if env is not None else None,
                start_new_session=True,
            )
        except (FileNotFoundError, PermissionError) as exc:
            logf.write(f"[could not start: {exc}]\n".encode())
            return 127

        async def pump() -> None:
            assert proc.stdout is not None
            carry = ""
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                logf.write(chunk)
                if on_line is not None:
                    text = carry + chunk.decode("utf-8", "replace").replace("\r", "\n")
                    *lines, carry = text.split("\n")
                    for line in lines:
                        if line:
                            on_line(line)
            if on_line is not None and carry:
                on_line(carry)

        try:
            await asyncio.wait_for(asyncio.gather(pump(), proc.wait()), timeout)
        except asyncio.TimeoutError:
            await terminate(proc, grace=5)
            logf.write(b"[timed out]\n")
            return 124
        except asyncio.CancelledError:
            await terminate(proc, grace=5)
            raise
        rc = proc.returncode if proc.returncode is not None else -1
        logf.write(f"[exit code {rc}]\n".encode())
        return rc
