"""llama-server / ComfyUI command lines and a small process supervisor."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import time
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Optional, Sequence

from . import constants as C
from .procs import terminate

log = logging.getLogger("box.services")


def llama_server_args(
    binary: Path,
    model: Path,
    mmproj: Path,
    draft: Path,
    ctx: int,
    mtp: str,
) -> list[str]:
    """The exact llama-server command line for a PB_MTP mode.

    fastmtp: HauhauCS FastMTP sidecar as the draft model, depth 3.
    embedded: the model's own MTP head (no sidecar), depth 2.
    off: no speculative decoding flags.
    """
    if not C.CTX_MIN <= ctx <= C.CTX_MAX:
        raise ValueError(f"ctx {ctx} outside {C.CTX_MIN}..{C.CTX_MAX}")
    args = [str(binary), "--model", str(model), "--mmproj", str(mmproj)]
    if mtp == "fastmtp":
        args += [
            "--spec-draft-model", str(draft),
            "--spec-draft-ngl", "all",
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "3",
            "--spec-draft-p-min", "0",
        ]
    elif mtp == "embedded":
        args += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "2",
            "--spec-draft-p-min", "0",
        ]
    elif mtp != "off":
        raise ValueError(f"unknown PB_MTP mode {mtp!r}")
    args += [
        "--ctx-size", str(ctx),
        "--parallel", "1",
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--n-gpu-layers", "all",
        "--split-mode", "none",
        "--flash-attn", "on",
        "--no-mmap",
        "--temp", "0.7",
        "--top-k", "20",
        "--top-p", "0.8",
        "--min-p", "0",
        "--presence-penalty", "1.5",
        "--repeat-penalty", "1.0",
        "--jinja",
        "--reasoning", "on",
        "--reasoning-effort", "xhigh",
        "--reasoning-preserve",
        "--reasoning-format", "deepseek",
        "--alias", C.MODEL_ALIAS,
        "--host", C.LOCAL_HOST,
        "--port", str(C.LLAMA_PORT),
    ]
    return args


def comfy_args(venv_python: Path) -> list[str]:
    return [
        str(venv_python), "main.py",
        "--listen", C.LOCAL_HOST,
        "--port", str(C.COMFY_PORT),
        "--disable-auto-launch",
    ]


def kill_stale(pid_path: Path, expect: str, grace: float = 10.0) -> bool:
    """Stop a process left by a previous orchestrator (only if its cmdline matches)."""
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    if expect not in cmdline:
        return False
    log.warning("stopping stale process %d (%s) from a previous run", pid, expect)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return True
        except PermissionError:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return True
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not Path(f"/proc/{pid}").exists():
                return True
            time.sleep(0.2)
    return True


def port_in_use(port: int, host: str = C.LOCAL_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


StatusCallback = Callable[[str], None]


class Supervisor:
    """Starts a service, waits for health, and restarts it (with backoff) if it dies.

    The first start and each restart share one budget: after ``max_restarts``
    restarts the service is marked ``error`` and ``on_fatal`` is called.
    """

    def __init__(
        self,
        name: str,
        argv: Sequence[str],
        *,
        log_path: Path,
        health: Callable[[], Awaitable[bool]],
        on_status: StatusCallback,
        on_fatal: Callable[[str], None],
        on_note: Callable[[Optional[str]], None] = lambda _note: None,
        post_ready: Optional[Callable[[], Awaitable[None]]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        pid_path: Optional[Path] = None,
        max_restarts: int = C.MAX_SERVICE_RESTARTS,
        backoff: Sequence[float] = (10.0, 30.0, 60.0),
        start_timeout: float = 1800.0,
        poll_interval: float = 2.0,
        label: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.name = name
        self.label = label or name
        self.argv = list(argv)
        self.log_path = log_path
        self.health = health
        self.on_status = on_status
        self.on_fatal = on_fatal
        self.on_note = on_note
        self.post_ready = post_ready
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self.pid_path = pid_path
        self.max_restarts = max_restarts
        self.backoff = list(backoff) or [10.0]
        self.start_timeout = start_timeout
        self.poll_interval = poll_interval
        self.port = port
        self.restarts = 0
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._stopping = False

    async def _spawn(self) -> asyncio.subprocess.Process:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as logf:
            logf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting {self.label}: {' '.join(self.argv)}\n".encode())
            logf.flush()
            proc = await asyncio.create_subprocess_exec(
                *self.argv,
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=str(self.cwd) if self.cwd else None,
                env=self.env,
                start_new_session=True,
            )
        if self.pid_path is not None:
            self.pid_path.parent.mkdir(parents=True, exist_ok=True)
            self.pid_path.write_text(f"{proc.pid}\n")
        log.info("%s started (pid %d)", self.label, proc.pid)
        return proc

    async def _wait_healthy(self, proc: asyncio.subprocess.Process) -> bool:
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                return False
            try:
                if await self.health():
                    return proc.returncode is None
            except Exception:  # health probes must never kill the supervisor
                pass
            try:
                await asyncio.wait_for(proc.wait(), self.poll_interval)
            except asyncio.TimeoutError:
                pass
        log.error("%s did not become healthy within %.0fs", self.label, self.start_timeout)
        return False

    async def run(self) -> None:
        if self.port is not None and port_in_use(self.port):
            self.on_status("error")
            self.on_fatal(f"{self.label} cannot start: 127.0.0.1:{self.port} is already in use by another process "
                          "(e.g. Jupyter on 8080 - launch the instance without Jupyter)")
            return
        while not self._stopping:
            self.on_status("starting")
            try:
                proc = self.proc = await self._spawn()
            except OSError as exc:
                self.on_status("error")
                self.on_fatal(f"{self.label} could not be started: {exc}")
                return
            healthy = await self._wait_healthy(proc)
            if self._stopping:
                return
            if healthy:
                if self.post_ready is not None:
                    try:
                        await self.post_ready()
                    except Exception as exc:
                        self.on_status("error")
                        self.on_fatal(str(exc) or f"{self.label} readiness check failed ({exc.__class__.__name__})")
                        await proc.wait()
                        return
                self.on_note(None)
                self.on_status("ready")
                rc = await proc.wait()
                if self._stopping:
                    return
                log.error("%s exited with code %s", self.label, rc)
                reason = f"exited with code {rc}"
            else:
                if proc.returncode is None:
                    await terminate(proc)
                reason = f"failed to start (exit code {proc.returncode})"
            if self.restarts >= self.max_restarts:
                self.on_status("error")
                self.on_note(None)
                self.on_fatal(f"{self.label} {reason}; restart limit ({self.max_restarts}) reached, see logs")
                return
            delay = self.backoff[min(self.restarts, len(self.backoff) - 1)]
            self.restarts += 1
            self.on_status("starting")
            self.on_note(f"{self.label} {reason}; restarting in {delay:.0f}s ({self.restarts}/{self.max_restarts})")
            await asyncio.sleep(delay)

    async def stop(self) -> None:
        self._stopping = True
        if self.proc is not None:
            await terminate(self.proc)
