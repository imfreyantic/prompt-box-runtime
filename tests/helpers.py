"""Shared test utilities (offline only)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from aiohttp import web

from box.config import Paths
from box.gateway import GatewayContext
from box.logs import Redactor
from box.state import BoxState

TOKEN = "0123456789abcdef" * 4
PROMPT_BOX_DIR = Path(__file__).resolve().parent.parent


class FakeClock:
    def __init__(self, now: float = 1_800_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_state() -> BoxState:
    return BoxState(versions={
        "llamaCommit": "c", "patchSha256": "p", "comfy": "v0.44.0", "quant": "Q8_K_P",
        "ctx": 131072, "mtp": "fastmtp", "runtimeCommit": None,
    })


def make_paths(tmp: Path) -> Paths:
    paths = Paths(root=tmp, runtime_dir=PROMPT_BOX_DIR)
    paths.logs.mkdir(parents=True, exist_ok=True)
    return paths


def make_ctx(tmp: Path, **kwargs) -> GatewayContext:
    defaults = dict(token=TOKEN, state=make_state(), paths=make_paths(tmp), redactor=Redactor([TOKEN]))
    defaults.update(kwargs)
    return GatewayContext(**defaults)


def ticket_for(exp: int, token: str = TOKEN) -> str:
    sig = hmac.new(token.encode(), f"comfy-login:{exp}".encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class RunningApp:
    """Serve an aiohttp app on an ephemeral port exactly like production (AppRunner)."""

    def __init__(self, app: web.Application, ssl_context=None):
        self.app = app
        self.ssl_context = ssl_context
        self.runner: Optional[web.AppRunner] = None
        self.port = 0

    async def __aenter__(self) -> "RunningApp":
        self.runner = web.AppRunner(self.app, handler_cancellation=True, access_log=None)
        await self.runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        site = web.TCPSite(self.runner, "127.0.0.1", self.port, ssl_context=self.ssl_context)
        await site.start()
        return self

    @property
    def url(self) -> str:
        scheme = "https" if self.ssl_context else "http"
        return f"{scheme}://127.0.0.1:{self.port}"

    async def __aexit__(self, *exc) -> None:
        assert self.runner is not None
        await self.runner.cleanup()


def make_self_signed(tmp: Path) -> Optional[tuple[bytes, bytes]]:
    """(cert_pem, key_pem) via the openssl CLI, or None if unavailable."""
    openssl = shutil.which("openssl")
    if openssl is None:
        return None
    cert, key = tmp / "c.pem", tmp / "k.pem"
    proc = subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
         "-days", "1", "-subj", "/CN=127.0.0.1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return cert.read_bytes(), key.read_bytes()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def tempdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="pb-test-")
