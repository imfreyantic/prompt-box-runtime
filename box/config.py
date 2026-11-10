"""Environment parsing and filesystem layout."""

from __future__ import annotations

import base64
import binascii
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from . import constants as C

TOKEN_RE = re.compile(r"[0-9a-f]{64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DEFAULT_ROOT = "/workspace/prompt-box"


class FatalConfigError(Exception):
    """The gateway cannot start (missing/malformed token or TLS material)."""


@dataclass(frozen=True)
class Paths:
    root: Path
    runtime_dir: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def llama_dir(self) -> Path:
        return self.root / "llama.cpp"

    @property
    def llama_bin_dir(self) -> Path:
        return self.llama_dir / "build" / "bin"

    @property
    def llama_server(self) -> Path:
        return self.llama_bin_dir / "llama-server"

    @property
    def comfy_dir(self) -> Path:
        return self.root / "ComfyUI"

    @property
    def venv_box(self) -> Path:
        return self.root / "venv-box"

    @property
    def venv_comfy(self) -> Path:
        return self.root / "venv-comfy"

    @property
    def tls_dir(self) -> Path:
        return self.root / "tls"

    @property
    def run_dir(self) -> Path:
        return self.root / "run"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    def log(self, source: str) -> Path:
        return self.logs / f"{source}.log"

    def ensure(self) -> None:
        for d in (self.models, self.logs, self.run_dir, self.state_dir, self.downloads):
            d.mkdir(parents=True, exist_ok=True)
        self.tls_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.tls_dir, 0o700)


@dataclass
class Config:
    token: str
    tls_cert_pem: bytes
    tls_key_pem: bytes
    tls_key_b64: str
    quant: str
    ctx: int
    mtp: str
    paths: Paths
    base_python: str
    prebuilt_url: Optional[str] = None
    prebuilt_sha256: Optional[str] = None
    build_jobs: Optional[int] = None
    runtime_commit_env: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text_model(self) -> Optional[C.PinnedFile]:
        return C.QUANT_FILES.get(self.quant)

    @property
    def use_prebuilt(self) -> bool:
        return bool(self.prebuilt_url and self.prebuilt_sha256)

    def secrets(self) -> list[str]:
        """Strings that must never appear in logs returned by the gateway."""
        out = [self.token, self.tls_key_b64]
        for line in self.tls_key_pem.decode("ascii", "replace").splitlines():
            line = line.strip()
            if len(line) >= 16 and not line.startswith("-----"):
                out.append(line)
        if self.prebuilt_url:
            out.append(self.prebuilt_url)
            if "?" in self.prebuilt_url:
                out.append(self.prebuilt_url.split("?", 1)[1])
        return [s for s in out if s]


def validate_token(token: Optional[str]) -> str:
    if token is None or token == "":
        raise FatalConfigError("PB_TOKEN is required (64 lowercase hex characters)")
    if not TOKEN_RE.fullmatch(token):
        raise FatalConfigError("PB_TOKEN is malformed: expected exactly 64 lowercase hex characters")
    return token


def decode_pem_b64(env: Mapping[str, str], name: str, kind: str) -> tuple[bytes, str]:
    raw = env.get(name, "")
    compact = "".join(raw.split())
    if not compact:
        raise FatalConfigError(f"{name} is required (base64 of a PEM {kind})")
    try:
        pem = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        raise FatalConfigError(f"{name} is not valid base64") from None
    if b"-----BEGIN " not in pem:
        raise FatalConfigError(f"{name} does not decode to a PEM {kind}")
    return pem, compact


def parse_ctx(raw: Optional[str], errors: list[str]) -> int:
    if raw is None or raw.strip() == "":
        return C.CTX_DEFAULT
    try:
        value = int(raw.strip())
    except ValueError:
        errors.append(f"PB_CTX must be an integer between {C.CTX_MIN} and {C.CTX_MAX}, got {raw!r}")
        return 0
    if not C.CTX_MIN <= value <= C.CTX_MAX:
        errors.append(f"PB_CTX must be between {C.CTX_MIN} and {C.CTX_MAX}, got {value}")
    return value


def load_config(env: Mapping[str, str] = os.environ) -> Config:
    """Parse the environment.

    Raises FatalConfigError when the gateway itself cannot run. Other problems are
    collected in ``Config.errors`` so they can be reported through /health.
    """
    token = validate_token(env.get("PB_TOKEN"))
    cert_pem, _ = decode_pem_b64(env, "PB_TLS_CERT_B64", "certificate")
    key_pem, key_b64 = decode_pem_b64(env, "PB_TLS_KEY_B64", "private key")

    errors: list[str] = []
    warnings: list[str] = []

    quant = (env.get("PB_QUANT") or "").strip() or C.DEFAULT_QUANT
    if quant not in C.QUANT_FILES:
        errors.append(f"PB_QUANT must be one of {', '.join(C.QUANT_FILES)}, got {quant!r}")

    ctx = parse_ctx(env.get("PB_CTX"), errors)

    mtp = (env.get("PB_MTP") or "").strip() or C.DEFAULT_MTP
    if mtp not in C.MTP_MODES:
        errors.append(f"PB_MTP must be one of {', '.join(C.MTP_MODES)}, got {mtp!r}")

    prebuilt_url = (env.get("PB_LLAMA_PREBUILT_URL") or "").strip() or None
    prebuilt_sha = (env.get("PB_LLAMA_PREBUILT_SHA256") or "").strip().lower() or None
    if bool(prebuilt_url) != bool(prebuilt_sha):
        warnings.append(
            "only one of PB_LLAMA_PREBUILT_URL / PB_LLAMA_PREBUILT_SHA256 is set; building llama.cpp from source"
        )
        prebuilt_url = prebuilt_sha = None
    elif prebuilt_sha and not SHA256_RE.fullmatch(prebuilt_sha):
        errors.append("PB_LLAMA_PREBUILT_SHA256 must be 64 hex characters")
    if prebuilt_url and not prebuilt_url.startswith(("https://", "http://")):
        errors.append("PB_LLAMA_PREBUILT_URL must be an http(s) URL")

    build_jobs: Optional[int] = None
    raw_jobs = (env.get("PB_BUILD_JOBS") or "").strip()
    if raw_jobs:
        try:
            build_jobs = max(1, int(raw_jobs))
        except ValueError:
            warnings.append(f"ignoring PB_BUILD_JOBS={raw_jobs!r} (not an integer)")

    root = Path(env.get("PB_ROOT") or DEFAULT_ROOT)
    runtime_dir = Path(__file__).resolve().parent.parent
    base_python = env.get("PB_BASE_PYTHON") or getattr(sys, "_base_executable", None) or sys.executable

    return Config(
        token=token,
        tls_cert_pem=cert_pem,
        tls_key_pem=key_pem,
        tls_key_b64=key_b64,
        quant=quant,
        ctx=ctx,
        mtp=mtp,
        paths=Paths(root=root, runtime_dir=runtime_dir),
        base_python=base_python,
        prebuilt_url=prebuilt_url,
        prebuilt_sha256=prebuilt_sha,
        build_jobs=build_jobs,
        runtime_commit_env=(env.get("PB_RUNTIME_COMMIT") or "").strip() or None,
        errors=errors,
        warnings=warnings,
    )


def write_tls_files(cfg: Config) -> tuple[Path, Path]:
    """Write cert/key with 0600 permissions (the directory is 0700)."""
    cfg.paths.tls_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cfg.paths.tls_dir, 0o700)
    cert_path = cfg.paths.tls_dir / "cert.pem"
    key_path = cfg.paths.tls_dir / "key.pem"
    for path, data in ((cert_path, cfg.tls_cert_pem), (key_path, cfg.tls_key_pem)):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    return cert_path, key_path
