"""System tools (git, compilers) and Python virtual environments."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

from .procs import StepError, run_logged

log = logging.getLogger("box.tools")

# command -> Debian/Ubuntu package
APT_PACKAGES = {"git": "git", "g++": "g++", "make": "make"}

GIT = ("git", "-c", "safe.directory=*", "-c", "advice.detachedHead=false")

_apt_lock = asyncio.Lock()


def missing_commands(commands: Iterable[str]) -> list[str]:
    return [c for c in commands if shutil.which(c) is None]


async def ensure_commands(commands: Iterable[str], log_path: Path) -> list[str]:
    """Install missing commands with apt-get when possible. Returns what is still missing."""
    async with _apt_lock:
        missing = missing_commands(commands)
        if not missing:
            return []
        apt = shutil.which("apt-get")
        is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        if apt is None or not is_root:
            log.warning("missing %s and cannot apt-get install (apt-get=%s, root=%s)", missing, apt, is_root)
            return missing
        packages = sorted({APT_PACKAGES.get(c, c) for c in missing})
        log.info("installing %s with apt-get", " ".join(packages))
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        rc = await run_logged([apt, "update", "-qq"], log_path, env=env, timeout=900)
        if rc != 0:
            log.warning("apt-get update failed with code %d", rc)
        rc = await run_logged(
            [apt, "install", "-y", "-qq", "--no-install-recommends", *packages], log_path, env=env, timeout=1800
        )
        if rc != 0:
            log.warning("apt-get install %s failed with code %d", packages, rc)
        return missing_commands(missing)


async def _pip_works(python: Path) -> bool:
    if not python.exists():
        return False
    proc = await asyncio.create_subprocess_exec(
        str(python), "-m", "pip", "--version",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    return await proc.wait() == 0


async def ensure_venv(
    venv: Path, base_python: str, log_path: Path, *, system_site_packages: bool = False
) -> Path:
    """Create ``venv`` (idempotent) with a working pip.

    Ubuntu's system Python lacks ensurepip unless python3-venv is installed, so this
    falls back to ``--without-pip`` + seeding pip from the system pip, then to
    installing python3-venv with apt-get.
    """
    python = venv / "bin" / "python"
    if await _pip_works(python):
        return python
    flags = ["--system-site-packages"] if system_site_packages else []
    rc = await run_logged([base_python, "-m", "venv", *flags, str(venv)], log_path, timeout=600)
    if rc == 0 and await _pip_works(python):
        return python
    log.warning("python -m venv %s failed (ensurepip missing?); seeding pip from the system pip", venv)
    await run_logged([base_python, "-m", "venv", "--without-pip", *flags, str(venv)], log_path, timeout=600)
    await run_logged(
        [base_python, "-m", "pip", "--python", str(python), "install",
         "--disable-pip-version-check", "--no-input", "pip"],
        log_path, timeout=900,
    )
    if await _pip_works(python):
        return python
    apt = shutil.which("apt-get")
    if apt and hasattr(os, "geteuid") and os.geteuid() == 0:
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        async with _apt_lock:
            await run_logged([apt, "update", "-qq"], log_path, env=env, timeout=900)
            await run_logged([apt, "install", "-y", "-qq", "--no-install-recommends", "python3-venv"],
                             log_path, env=env, timeout=1800)
        await run_logged([base_python, "-m", "venv", *flags, str(venv)], log_path, timeout=600)
        if await _pip_works(python):
            return python
    raise StepError(f"could not create a Python virtual environment at {venv}")


async def git(args: list[str], log_path: Path, *, cwd: Optional[Path] = None, timeout: float = 1800) -> int:
    return await run_logged([*GIT, *args], log_path, cwd=cwd, timeout=timeout)
