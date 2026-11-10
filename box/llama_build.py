"""Fetch, patch and build llama.cpp (or install a verified prebuilt tarball)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable, Optional

import aiohttp

from . import constants as C
from .config import Config
from .download import fetch_pinned_small_file, fetch_url_to_file, safe_url
from .procs import StepError, run_logged
from .state import BoxState
from .sysinfo import effective_build_jobs, run_capture
from .tools import GIT, ensure_commands, git

log = logging.getLogger("box.llama")

_NINJA_RE = re.compile(r"^\[(\d+)/(\d+)\]")
_MAKE_RE = re.compile(r"^\[\s*(\d+)%\]")


def parse_build_percent(line: str) -> Optional[int]:
    """Progress from ninja (``[12/345] ...``) or make (``[ 42%] ...``) output."""
    m = _NINJA_RE.match(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        return int(done * 100 / total) if total else None
    m = _MAKE_RE.match(line)
    if m:
        return min(100, int(m.group(1)))
    return None


def build_marker_expected() -> dict:
    return {
        "commit": C.LLAMA_COMMIT,
        "patchSha256": C.PATCH_FILE.sha256,
        "cmakeArgs": list(C.LLAMA_CMAKE_ARGS),
        "target": C.LLAMA_BUILD_TARGET,
    }


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def build_env(venv_bin: Path, build_dir: Path) -> dict[str, str]:
    """PATH with pip-installed cmake/ninja and CUDA; Ninja unless an old cache says otherwise."""
    env = dict(os.environ)
    path = [str(venv_bin)]
    if Path("/usr/local/cuda/bin").is_dir():
        path.append("/usr/local/cuda/bin")
        if Path("/usr/local/cuda/bin/nvcc").exists():
            env.setdefault("CUDACXX", "/usr/local/cuda/bin/nvcc")
    env["PATH"] = os.pathsep.join(path + [env.get("PATH", "")])
    cache = build_dir / "CMakeCache.txt"
    cached_gen = None
    if cache.exists():
        m = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.+)$", cache.read_text(errors="replace"), re.M)
        cached_gen = m.group(1).strip() if m else None
    if cached_gen:
        env["CMAKE_GENERATOR"] = cached_gen
    elif shutil.which("ninja", path=env["PATH"]):
        env["CMAKE_GENERATOR"] = "Ninja"
    return env


async def ensure_checkout(repo: Path, log_path: Path) -> None:
    """Shallow-fetch exactly LLAMA_COMMIT into ``repo`` and check it out (idempotent)."""
    if not (repo / ".git").exists():
        if repo.exists() and any(repo.iterdir()):
            raise StepError(f"{repo} exists but is not a git checkout; refusing to modify it")
        repo.mkdir(parents=True, exist_ok=True)
        if await git(["init", "-q", str(repo)], log_path) != 0:
            raise StepError("git init failed")
    rc, head = await run_capture([*GIT, "-C", str(repo), "rev-parse", "HEAD"])
    if rc == 0 and head.strip() == C.LLAMA_COMMIT:
        return
    if await git(["-C", str(repo), "fetch", "--depth", "1", C.LLAMA_REPO_URL, C.LLAMA_COMMIT], log_path) != 0:
        log.warning("shallow fetch of %s failed; fetching full history", C.LLAMA_COMMIT)
        if await git(["-C", str(repo), "fetch", C.LLAMA_REPO_URL], log_path, timeout=3600) != 0:
            raise StepError(f"could not fetch llama.cpp {C.LLAMA_COMMIT}")
    # --force only touches tracked files (e.g. our earlier patch); build/ is untracked and kept.
    if await git(["-C", str(repo), "checkout", "--force", "--detach", C.LLAMA_COMMIT], log_path) != 0:
        raise StepError(f"could not check out llama.cpp {C.LLAMA_COMMIT}")
    rc, head = await run_capture([*GIT, "-C", str(repo), "rev-parse", "HEAD"])
    if rc != 0 or head.strip() != C.LLAMA_COMMIT:
        raise StepError(f"llama.cpp HEAD is {head.strip()!r}, expected {C.LLAMA_COMMIT}")


def _note(log_path: Path, text: str) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")


async def ensure_patch_applied(repo: Path, patch: Path, log_path: Path) -> str:
    """Apply the FastMTP patch once. Returns 'already-applied' or 'applied'."""
    _note(log_path, "checking whether the FastMTP patch is already applied (a failed reverse check is expected on first run)")
    if await git(["apply", "--reverse", "--check", str(patch)], log_path, cwd=repo) == 0:
        return "already-applied"
    if await git(["apply", "--check", str(patch)], log_path, cwd=repo) != 0:
        raise StepError("the FastMTP patch does not apply cleanly to the llama.cpp checkout")
    if await git(["apply", str(patch)], log_path, cwd=repo) != 0:
        raise StepError("git apply of the FastMTP patch failed")
    return "applied"


async def verify_binary(binary: Path, log_path: Path) -> str:
    if not binary.is_file():
        raise StepError(f"{binary} was not produced")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(binary.parent), env.get("LD_LIBRARY_PATH")]))
    rc = await run_logged([str(binary), "--version"], log_path, env=env, timeout=120)
    if rc != 0:
        raise StepError(f"{binary.name} --version failed with code {rc} (see /logs?source=llama)")
    return "ok"


async def ensure_build_tools(venv_bin: Path, runtime_dir: Path, log_path: Path) -> None:
    """pip-install the pinned cmake/ninja into the orchestrator venv (once)."""
    if (venv_bin / "cmake").exists() and (venv_bin / "ninja").exists():
        return
    rc = await run_logged(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--progress-bar", "off",
         "-r", str(runtime_dir / "requirements-build.txt")],
        log_path, timeout=1800,
    )
    if rc != 0:
        raise StepError(f"pip install of cmake/ninja failed with code {rc} (see /logs?source=llama)")


async def build_from_source(
    cfg: Config, state: BoxState, session: aiohttp.ClientSession, log_path: Path, hf_base: str = C.HF_BASE
) -> Path:
    paths = cfg.paths
    repo = paths.llama_dir
    binary = paths.llama_server
    marker = repo / "build" / ".pb-build.json"
    if _read_json(marker) == build_marker_expected() and binary.is_file():
        log.info("llama.cpp already built at %s", binary)
        state.build_percent = 100
        return binary

    venv_bin = Path(sys.executable).parent
    await ensure_build_tools(venv_bin, paths.runtime_dir, log_path)
    env = build_env(venv_bin, repo / "build")
    needed = ["git", "g++"] + ([] if env.get("CMAKE_GENERATOR") == "Ninja" else ["make"])
    still_missing = await ensure_commands(needed, paths.log("box"))
    if still_missing:
        raise StepError(f"required build tools are missing: {', '.join(still_missing)} (apt-get install failed?)")
    if shutil.which("cmake", path=env["PATH"]) is None:
        raise StepError("cmake not found after installing requirements-build.txt")

    await ensure_checkout(repo, log_path)
    patch = paths.root / "patches" / C.PATCH_FILE.name
    await fetch_pinned_small_file(session, C.hf_url(C.PATCH_FILE.name, hf_base), C.PATCH_FILE, patch)
    result = await ensure_patch_applied(repo, patch, log_path)
    log.info("FastMTP patch %s", result)

    jobs = effective_build_jobs(cfg.build_jobs)
    log.info("configuring llama.cpp (generator=%s, jobs=%d)", env.get("CMAKE_GENERATOR", "default"), jobs)
    rc = await run_logged(["cmake", "-S", ".", "-B", "build", *C.LLAMA_CMAKE_ARGS], log_path, cwd=repo, env=env,
                          timeout=1800)
    if rc != 0:
        raise StepError(f"cmake configure failed with code {rc} (see /logs?source=llama)")

    def on_line(line: str) -> None:
        pct = parse_build_percent(line)
        if pct is not None:
            state.build_percent = pct

    rc = await run_logged(
        ["cmake", "--build", "build", "--config", "Release", "-j", str(jobs), "--target", C.LLAMA_BUILD_TARGET],
        log_path, cwd=repo, env=env, timeout=4 * 3600, on_line=on_line,
    )
    if rc != 0:
        raise StepError(f"llama.cpp build failed with code {rc} (see /logs?source=llama)")
    await verify_binary(binary, log_path)
    _write_json(marker, build_marker_expected())
    state.build_percent = 100
    return binary


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
        return
    root = dest.resolve()  # pragma: no cover - Python without extraction filters
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(root) + os.sep) and target != root:
            raise StepError(f"unsafe path in prebuilt archive: {member.name}")
        if member.isdev() or ((member.issym() or member.islnk()) and os.path.isabs(member.linkname)):
            raise StepError(f"unsafe member in prebuilt archive: {member.name}")
    tar.extractall(dest)


def install_prebuilt_archive(archive: Path, bin_dir: Path) -> Path:
    """Extract a tarball and move the directory that holds ``llama-server`` into ``bin_dir``."""
    staging = bin_dir.parent / f".prebuilt-staging-{int(time.time())}"
    staging.mkdir(parents=True)
    with tarfile.open(archive, "r:*") as tar:
        _safe_extract(tar, staging)
    found = next((p for p in staging.rglob("llama-server") if p.is_file()), None)
    if found is None:
        raise StepError("prebuilt archive does not contain a llama-server executable")
    bin_dir.mkdir(parents=True, exist_ok=True)
    for item in found.parent.iterdir():
        dest = bin_dir / item.name
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        elif dest.exists() or dest.is_symlink():
            dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(staging, ignore_errors=True)
    server = bin_dir / "llama-server"
    server.chmod(server.stat().st_mode | 0o755)
    return server


async def install_prebuilt(cfg: Config, state: BoxState, session: aiohttp.ClientSession, log_path: Path) -> Path:
    assert cfg.prebuilt_url and cfg.prebuilt_sha256
    paths = cfg.paths
    marker = paths.llama_dir / "build" / ".pb-prebuilt.json"
    binary = paths.llama_server
    if _read_json(marker) == {"sha256": cfg.prebuilt_sha256} and binary.is_file():
        state.build_percent = 100
        return binary
    archive = paths.downloads / "llama-prebuilt.tar.gz"
    log.info("downloading prebuilt llama.cpp from %s", safe_url(cfg.prebuilt_url))

    def on_progress(done: int, total: Optional[int]) -> None:
        if total:
            state.build_percent = min(99, int(done * 100 / total))

    digest = await fetch_url_to_file(session, cfg.prebuilt_url, archive, on_progress)
    if digest != cfg.prebuilt_sha256:
        raise StepError(f"prebuilt llama.cpp sha256 mismatch: expected {cfg.prebuilt_sha256}, got {digest}")
    server = await asyncio.to_thread(install_prebuilt_archive, archive, paths.llama_bin_dir)
    await verify_binary(server, log_path)
    _write_json(marker, {"sha256": cfg.prebuilt_sha256})
    state.build_percent = 100
    return server


async def ensure_llama(
    cfg: Config,
    state: BoxState,
    session: aiohttp.ClientSession,
    hf_base: str = C.HF_BASE,
    on_done: Optional[Callable[[], None]] = None,
) -> Path:
    log_path = cfg.paths.log("llama")
    state.build_state = "running"
    try:
        if cfg.use_prebuilt:
            binary = await install_prebuilt(cfg, state, session, log_path)
        else:
            binary = await build_from_source(cfg, state, session, log_path, hf_base)
    except BaseException:
        state.build_state = "error"
        raise
    state.build_state = "done"
    if on_done:
        on_done()
    return binary
