"""ComfyUI checkout, virtualenv, custom node and workflow installation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from . import constants as C
from .config import Config
from .procs import StepError, run_logged
from .state import BoxState
from .sysinfo import run_capture
from .tools import GIT, ensure_commands, ensure_venv, git

log = logging.getLogger("box.comfy")

_TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
_CONSTRAINT_SCRIPT = (
    "import importlib.metadata as m\n"
    f"for p in {_TORCH_PACKAGES!r}:\n"
    "    try:\n"
    "        print(p + '==' + m.version(p))\n"
    "    except m.PackageNotFoundError:\n"
    "        pass\n"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> Optional[str]:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


async def ensure_checkout(comfy: Path, log_path: Path) -> None:
    """git clone --depth 1 --branch v0.44.0, verify HEAD; repairs an interrupted clone."""
    if not (comfy / ".git").exists():
        if comfy.exists() and any(comfy.iterdir()):
            raise StepError(f"{comfy} exists but is not a git checkout; refusing to modify it")
        rc = await git(["clone", "--depth", "1", "--branch", C.COMFY_TAG, C.COMFY_REPO_URL, str(comfy)], log_path)
        if rc != 0:
            raise StepError(f"git clone of ComfyUI {C.COMFY_TAG} failed (code {rc})")
    rc, head = await run_capture([*GIT, "-C", str(comfy), "rev-parse", "HEAD"])
    if rc == 0 and head.strip() == C.COMFY_COMMIT:
        return
    unborn = rc != 0
    tag_ref = f"refs/tags/{C.COMFY_TAG}"
    if await git(["-C", str(comfy), "fetch", "--depth", "1", C.COMFY_REPO_URL, f"{tag_ref}:{tag_ref}"], log_path) != 0:
        raise StepError(f"could not fetch ComfyUI {C.COMFY_TAG}")
    checkout = ["-C", str(comfy), "checkout", "--detach"] + (["--force"] if unborn else []) + [C.COMFY_COMMIT]
    if await git(checkout, log_path) != 0:
        raise StepError(f"could not check out ComfyUI {C.COMFY_COMMIT}")
    rc, head = await run_capture([*GIT, "-C", str(comfy), "rev-parse", "HEAD"])
    if rc != 0 or head.strip() != C.COMFY_COMMIT:
        raise StepError(f"ComfyUI HEAD is {head.strip()!r}, expected {C.COMFY_COMMIT} ({C.COMFY_TAG})")


async def ensure_requirements(cfg: Config, python: Path, log_path: Path) -> None:
    comfy = cfg.paths.comfy_dir
    req = comfy / "requirements.txt"
    want = {"requirementsSha256": _sha256_path(req), "comfyCommit": C.COMFY_COMMIT}
    marker = cfg.paths.venv_comfy / ".pb-requirements.json"
    try:
        if json.loads(marker.read_text()) == want:
            return
    except (OSError, ValueError):
        pass
    # Pin the image's torch so pip reuses it instead of replacing it.
    rc, out = await run_capture([cfg.base_python, "-c", _CONSTRAINT_SCRIPT], timeout=120)
    constraints = cfg.paths.state_dir / "comfy-constraints.txt"
    lines = [line.strip() for line in out.splitlines() if rc == 0 and "==" in line]
    constraints.write_text("\n".join(lines) + ("\n" if lines else ""))
    log.info("installing ComfyUI requirements (constraints: %s)", ", ".join(lines) or "none")
    rc = await run_logged(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
         "--progress-bar", "off", "-r", str(req), "-c", str(constraints)],
        log_path, cwd=comfy, timeout=3600,
    )
    if rc != 0:
        raise StepError(f"pip install -r ComfyUI/requirements.txt failed with code {rc} (see /logs?source=comfy)")
    rc = await run_logged(
        [str(python), "-c", "import torch, aiohttp, yaml, PIL, numpy; "
                            "print('torch', torch.__version__, 'cuda', torch.version.cuda)"],
        log_path, timeout=300,
    )
    if rc != 0:
        raise StepError("ComfyUI venv cannot import torch/aiohttp/yaml/PIL/numpy (see /logs?source=comfy)")
    marker.write_text(json.dumps(want))


def install_custom_node(runtime_dir: Path, comfy: Path) -> Path:
    src = runtime_dir / "custom_nodes" / C.CUSTOM_NODE_PACKAGE
    dest = comfy / "custom_nodes" / C.CUSTOM_NODE_PACKAGE
    if not (src / "__init__.py").is_file():
        raise StepError(f"custom node source missing at {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest


def install_workflows(runtime_dir: Path, comfy: Path, record_path: Path) -> dict[str, str]:
    """Copy UI workflows into ComfyUI's user workflow folder.

    A workflow the user edited in place (hash differs from what we installed last
    time) is left alone. Returns {filename: action}.
    """
    src_dir = runtime_dir / "workflows"
    dest_dir = comfy / "user" / "default" / "workflows"
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        record: dict[str, str] = json.loads(record_path.read_text())
    except (OSError, ValueError):
        record = {}
    actions: dict[str, str] = {}
    for src in sorted(src_dir.glob("*.json")):
        data = src.read_bytes()
        new_hash = _sha256_bytes(data)
        dest = dest_dir / src.name
        current = _sha256_path(dest)
        if current is None or current == record.get(src.name):
            if current != new_hash:
                tmp = dest.with_name(dest.name + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                actions[src.name] = "installed" if current is None else "updated"
            else:
                actions[src.name] = "unchanged"
            record[src.name] = new_hash
        elif current == new_hash:
            record[src.name] = new_hash
            actions[src.name] = "unchanged"
        else:
            actions[src.name] = "kept-user-edit"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return actions


async def ensure_comfy(cfg: Config, state: BoxState) -> Path:
    """Returns the venv python used to run ComfyUI."""
    log_path = cfg.paths.log("comfy")
    state.comfy_state = "running"
    try:
        missing = await ensure_commands(["git"], cfg.paths.log("box"))
        if missing:
            raise StepError("git is missing and could not be installed with apt-get")
        await ensure_checkout(cfg.paths.comfy_dir, log_path)
        python = await ensure_venv(cfg.paths.venv_comfy, cfg.base_python, log_path, system_site_packages=True)
        await ensure_requirements(cfg, python, log_path)
        node_dir = install_custom_node(cfg.paths.runtime_dir, cfg.paths.comfy_dir)
        actions = install_workflows(
            cfg.paths.runtime_dir, cfg.paths.comfy_dir, cfg.paths.state_dir / "workflows-installed.json"
        )
        log.info("custom node installed at %s; workflows: %s", node_dir, actions)
    except BaseException:
        state.comfy_state = "error"
        raise
    state.comfy_state = "done"
    return python
