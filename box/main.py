"""Orchestrator entry point: ``python -m box.main``.

Starts the HTTPS gateway first, then downloads the model, builds llama.cpp and
installs ComfyUI in parallel, starts both services and supervises them. It never
shuts the machine down and never deletes downloaded/built files on errors; a
failure puts /health into ``error`` while the gateway keeps serving.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Optional

import aiohttp

from . import __version__
from . import constants as C
from .comfy_install import ensure_comfy
from .config import Config, FatalConfigError, load_config, write_tls_files
from .download import DownloadError, FileDownloader, missing_bytes, new_download_session
from .gateway import context_from_config, make_ssl_context, start_gateway
from .llama_build import ensure_llama
from .logs import Redactor, setup_logging
from .procs import StepError
from .services import Supervisor, comfy_args, kill_stale, llama_server_args
from .state import BoxState, FileProgress
from .sysinfo import git_head, gpu_info

log = logging.getLogger("box")
DISK_MARGIN = 2 * 1024**3


class Orchestrator:
    def __init__(
        self,
        cfg: Config,
        hf_base: str = C.HF_BASE,
        host: str = "0.0.0.0",
        api_port: int = C.API_PORT,
        ui_port: int = C.UI_PORT,
    ):
        self.cfg = cfg
        self.paths = cfg.paths
        self.hf_base = hf_base
        self.host, self.api_port, self.ui_port = host, api_port, ui_port
        text = cfg.text_model
        self.state = BoxState(
            versions={
                "llamaCommit": C.LLAMA_COMMIT,
                "patchSha256": C.PATCH_FILE.sha256,
                "comfy": C.COMFY_TAG,
                "quant": cfg.quant,
                "ctx": cfg.ctx,
                "mtp": cfg.mtp,
                "runtimeCommit": cfg.runtime_commit_env,
            }
        )
        specs = ([text] if text else []) + [C.MMPROJ_FILE, C.FASTMTP_FILE]
        self.state.files = [FileProgress(s.name, s.size) for s in specs]
        self.redactor = Redactor(cfg.secrets())
        self.stop_event = asyncio.Event()
        self.supervisors: dict[str, Supervisor] = {}
        self.http: Optional[aiohttp.ClientSession] = None
        self.local: Optional[aiohttp.ClientSession] = None
        self.pipeline_task: Optional[asyncio.Task] = None
        self.step_tasks: list[asyncio.Task] = []

    # --- helpers -------------------------------------------------------------------------
    def fail(self, where: str, exc: BaseException | str) -> None:
        message = str(exc) or exc.__class__.__name__
        log.error("%s failed: %s", where, message)
        self.state.fail(where, message)

    async def _probe(self, url: str) -> bool:
        assert self.local is not None
        try:
            async with self.local.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                await resp.read()
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    # --- steps ---------------------------------------------------------------------------------
    async def step_download(self) -> bool:
        cfg = self.cfg
        text = cfg.text_model
        assert text is not None and self.http is not None
        specs = [(text, C.TEXT_MODEL_CONNECTIONS), (C.MMPROJ_FILE, C.SIDECAR_CONNECTIONS),
                 (C.FASTMTP_FILE, C.SIDECAR_CONNECTIONS)]
        need = missing_bytes([s for s, _ in specs], self.paths.models)
        free = shutil.disk_usage(self.paths.models).free
        if need and free < need + DISK_MARGIN:
            self.fail("download", f"not enough disk space in {self.paths.models}: need {need / 1e9:.1f} GB "
                                  f"(+{DISK_MARGIN / 1e9:.0f} GB margin), {free / 1e9:.1f} GB free")
            return False
        ok = True

        async def one(spec: C.PinnedFile, connections: int, progress: FileProgress) -> None:
            nonlocal ok
            try:
                await FileDownloader(self.http, spec, C.hf_url(spec.name, self.hf_base), self.paths.models,
                                     progress, connections=connections).run()
            except Exception as exc:
                ok = False
                if not isinstance(exc, (DownloadError, OSError)):
                    log.exception("unexpected download failure")
                self.fail("download", exc)

        await asyncio.gather(*(one(spec, conn, prog) for (spec, conn), prog in zip(specs, self.state.files)))
        return ok

    async def step_build(self) -> Optional[Path]:
        assert self.http is not None
        try:
            return await ensure_llama(self.cfg, self.state, self.http, self.hf_base)
        except Exception as exc:
            if not isinstance(exc, (StepError, DownloadError, OSError, aiohttp.ClientError)):
                log.exception("unexpected build failure")
            self.fail("build", exc)
            return None

    async def step_comfy_install(self) -> Optional[Path]:
        try:
            return await ensure_comfy(self.cfg, self.state)
        except Exception as exc:
            if not isinstance(exc, (StepError, OSError)):
                log.exception("unexpected ComfyUI install failure")
            self.fail("comfy", exc)
            return None

    async def step_start_llama(self, download: "asyncio.Task[bool]", build: "asyncio.Task[Optional[Path]]") -> None:
        downloaded, binary = await download, await build
        if not downloaded or binary is None or self.stop_event.is_set():
            return
        text = self.cfg.text_model
        assert text is not None
        models = self.paths.models
        argv = llama_server_args(binary, models / text.name, models / C.MMPROJ_FILE.name,
                                 models / C.FASTMTP_FILE.name, self.cfg.ctx, self.cfg.mtp)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(binary.parent), env.get("LD_LIBRARY_PATH")]))
        sup = Supervisor(
            "llama", argv, label="llama-server", log_path=self.paths.log("llama"), env=env,
            cwd=binary.parent, pid_path=self.paths.run_dir / "llama.pid",
            health=lambda: self._probe(f"http://{C.LOCAL_HOST}:{C.LLAMA_PORT}/health"),
            on_status=lambda s: self.state.set_service("llama", s),
            on_fatal=lambda m: self.fail("llama_start", m),
            on_note=lambda n: self.state.set_note("llama", n),
            start_timeout=3600, port=C.LLAMA_PORT,
        )
        self.supervisors["llama"] = sup
        await sup.run()

    async def _check_comfy_nodes(self) -> None:
        assert self.local is not None
        url = f"http://{C.LOCAL_HOST}:{C.COMFY_PORT}/object_info"
        info: dict = {}
        for attempt in range(3):
            try:
                async with self.local.get(url, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                    if resp.status != 200:
                        raise StepError(f"ComfyUI /object_info returned HTTP {resp.status}")
                    info = await resp.json(content_type=None)
                break
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                if attempt == 2:
                    raise StepError(f"ComfyUI /object_info failed: {exc!r}") from exc
                await asyncio.sleep(5)
        missing = [n for n in (C.CUSTOM_NODE_CLASS, C.CORE_PREVIEW_NODE) if n not in info]
        if missing:
            raise StepError(f"ComfyUI is running but node(s) {', '.join(missing)} are not registered "
                            "(see /logs?source=comfy)")

    async def step_start_comfy(self, install: "asyncio.Task[Optional[Path]]") -> None:
        python = await install
        if python is None or self.stop_event.is_set():
            return
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        sup = Supervisor(
            "comfy", comfy_args(python), label="ComfyUI", log_path=self.paths.log("comfy"), env=env,
            cwd=self.paths.comfy_dir, pid_path=self.paths.run_dir / "comfy.pid",
            health=lambda: self._probe(f"http://{C.LOCAL_HOST}:{C.COMFY_PORT}/system_stats"),
            post_ready=self._check_comfy_nodes,
            on_status=lambda s: self.state.set_service("comfy", s),
            on_fatal=lambda m: self.fail("comfy_start", m),
            on_note=lambda n: self.state.set_note("comfy", n),
            start_timeout=1800, port=C.COMFY_PORT,
        )
        self.supervisors["comfy"] = sup
        await sup.run()

    async def pipeline(self) -> None:
        if self.state.versions.get("runtimeCommit") is None:
            self.state.versions["runtimeCommit"] = await git_head(self.paths.runtime_dir)
        await asyncio.gather(
            asyncio.to_thread(kill_stale, self.paths.run_dir / "llama.pid", "llama-server"),
            asyncio.to_thread(kill_stale, self.paths.run_dir / "comfy.pid", f"--port {C.COMFY_PORT}"),
        )
        download = asyncio.create_task(self.step_download())
        build = asyncio.create_task(self.step_build())
        install = asyncio.create_task(self.step_comfy_install())
        self.step_tasks = [download, build, install]
        await asyncio.gather(self.step_start_llama(download, build), self.step_start_comfy(install))
        if self.state.versions.get("runtimeCommit") is None:
            self.state.versions["runtimeCommit"] = await git_head(self.paths.runtime_dir)

    # --- lifecycle ------------------------------------------------------------------------------
    async def run(self) -> int:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.stop_event.set)
            except (NotImplementedError, RuntimeError):  # pragma: no cover
                pass
        cert, key = write_tls_files(self.cfg)
        ssl_context = make_ssl_context(cert, key)
        gateway_ctx = context_from_config(self.cfg, self.state, self.redactor)
        runners = await start_gateway(gateway_ctx, ssl_context, self.host, self.api_port, self.ui_port)
        log.info("prompt-box %s gateway listening on :%d (API) and :%d (ComfyUI)", __version__, self.api_port,
                 self.ui_port)
        for warning in self.cfg.warnings:
            log.warning("%s", warning)
        self.http = new_download_session()
        self.local = aiohttp.ClientSession(trust_env=False)
        gpu_task = asyncio.create_task(self._capture_gpu())
        try:
            if self.cfg.errors:
                for err in self.cfg.errors:
                    self.fail("config", err)
            else:
                self.pipeline_task = asyncio.create_task(self.pipeline())
                self.pipeline_task.add_done_callback(self._pipeline_done)
            await self.stop_event.wait()
            log.info("stopping")
        finally:
            gpu_task.cancel()
            pending = [t for t in [self.pipeline_task, *self.step_tasks] if t is not None]
            for task in pending:
                task.cancel()
            await asyncio.gather(*(s.stop() for s in self.supervisors.values()), return_exceptions=True)
            await asyncio.gather(*pending, return_exceptions=True)
            for runner in runners:
                await runner.cleanup()
            await self.http.close()
            await self.local.close()
        return 0

    async def _capture_gpu(self) -> None:
        try:
            self.state.gpu = await gpu_info()
            log.info("GPU: %s", self.state.gpu)
        except Exception:  # GPU info is informational only
            log.exception("could not read GPU info")

    def _pipeline_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("pipeline crashed", exc_info=exc)
            self.fail("internal", f"unexpected {exc.__class__.__name__}: {exc}")


def acquire_lock(run_dir: Path) -> Optional[int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(run_dir / "box.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def main() -> int:
    try:
        cfg = load_config(os.environ)
    except FatalConfigError as exc:
        print(f"prompt-box: refusing to start: {exc}", file=sys.stderr, flush=True)
        return 2
    cfg.paths.ensure()
    lock_fd = acquire_lock(cfg.paths.run_dir)
    if lock_fd is None:
        print("prompt-box: another orchestrator is already running (run/box.lock is held)", file=sys.stderr, flush=True)
        return 1
    setup_logging(cfg.paths.log("box"), Redactor(cfg.secrets()))
    log.info("prompt-box %s starting (quant=%s ctx=%d mtp=%s prebuilt=%s)",
             __version__, cfg.quant, cfg.ctx, cfg.mtp, cfg.use_prebuilt)
    try:
        return asyncio.run(Orchestrator(cfg).run())
    except Exception:
        log.exception("orchestrator crashed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
