"""Parallel, resumable, SHA-256 verified downloads.

Each file is fetched with HTTP range requests over several connections into
``<name>.part``. Completed chunks are recorded in ``<name>.part.json`` after the
data has been flushed to disk, so a reboot only re-fetches unfinished chunks.
After the last chunk the file is hashed, renamed to its final name and a
``<name>.verified`` marker containing the hash is written; later boots trust the
marker (plus the exact size) and skip re-hashing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional
from urllib.parse import urljoin, urlsplit

import aiohttp

from .constants import PinnedFile
from .state import FileProgress

CHUNK_SIZE = 64 * 1024 * 1024
READ_SIZE = 1024 * 1024
WRITE_BUFFER = 8 * 1024 * 1024
MAX_CONSECUTIVE_FAILURES = 20
URL_MAX_AGE_S = 30 * 60
CHECKPOINT_INTERVAL_S = 10.0
USER_AGENT = "prompt-box/1.0"

log = logging.getLogger("box.download")
_CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


class DownloadError(Exception):
    """Fatal download problem (shown in /health)."""


class _Retry(Exception):
    """Transient problem; the chunk is retried with backoff."""


# --- small helpers ----------------------------------------------------------------------
def safe_url(url: str) -> str:
    """URL without query string/fragment (signed URLs carry credentials there)."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def marker_path(final: Path) -> Path:
    return final.with_name(final.name + ".verified")


def part_path(final: Path) -> Path:
    return final.with_name(final.name + ".part")


def part_state_path(final: Path) -> Path:
    return final.with_name(final.name + ".part.json")


def read_marker(final: Path) -> Optional[str]:
    try:
        return marker_path(final).read_text().strip().lower()
    except OSError:
        return None


def is_verified(final: Path, spec: PinnedFile) -> bool:
    """Cheap check used on every boot: marker hash matches and the size is exact."""
    try:
        return read_marker(final) == spec.sha256 and final.is_file() and final.stat().st_size == spec.size
    except OSError:
        return False


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_marker(final: Path, sha256: str) -> None:
    atomic_write_text(marker_path(final), sha256 + "\n")


def sha256_file(path: Path, read_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(read_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_content_range(value: Optional[str]) -> Optional[tuple[int, int, Optional[int]]]:
    if not value:
        return None
    m = _CONTENT_RANGE_RE.fullmatch(value.strip())
    if not m:
        return None
    total = None if m.group(3) == "*" else int(m.group(3))
    return int(m.group(1)), int(m.group(2)), total


def chunk_bounds(index: int, size: int, chunk_size: int) -> tuple[int, int]:
    """Inclusive byte range of chunk ``index``."""
    start = index * chunk_size
    return start, min(start + chunk_size, size) - 1


def chunk_count(size: int, chunk_size: int) -> int:
    return (size + chunk_size - 1) // chunk_size


def _sync_data(fd: int) -> None:
    sync = getattr(os, "fdatasync", os.fsync)
    sync(fd)


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        view = view[written:]
        offset += written


def missing_bytes(specs: list[PinnedFile], dest_dir: Path) -> int:
    """Bytes of disk still needed for the given files (sparse .part files count as missing)."""
    from .sysinfo import allocated_bytes

    need = 0
    for spec in specs:
        final = dest_dir / spec.name
        if is_verified(final, spec) or (final.is_file() and final.stat().st_size == spec.size):
            continue
        need += max(0, spec.size - allocated_bytes(part_path(final)))
    return need


# --- persisted chunk state ----------------------------------------------------------------
class PartState:
    VERSION = 1

    def __init__(self, path: Path, spec: PinnedFile, chunk_size: int, done: Optional[set[int]] = None):
        self.path = path
        self.spec = spec
        self.chunk_size = chunk_size
        self.done: set[int] = set(done or ())

    @classmethod
    def load(cls, path: Path, spec: PinnedFile, chunk_size: int) -> "PartState":
        try:
            data = json.loads(path.read_text())
            if (
                data.get("version") == cls.VERSION
                and data.get("size") == spec.size
                and data.get("sha256") == spec.sha256
                and data.get("chunkSize") == chunk_size
            ):
                n = chunk_count(spec.size, chunk_size)
                done = {int(i) for i in data.get("done", []) if 0 <= int(i) < n}
                return cls(path, spec, chunk_size, done)
        except (OSError, ValueError, TypeError):
            pass
        return cls(path, spec, chunk_size)

    def save(self) -> None:
        atomic_write_text(
            self.path,
            json.dumps(
                {
                    "version": self.VERSION,
                    "name": self.spec.name,
                    "size": self.spec.size,
                    "sha256": self.spec.sha256,
                    "chunkSize": self.chunk_size,
                    "done": sorted(self.done),
                }
            ),
        )

    def done_bytes(self) -> int:
        total = 0
        for idx in self.done:
            start, end = chunk_bounds(idx, self.spec.size, self.chunk_size)
            total += end - start + 1
        return total


# --- URL resolution -------------------------------------------------------------------
class UrlResolver:
    """Resolves the Hugging Face ``resolve`` URL to the signed CDN URL and caches it.

    The CDN URL expires (about an hour), so it is refreshed after ``max_age`` seconds
    or when the CDN answers 401/403/404/410.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        url: str,
        spec: PinnedFile,
        max_age: float = URL_MAX_AGE_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._session = session
        self._origin = url
        self._spec = spec
        self._max_age = max_age
        self._clock = clock
        self._url: Optional[str] = None
        self._resolved_at = 0.0
        self._lock = asyncio.Lock()
        self.resolutions = 0

    async def get(self) -> str:
        async with self._lock:
            if self._url is not None and self._clock() - self._resolved_at < self._max_age:
                return self._url
            self._url = await self._resolve()
            self._resolved_at = self._clock()
            self.resolutions += 1
            return self._url

    def invalidate(self) -> None:
        self._url = None

    def _check_linked(self, headers: "aiohttp.typedefs.LooseHeaders") -> None:
        size = headers.get("X-Linked-Size")  # type: ignore[union-attr]
        if size and size.isdigit() and int(size) != self._spec.size:
            raise DownloadError(
                f"{self._spec.name}: server reports {size} bytes, expected {self._spec.size}"
            )
        etag = (headers.get("X-Linked-Etag") or "").strip().strip('"').lower()  # type: ignore[union-attr]
        if re.fullmatch(r"[0-9a-f]{64}", etag) and etag != self._spec.sha256:
            raise DownloadError(f"{self._spec.name}: server reports sha256 {etag}, expected {self._spec.sha256}")

    async def _resolve(self) -> str:
        url = self._origin
        timeout = aiohttp.ClientTimeout(total=60)
        for _ in range(8):
            try:
                async with self._session.head(url, allow_redirects=False, timeout=timeout) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        location = resp.headers.get("Location")
                        if not location:
                            raise _Retry(f"redirect without Location for {self._spec.name}")
                        self._check_linked(resp.headers)
                        nxt = urljoin(url, location)
                        if urlsplit(nxt).netloc != urlsplit(url).netloc:
                            return nxt  # signed CDN URL
                        url = nxt
                        continue
                    if resp.status == 200:
                        self._check_linked(resp.headers)
                        return url
                    if resp.status in (401, 403, 404):
                        raise DownloadError(
                            f"HTTP {resp.status} resolving {self._spec.name} at {safe_url(url)} "
                            "(repository or revision not accessible)"
                        )
                    raise _Retry(f"HTTP {resp.status} resolving {self._spec.name}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise _Retry(f"resolving {self._spec.name}: {exc!r}") from exc
        raise _Retry(f"too many redirects resolving {self._spec.name}")


# --- the downloader -----------------------------------------------------------------------
class FileDownloader:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        spec: PinnedFile,
        url: str,
        dest_dir: Path,
        progress: FileProgress,
        *,
        connections: int = 8,
        chunk_size: int = CHUNK_SIZE,
        max_failures: int = MAX_CONSECUTIVE_FAILURES,
        checkpoint_interval: float = CHECKPOINT_INTERVAL_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        resolver: Optional[UrlResolver] = None,
    ):
        self.session = session
        self.spec = spec
        self.url = url
        self.final = dest_dir / spec.name
        self.progress = progress
        self.connections = max(1, connections)
        self.chunk_size = chunk_size
        self.max_failures = max_failures
        self.checkpoint_interval = checkpoint_interval
        self._sleep = sleep
        self.resolver = resolver or UrlResolver(session, url, spec)
        self._completed: set[int] = set()
        self._ckpt_lock = asyncio.Lock()
        self._state: Optional[PartState] = None
        self.requests = 0

    async def run(self) -> Path:
        spec, final, progress = self.spec, self.final, self.progress
        final.parent.mkdir(parents=True, exist_ok=True)
        if is_verified(final, spec):
            progress.done, progress.verified, progress.phase = spec.size, True, "done"
            return final
        if final.exists():
            # Present but unmarked, e.g. a crash between rename and marker: verify it.
            if final.stat().st_size != spec.size:
                progress.phase = "error"
                raise DownloadError(
                    f"{final} exists with an unexpected size ({final.stat().st_size} != {spec.size}); "
                    "it was left untouched - move it away to download again"
                )
            progress.phase = "verifying"
            digest = await asyncio.to_thread(sha256_file, final)
            if digest != spec.sha256:
                progress.phase = "error"
                raise DownloadError(
                    f"SHA-256 mismatch for existing {final.name}: expected {spec.sha256}, got {digest}; "
                    "the file was left untouched - move it away to download again"
                )
            write_marker(final, digest)
            progress.done, progress.verified, progress.phase = spec.size, True, "done"
            return final

        part = part_path(final)
        state = PartState.load(part_state_path(final), spec, self.chunk_size)
        if not part.exists() or part.stat().st_size != spec.size:
            state.done.clear()
        self._state = state
        fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
        finished = False
        try:
            if os.fstat(fd).st_size != spec.size:
                os.ftruncate(fd, spec.size)
            state.save()
            progress.done = state.done_bytes()
            progress.phase = "downloading"
            pending = [i for i in range(chunk_count(spec.size, self.chunk_size)) if i not in state.done]
            if pending:
                log.info("%s: %d/%d chunks to fetch over %d connections",
                         spec.name, len(pending), chunk_count(spec.size, self.chunk_size), self.connections)
                await self._fetch_all(fd, pending)
            await self._checkpoint(fd)
            finished = True
        finally:
            if not finished:
                self._checkpoint_sync(fd)
            os.close(fd)

        progress.phase = "verifying"
        digest = await asyncio.to_thread(sha256_file, part)
        if digest != spec.sha256:
            progress.phase = "error"
            raise DownloadError(
                f"SHA-256 mismatch for {spec.name}: expected {spec.sha256}, got {digest}. "
                f"The data was kept in {part}; delete it and {part_state_path(final).name} to download again"
            )
        os.replace(part, final)
        write_marker(final, digest)
        try:
            part_state_path(final).unlink()
        except FileNotFoundError:
            pass
        progress.done, progress.verified, progress.phase = spec.size, True, "done"
        log.info("%s: verified sha256 %s", spec.name, digest)
        return final

    async def _fetch_all(self, fd: int, pending: list[int]) -> None:
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in pending:
            queue.put_nowait(idx)

        async def worker() -> None:
            while True:
                try:
                    idx = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                start, end = chunk_bounds(idx, self.spec.size, self.chunk_size)
                await self._fetch_range(fd, start, end)
                self._completed.add(idx)

        async def checkpointer() -> None:
            while True:
                await asyncio.sleep(self.checkpoint_interval)
                await self._checkpoint(fd)

        workers = [asyncio.create_task(worker()) for _ in range(min(self.connections, len(pending)))]
        ckpt = asyncio.create_task(checkpointer())
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers + [ckpt]:
                task.cancel()
            await asyncio.gather(*workers, ckpt, return_exceptions=True)

    async def _fetch_range(self, fd: int, start: int, end: int) -> None:
        pos = start
        failures = 0
        while pos <= end:
            progressed = 0
            try:
                url = await self.resolver.get()
                headers = {"Range": f"bytes={pos}-{end}", "Accept-Encoding": "identity"}
                timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
                self.requests += 1
                async with self.session.get(url, headers=headers, timeout=timeout) as resp:
                    if resp.status in (401, 403, 404, 410):
                        self.resolver.invalidate()
                        raise _Retry(f"HTTP {resp.status} from CDN")
                    if resp.status != 206:
                        raise _Retry(f"expected HTTP 206 for a range request, got {resp.status}")
                    cr = parse_content_range(resp.headers.get("Content-Range"))
                    if cr is None or cr[0] != pos or cr[1] != end or cr[2] not in (None, self.spec.size):
                        raise _Retry(f"unexpected Content-Range {resp.headers.get('Content-Range')!r}")
                    buf = bytearray()
                    async for data in resp.content.iter_chunked(READ_SIZE):
                        if pos + len(buf) + len(data) > end + 1:
                            raise _Retry("server sent more data than requested")
                        buf += data
                        if len(buf) >= WRITE_BUFFER:
                            await asyncio.to_thread(_pwrite_all, fd, bytes(buf), pos)
                            pos += len(buf)
                            progressed += len(buf)
                            self.progress.done += len(buf)
                            buf.clear()
                    if buf:
                        await asyncio.to_thread(_pwrite_all, fd, bytes(buf), pos)
                        pos += len(buf)
                        progressed += len(buf)
                        self.progress.done += len(buf)
                    if pos != end + 1:
                        raise _Retry(f"connection closed early at byte {pos} of {end}")
            except (_Retry, aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
                failures = 1 if progressed >= READ_SIZE else failures + 1
                if failures >= self.max_failures:
                    raise DownloadError(
                        f"{self.spec.name}: giving up on bytes {pos}-{end} after {failures} failed attempts ({exc})"
                    ) from exc
                delay = min(30.0, 2.0 ** min(failures, 5)) * (0.5 + random.random() / 2)
                log.warning("%s: bytes %d-%d failed (%s); retry %d in %.1fs",
                            self.spec.name, pos, end, exc, failures, delay)
                await self._sleep(delay)

    async def _checkpoint(self, fd: int) -> None:
        async with self._ckpt_lock:
            if not self._completed or self._state is None:
                return
            batch = set(self._completed)
            await asyncio.to_thread(_sync_data, fd)
            self._state.done |= batch
            self._completed -= batch
            await asyncio.to_thread(self._state.save)

    def _checkpoint_sync(self, fd: int) -> None:
        """Best effort on cancellation/failure so a restart resumes from here."""
        if not self._completed or self._state is None:
            return
        try:
            _sync_data(fd)
            self._state.done |= self._completed
            self._completed.clear()
            self._state.save()
        except OSError:  # pragma: no cover - disk trouble while shutting down
            log.exception("could not save download state for %s", self.spec.name)


# --- other fetches ------------------------------------------------------------------------
async def fetch_pinned_small_file(session: aiohttp.ClientSession, url: str, spec: PinnedFile, dest: Path) -> Path:
    """Fetch a small pinned file into memory, verify size+sha256 and write it atomically."""
    if dest.is_file() and dest.stat().st_size == spec.size and sha256_file(dest) == spec.sha256:
        return dest
    timeout = aiohttp.ClientTimeout(total=120)
    last: Optional[BaseException] = None
    data = b""
    for attempt in range(5):
        if attempt:
            await asyncio.sleep(2 ** attempt)
        try:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    raise _Retry(f"HTTP {resp.status}")
                buf = bytearray()
                async for block in resp.content.iter_chunked(65536):
                    buf += block
                    if len(buf) > spec.size:
                        break
                data = bytes(buf)
            break
        except (_Retry, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last = exc
    else:
        raise DownloadError(f"could not download {spec.name}: {last}")
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != spec.size or digest != spec.sha256:
        raise DownloadError(
            f"{spec.name}: got {len(data)} bytes with sha256 {digest}, expected {spec.size} bytes with {spec.sha256}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return dest


async def fetch_url_to_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    on_progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> str:
    """Stream ``url`` into ``dest`` (via ``.part``) and return its sha256."""
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    done = 0
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
    async with session.get(url, timeout=timeout) as resp:
        if resp.status != 200:
            raise DownloadError(f"HTTP {resp.status} downloading {safe_url(url)}")
        total = resp.content_length
        with open(tmp, "wb") as fh:
            async for data in resp.content.iter_chunked(READ_SIZE):
                fh.write(data)
                digest.update(data)
                done += len(data)
                if on_progress:
                    on_progress(done, total)
    os.replace(tmp, dest)
    return digest.hexdigest()


def new_download_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=64, ttl_dns_cache=300),
        headers={"User-Agent": USER_AGENT},
        trust_env=True,
    )
