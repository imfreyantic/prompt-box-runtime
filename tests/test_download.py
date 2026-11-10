import asyncio
import hashlib
import json
import os
import random
import re
import unittest
from pathlib import Path
from unittest import mock

import aiohttp
from aiohttp import web

from box import download as D
from box.constants import PinnedFile, hf_url
from box.state import FileProgress
from tests.helpers import RunningApp, tempdir

CHUNK = 64 * 1024


async def no_sleep(_delay: float) -> None:
    return None


class FakeHub:
    """Hugging Face 'resolve' endpoint + a separate CDN host supporting Range requests."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.sig = 1
        self.head_requests = 0
        self.cdn_requests = 0
        self.ranges: list[tuple[str, int, int]] = []
        self.fail_next = 0
        self.ignore_range = False
        self.truncate_next = 0
        self.linked_size: dict[str, int] = {}
        self.fail_after = None
        self.cdn_url = ""

    def hub_app(self) -> web.Application:
        app = web.Application()

        async def resolve(request: web.Request) -> web.StreamResponse:
            name = request.match_info["name"]
            self.head_requests += 1
            if name not in self.files:
                return web.Response(status=404)
            return web.Response(status=302, headers={
                "Location": f"{self.cdn_url}/cdn/{name}?sig={self.sig}",
                "X-Linked-Size": str(self.linked_size.get(name, len(self.files[name]))),
                "X-Linked-Etag": '"' + hashlib.sha256(self.files[name]).hexdigest() + '"',
            })

        async def small_get(request: web.Request) -> web.StreamResponse:
            name = request.match_info["name"]
            if request.path.startswith("/api/resolve-cache/"):
                return web.Response(body=self.files[name])
            return web.Response(status=307, headers={"Location": f"/api/resolve-cache/x/{name}"})

        app.router.add_route("HEAD", "/{repo:.+}/resolve/{rev}/{name}", resolve)
        app.router.add_get("/api/resolve-cache/x/{name}", small_get)
        app.router.add_get("/{repo:.+}/resolve/{rev}/{name}", small_get)
        return app

    def cdn_app(self) -> web.Application:
        app = web.Application()

        async def cdn(request: web.Request) -> web.StreamResponse:
            name = request.match_info["name"]
            self.cdn_requests += 1
            if request.query.get("sig") != str(self.sig):
                return web.Response(status=403)
            if self.fail_next > 0:
                self.fail_next -= 1
                return web.Response(status=500)
            if self.fail_after is not None and len(self.ranges) >= self.fail_after:
                return web.Response(status=500)
            data = self.files[name]
            if self.ignore_range or "Range" not in request.headers:
                return web.Response(body=data)
            m = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"])
            start, end = int(m.group(1)), int(m.group(2))
            self.ranges.append((name, start, end))
            body = data[start:end + 1]
            headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
            if self.truncate_next > 0:
                self.truncate_next -= 1
                resp = web.StreamResponse(status=206, headers=headers)
                resp.content_length = len(body)
                await resp.prepare(request)
                await resp.write(body[: len(body) // 2])
                await asyncio.sleep(0.05)
                request.transport.close()
                return resp
            return web.Response(status=206, body=body, headers=headers)

        app.router.add_get("/cdn/{name}", cdn)
        return app


def make_blob(size: int, seed: int = 7) -> bytes:
    return random.Random(seed).randbytes(size)


def spec_for(name: str, data: bytes) -> PinnedFile:
    return PinnedFile(name, len(data), hashlib.sha256(data).hexdigest())


class DownloadTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempdir()
        self.dest = Path(self._tmp.name)
        self.data = make_blob(5 * CHUNK + 1234)
        self.spec = spec_for("model.gguf", self.data)
        self.hub = FakeHub({self.spec.name: self.data, "small.patch": b"diff --git a/x b/x\n"})
        self.hub_srv = await RunningApp(self.hub.hub_app()).__aenter__()
        self.cdn_srv = await RunningApp(self.hub.cdn_app()).__aenter__()
        self.hub.cdn_url = self.cdn_srv.url
        self.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        await self.cdn_srv.__aexit__(None, None, None)
        await self.hub_srv.__aexit__(None, None, None)
        self._tmp.cleanup()

    def downloader(self, spec=None, **kwargs) -> D.FileDownloader:
        spec = spec or self.spec
        kwargs.setdefault("connections", 3)
        kwargs.setdefault("chunk_size", CHUNK)
        kwargs.setdefault("sleep", no_sleep)
        self.progress = FileProgress(spec.name, spec.size)
        return D.FileDownloader(self.session, spec, hf_url(spec.name, self.hub_srv.url), self.dest, self.progress,
                                **kwargs)


class FreshDownloadTests(DownloadTestCase):
    async def test_parallel_ranges_assemble_verified_file(self):
        dl = self.downloader()
        final = await dl.run()
        self.assertEqual(final.read_bytes(), self.data)
        self.assertEqual(D.read_marker(final), self.spec.sha256)
        self.assertTrue(D.is_verified(final, self.spec))
        self.assertFalse(D.part_path(final).exists())
        self.assertFalse(D.part_state_path(final).exists())
        self.assertTrue(self.progress.verified)
        self.assertEqual(self.progress.as_json()["doneBytes"], self.spec.size)
        expected = [D.chunk_bounds(i, self.spec.size, CHUNK) for i in range(D.chunk_count(self.spec.size, CHUNK))]
        self.assertEqual(sorted((s, e) for _, s, e in self.hub.ranges), expected)
        self.assertEqual(self.hub.head_requests, 1)  # CDN URL cached

    async def test_verified_marker_skips_network_and_rehash(self):
        await self.downloader().run()
        requests = self.hub.cdn_requests
        with mock.patch.object(D, "sha256_file", side_effect=AssertionError("must not re-hash")):
            final = await self.downloader().run()
        self.assertEqual(final.read_bytes(), self.data)
        self.assertEqual(self.hub.cdn_requests, requests)
        self.assertTrue(self.progress.verified)

    async def test_marker_with_wrong_size_is_not_trusted(self):
        final = self.dest / self.spec.name
        final.write_bytes(self.data[:-1])
        D.write_marker(final, self.spec.sha256)
        self.assertFalse(D.is_verified(final, self.spec))
        with self.assertRaises(D.DownloadError):
            await self.downloader().run()
        self.assertEqual(final.read_bytes(), self.data[:-1])  # never deleted

    async def test_unmarked_final_file_is_hashed_once(self):
        final = self.dest / self.spec.name
        final.write_bytes(self.data)
        await self.downloader().run()
        self.assertEqual(self.hub.cdn_requests, 0)
        self.assertEqual(D.read_marker(final), self.spec.sha256)


class ResumeTests(DownloadTestCase):
    async def test_resume_fetches_only_missing_chunks(self):
        final = self.dest / self.spec.name
        part = D.part_path(final)
        garbage = bytearray(os.urandom(self.spec.size))
        done = {0, 2, 3}
        for i in done:
            s, e = D.chunk_bounds(i, self.spec.size, CHUNK)
            garbage[s:e + 1] = self.data[s:e + 1]
        part.write_bytes(bytes(garbage))
        state = D.PartState(D.part_state_path(final), self.spec, CHUNK, done)
        state.save()

        dl = self.downloader()
        await dl.run()
        self.assertEqual(final.read_bytes(), self.data)
        fetched = sorted({s for _, s, _ in self.hub.ranges})
        expected = [D.chunk_bounds(i, self.spec.size, CHUNK)[0] for i in range(6) if i not in done]
        self.assertEqual(fetched, expected)

    async def test_state_for_other_chunk_size_is_ignored(self):
        final = self.dest / self.spec.name
        D.part_path(final).write_bytes(b"\0" * self.spec.size)
        D.PartState(D.part_state_path(final), self.spec, CHUNK * 2, {0, 1}).save()
        loaded = D.PartState.load(D.part_state_path(final), self.spec, CHUNK)
        self.assertEqual(loaded.done, set())
        await self.downloader().run()
        self.assertEqual(final.read_bytes(), self.data)
        self.assertEqual(len(self.hub.ranges), 6)

    async def test_completed_chunks_survive_a_failure(self):
        self.hub.fail_after = 2  # CDN breaks after two chunks
        with self.assertRaises(D.DownloadError):
            await self.downloader(connections=1, max_failures=2).run()
        final = self.dest / self.spec.name
        self.assertTrue(D.part_path(final).exists())  # partial data kept
        self.assertFalse(final.exists())
        state = json.loads(D.part_state_path(final).read_text())
        self.assertEqual(state["done"], [0, 1])
        self.hub.fail_after = None
        self.hub.ranges.clear()
        await self.downloader().run()
        self.assertEqual(final.read_bytes(), self.data)
        self.assertEqual(sorted(s for _, s, _ in self.hub.ranges), [i * CHUNK for i in range(2, 6)])

    async def test_truncated_response_resumes_inside_the_chunk(self):
        self.hub.truncate_next = 1
        with mock.patch.object(D, "WRITE_BUFFER", 8 * 1024):
            dl = self.downloader(connections=1)
            await dl.run()
        self.assertEqual((self.dest / self.spec.name).read_bytes(), self.data)
        starts = [s for _, s, _ in self.hub.ranges]
        # the first chunk was requested twice: from 0, then from where the cut-off left it
        self.assertEqual(starts[0], 0)
        self.assertTrue(0 < starts[1] < CHUNK, starts[:3])


class FailureTests(DownloadTestCase):
    async def test_transient_errors_are_retried(self):
        self.hub.fail_next = 4
        await self.downloader().run()
        self.assertTrue(self.progress.verified)

    async def test_expired_cdn_url_is_re_resolved(self):
        dl = self.downloader(connections=1)
        await dl.resolver.get()
        self.hub.sig = 2  # previously signed URL now answers 403
        await dl.run()
        self.assertGreaterEqual(dl.resolver.resolutions, 2)
        self.assertEqual((self.dest / self.spec.name).read_bytes(), self.data)

    async def test_server_ignoring_range_fails_after_retries(self):
        self.hub.ignore_range = True
        with self.assertRaises(D.DownloadError):
            await self.downloader(max_failures=3).run()

    async def test_hash_mismatch_keeps_data_and_reports(self):
        bad_spec = PinnedFile(self.spec.name, self.spec.size, "0" * 64)
        self.hub.files[self.spec.name] = self.data  # etag from server is the real hash -> mismatch detected early
        with self.assertRaises(D.DownloadError) as cm:
            await self.downloader(spec=bad_spec).run()
        self.assertIn("sha256", str(cm.exception).lower())

    async def test_sha_mismatch_after_download_keeps_part(self):
        bad_spec = PinnedFile(self.spec.name, self.spec.size, "f" * 64)
        with mock.patch.object(D.UrlResolver, "_check_linked", lambda self, headers: None):
            with self.assertRaises(D.DownloadError) as cm:
                await self.downloader(spec=bad_spec).run()
        self.assertIn("SHA-256 mismatch", str(cm.exception))
        final = self.dest / self.spec.name
        self.assertTrue(D.part_path(final).exists())
        self.assertFalse(final.exists())
        self.assertEqual(self.progress.phase, "error")

    async def test_linked_size_mismatch_is_fatal(self):
        self.hub.linked_size[self.spec.name] = self.spec.size + 1
        with self.assertRaises(D.DownloadError):
            await self.downloader().run()
        self.assertEqual(self.hub.cdn_requests, 0)

    async def test_missing_file_is_fatal(self):
        spec = PinnedFile("nope.gguf", 10, "0" * 64)
        with self.assertRaises(D.DownloadError):
            await self.downloader(spec=spec).run()


class SmallFileTests(DownloadTestCase):
    async def test_pinned_small_file_follows_relative_redirect(self):
        data = self.hub.files["small.patch"]
        spec = PinnedFile("small.patch", len(data), hashlib.sha256(data).hexdigest())
        dest = self.dest / "patches" / "small.patch"
        await D.fetch_pinned_small_file(self.session, hf_url(spec.name, self.hub_srv.url), spec, dest)
        self.assertEqual(dest.read_bytes(), data)

    async def test_pinned_small_file_rejects_wrong_hash(self):
        data = self.hub.files["small.patch"]
        spec = PinnedFile("small.patch", len(data), "1" * 64)
        with self.assertRaises(D.DownloadError):
            await D.fetch_pinned_small_file(self.session, hf_url(spec.name, self.hub_srv.url), spec,
                                            self.dest / "x.patch")
        self.assertFalse((self.dest / "x.patch").exists())


class HelperTests(unittest.TestCase):
    def test_parse_content_range(self):
        self.assertEqual(D.parse_content_range("bytes 0-9/100"), (0, 9, 100))
        self.assertEqual(D.parse_content_range("bytes 5-9/*"), (5, 9, None))
        self.assertIsNone(D.parse_content_range("items 0-9/100"))
        self.assertIsNone(D.parse_content_range(None))

    def test_chunking(self):
        self.assertEqual(D.chunk_count(10, 4), 3)
        self.assertEqual([D.chunk_bounds(i, 10, 4) for i in range(3)], [(0, 3), (4, 7), (8, 9)])

    def test_safe_url_drops_signature(self):
        self.assertEqual(D.safe_url("https://cdn.example/x/y.gguf?Signature=abc&Expires=1"),
                         "https://cdn.example/x/y.gguf")

    def test_missing_bytes(self):
        with tempdir() as tmp:
            dest = Path(tmp)
            a = PinnedFile("a.bin", 1000, "0" * 64)
            b = PinnedFile("b.bin", 500, hashlib.sha256(b"x" * 500).hexdigest())
            (dest / "b.bin").write_bytes(b"x" * 500)
            D.write_marker(dest / "b.bin", b.sha256)
            self.assertEqual(D.missing_bytes([a, b], dest), 1000)

    def test_part_state_roundtrip(self):
        with tempdir() as tmp:
            spec = PinnedFile("m", 10 * CHUNK, "a" * 64)
            path = Path(tmp) / "m.part.json"
            D.PartState(path, spec, CHUNK, {1, 3, 99}).save()
            loaded = D.PartState.load(path, spec, CHUNK)
            self.assertEqual(loaded.done, {1, 3})  # out-of-range index dropped
            self.assertEqual(loaded.done_bytes(), 2 * CHUNK)
            self.assertEqual(json.loads(path.read_text())["sha256"], "a" * 64)
            other = PinnedFile("m", 10 * CHUNK, "b" * 64)
            self.assertEqual(D.PartState.load(path, other, CHUNK).done, set())


if __name__ == "__main__":
    unittest.main()
