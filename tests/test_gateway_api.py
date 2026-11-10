import asyncio
import hmac
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import aiohttp
from aiohttp import web

from box import gateway as G
from box.state import FileProgress
from tests.helpers import TOKEN, RunningApp, auth, make_ctx, tempdir

HEALTH_KEYS = {"state", "stage", "message", "error", "startedAt", "readyAt", "progress", "services", "versions", "gpu"}


class FakeLlama:
    """Records requests and serves /v1/models, JSON and SSE completions."""

    def __init__(self):
        self.requests: list[dict] = []
        self.gate = asyncio.Event()
        self.stream_closed = asyncio.Event()

    def app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)

        async def models(request: web.Request) -> web.Response:
            self.requests.append({"path": request.path_qs, "headers": dict(request.headers)})
            return web.json_response({"object": "list", "data": [{"id": "qwen3.8-27b-aggressive"}]})

        async def chat(request: web.Request) -> web.StreamResponse:
            body = await request.read()
            payload = json.loads(body)
            self.requests.append({"path": request.path_qs, "headers": dict(request.headers), "size": len(body),
                                  "payload": payload})
            if payload.get("mode") == "slow":
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                try:
                    for _ in range(400):
                        await resp.write(b"data: {\"x\":1}\n\n")
                        await asyncio.sleep(0.02)
                except (ConnectionResetError, asyncio.CancelledError):
                    self.stream_closed.set()
                    raise
                return resp
            if payload.get("stream"):
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                await resp.write(b"data: one\n\n")
                await self.gate.wait()
                await resp.write(b"data: two\n\n")
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
                return resp
            return web.json_response({"choices": [{"message": {"content": "hi"}}]})

        app.router.add_get("/v1/models", models)
        app.router.add_post("/v1/chat/completions", chat)
        return app


class ApiTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempdir()
        self.tmp = Path(self._tmp.name)
        self.llama = FakeLlama()
        self.llama_srv = await RunningApp(self.llama.app()).__aenter__()
        self.ctx = make_ctx(self.tmp, llama_base=self.llama_srv.url)
        self.ctx.state.services["llama"] = "ready"
        self.api = await RunningApp(G.create_api_app(self.ctx)).__aenter__()
        self.client = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        await self.api.__aexit__(None, None, None)
        await self.llama_srv.__aexit__(None, None, None)
        self._tmp.cleanup()


class AuthTests(ApiTestCase):
    async def test_missing_or_wrong_token_is_401_everywhere(self):
        cases = [
            ("GET", "/health", {}),
            ("GET", "/health", {"Authorization": "Bearer " + "f" * 64}),
            ("GET", "/health", {"Authorization": TOKEN}),
            ("GET", "/health", {"Authorization": "Basic " + TOKEN}),
            ("GET", "/health", {"Authorization": "Bearer " + TOKEN[:-1]}),
            ("GET", "/logs", {}),
            ("GET", "/v1/models", {}),
            ("POST", "/v1/chat/completions", {}),
            ("GET", "/does-not-exist", {}),
        ]
        for method, path, headers in cases:
            with self.subTest(method=method, path=path, headers=headers):
                async with self.client.request(method, self.api.url + path, headers=headers) as resp:
                    self.assertEqual(resp.status, 401)
                    self.assertEqual(await resp.json(), {"error": "unauthorized"})
        self.assertEqual(self.llama.requests, [])

    async def test_valid_token_and_case_insensitive_scheme(self):
        for header in (f"Bearer {TOKEN}", f"bearer {TOKEN}"):
            async with self.client.get(self.api.url + "/health", headers={"Authorization": header}) as resp:
                self.assertEqual(resp.status, 200)

    async def test_constant_time_compare_is_used(self):
        with mock.patch.object(G.hmac, "compare_digest", wraps=hmac.compare_digest) as spy:
            async with self.client.get(self.api.url + "/health", headers=auth()) as resp:
                self.assertEqual(resp.status, 200)
            async with self.client.get(self.api.url + "/health", headers=auth("e" * 64)) as resp:
                self.assertEqual(resp.status, 401)
        self.assertGreaterEqual(spy.call_count, 2)

    async def test_unknown_path_with_token_is_json_404(self):
        async with self.client.get(self.api.url + "/nope", headers=auth()) as resp:
            self.assertEqual(resp.status, 404)
            self.assertEqual(await resp.json(), {"error": "not_found"})


class HealthTests(ApiTestCase):
    async def test_shape(self):
        state = self.ctx.state
        state.files = [FileProgress("a.gguf", 100, done=40), FileProgress("b.gguf", 50, done=50, verified=True)]
        state.gpu = {"name": "NVIDIA RTX PRO 6000", "vramMiB": 97887, "powerLimitW": 600.0, "powerMaxW": 600.0,
                     "driver": "580", "cuda": "13.0"}
        async with self.client.get(self.api.url + "/health", headers=auth()) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers["Cache-Control"], "no-store")
            body = await resp.json()
        self.assertEqual(set(body), HEALTH_KEYS)
        self.assertIn(body["state"], ("installing", "ready", "error"))
        self.assertIn(body["stage"], ("download", "build", "comfy", "llama_start", "comfy_start", "ready", "error"))
        self.assertIsInstance(body["message"], str)
        self.assertIsNone(body["error"])
        self.assertIsInstance(body["startedAt"], int)
        self.assertIsNone(body["readyAt"])
        dl = body["progress"]["download"]
        self.assertEqual((dl["doneBytes"], dl["totalBytes"]), (90, 150))
        self.assertEqual(dl["files"][0], {"name": "a.gguf", "doneBytes": 40, "totalBytes": 100, "verified": False})
        self.assertEqual(dl["files"][1]["verified"], True)
        self.assertEqual(body["progress"]["build"], {"state": "pending", "percent": None})
        self.assertEqual(body["progress"]["comfy"], {"state": "pending"})
        self.assertEqual(set(body["services"]), {"llama", "comfy"})
        self.assertEqual(set(body["versions"]),
                         {"llamaCommit", "patchSha256", "comfy", "quant", "ctx", "mtp", "runtimeCommit"})
        self.assertIsInstance(body["versions"]["ctx"], int)
        self.assertEqual(body["gpu"]["vramMiB"], 97887)

    async def test_error_state(self):
        self.ctx.state.fail("build", "cmake failed")
        async with self.client.get(self.api.url + "/health", headers=auth()) as resp:
            body = await resp.json()
        self.assertEqual((body["state"], body["stage"]), ("error", "error"))
        self.assertEqual(body["error"], "build: cmake failed")
        self.assertEqual(body["message"], "build: cmake failed")


class LogsTests(ApiTestCase):
    def write_log(self, source: str, lines: list[str]) -> None:
        self.ctx.paths.log(source).write_text("\n".join(lines) + "\n")

    async def get_logs(self, query: str):
        async with self.client.get(self.api.url + "/logs" + query, headers=auth()) as resp:
            return resp.status, await resp.json()

    async def test_default_source_and_lines(self):
        self.write_log("box", [f"line {i}" for i in range(500)])
        status, body = await self.get_logs("")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["lines"]), 200)
        self.assertEqual(body["lines"][-1], "line 499")
        self.assertEqual(body["lines"][0], "line 300")

    async def test_sources_and_limits(self):
        for source in ("box", "llama", "comfy", "bootstrap"):
            self.write_log(source, [f"{source} {i}" for i in range(1500)])
            status, body = await self.get_logs(f"?source={source}&lines=5")
            self.assertEqual(status, 200)
            self.assertEqual(body["lines"], [f"{source} {i}" for i in range(1495, 1500)])
        status, body = await self.get_logs("?source=llama&lines=5000")
        self.assertEqual(len(body["lines"]), 1000)
        status, body = await self.get_logs("?source=comfy&lines=0")
        self.assertEqual(len(body["lines"]), 1)

    async def test_bad_parameters(self):
        status, body = await self.get_logs("?source=../../etc/passwd")
        self.assertEqual((status, body["error"]), (400, "invalid_source"))
        status, body = await self.get_logs("?lines=abc")
        self.assertEqual((status, body["error"]), (400, "invalid_lines"))

    async def test_missing_log_is_empty(self):
        status, body = await self.get_logs("?source=bootstrap")
        self.assertEqual((status, body), (200, {"lines": []}))

    async def test_secrets_are_redacted(self):
        self.write_log("llama", [f"token={TOKEN}", "Authorization: Bearer abcdefghijklmnop", "ok line"])
        status, body = await self.get_logs("?source=llama&lines=10")
        joined = "\n".join(body["lines"])
        self.assertNotIn(TOKEN, joined)
        self.assertNotIn("abcdefghijklmnop", joined)
        self.assertIn("ok line", joined)


class ProxyTests(ApiTestCase):
    async def test_models_proxied_without_client_authorization(self):
        async with self.client.get(self.api.url + "/v1/models?x=1", headers=auth()) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual((await resp.json())["data"][0]["id"], "qwen3.8-27b-aggressive")
        seen = self.llama.requests[-1]
        self.assertEqual(seen["path"], "/v1/models?x=1")
        self.assertNotIn("Authorization", seen["headers"])
        self.assertNotIn(TOKEN, json.dumps(seen["headers"]))

    async def test_non_streaming_completion(self):
        async with self.client.post(self.api.url + "/v1/chat/completions", json={"messages": []},
                                    headers=auth()) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual((await resp.json())["choices"][0]["message"]["content"], "hi")
        self.assertNotIn("Authorization", self.llama.requests[-1]["headers"])

    async def test_sse_is_streamed_unbuffered(self):
        async with self.client.post(self.api.url + "/v1/chat/completions", json={"stream": True},
                                    headers=auth()) as resp:
            self.assertEqual(resp.status, 200)
            self.assertTrue(resp.headers["Content-Type"].startswith("text/event-stream"))
            self.assertEqual(resp.headers["Cache-Control"], "no-cache")
            self.assertEqual(resp.headers["X-Accel-Buffering"], "no")
            first = await asyncio.wait_for(resp.content.readuntil(b"\n\n"), 5)
            self.assertEqual(first, b"data: one\n\n")  # arrived while upstream is still blocked
            self.llama.gate.set()
            rest = await asyncio.wait_for(resp.content.read(), 5)
        self.assertEqual(rest, b"data: two\n\ndata: [DONE]\n\n")

    async def test_client_disconnect_cancels_upstream(self):
        resp = await self.client.post(self.api.url + "/v1/chat/completions", json={"mode": "slow"}, headers=auth())
        await asyncio.wait_for(resp.content.readuntil(b"\n\n"), 5)
        resp.close()  # drop the connection mid-stream
        await asyncio.wait_for(self.llama.stream_closed.wait(), 5)

    async def test_not_ready_is_503(self):
        for status in ("stopped", "starting", "error"):
            self.ctx.state.services["llama"] = status
            for method, path in (("GET", "/v1/models"), ("POST", "/v1/chat/completions")):
                async with self.client.request(method, self.api.url + path, json={}, headers=auth()) as resp:
                    self.assertEqual(resp.status, 503)
                    body = await resp.json()
                    self.assertEqual(body["error"], "not_ready")
                    self.assertEqual(body["state"], "installing")
                    self.assertEqual(body["llama"], status)
        self.assertEqual(self.llama.requests, [])

    async def test_body_limit(self):
        self.ctx.max_body = 1024
        payload = json.dumps({"pad": "x" * 2000}).encode()
        async with self.client.post(self.api.url + "/v1/chat/completions", data=payload, headers=auth()) as resp:
            self.assertEqual(resp.status, 413)
            self.assertEqual((await resp.json())["error"], "payload_too_large")

        async def chunked():
            for _ in range(4):
                yield b" " * 400
        async with self.client.post(self.api.url + "/v1/chat/completions", data=chunked(), headers=auth()) as resp:
            self.assertEqual(resp.status, 413)
        ok = json.dumps({"pad": "y" * 900}).encode()
        async with self.client.post(self.api.url + "/v1/chat/completions", data=ok, headers=auth()) as resp:
            self.assertEqual(resp.status, 200)
        self.assertEqual(self.llama.requests[-1]["size"], len(ok))

    async def test_default_body_limit_is_32_mib(self):
        self.assertEqual(make_ctx(self.tmp).max_body, 32 * 1024 * 1024)
        big = io.BytesIO(b'{"pad":"' + b"z" * (32 * 1024 * 1024) + b'"}')
        async with self.client.post(self.api.url + "/v1/chat/completions", data=big, headers=auth()) as resp:
            self.assertEqual(resp.status, 413)
        fits = b'{"pad":"' + b"z" * (32 * 1024 * 1024 - 10) + b'"}'
        self.assertEqual(len(fits), 32 * 1024 * 1024)
        async with self.client.post(self.api.url + "/v1/chat/completions", data=io.BytesIO(fits),
                                    headers=auth()) as resp:
            self.assertEqual(resp.status, 200)
        self.assertEqual(self.llama.requests[-1]["size"], len(fits))

    async def test_upstream_down_is_502(self):
        await self.llama_srv.__aexit__(None, None, None)
        self.llama_srv = await RunningApp(web.Application()).__aenter__()  # placeholder for teardown
        self.ctx.llama_base = "http://127.0.0.1:9"  # nothing listens on the discard port
        async with self.client.get(self.api.url + "/v1/models", headers=auth()) as resp:
            self.assertEqual(resp.status, 502)
            self.assertEqual(await resp.json(), {"error": "upstream_unavailable"})


if __name__ == "__main__":
    unittest.main()
