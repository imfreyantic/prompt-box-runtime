import asyncio
import hashlib
import io
import unittest
from pathlib import Path

import aiohttp
from aiohttp import web
from multidict import CIMultiDict, CIMultiDictProxy

from box import gateway as G
from tests.helpers import TOKEN, FakeClock, RunningApp, make_ctx, tempdir, ticket_for


class FakeComfy:
    def __init__(self):
        self.requests: list[dict] = []
        self.ws_headers: dict = {}

    def app(self) -> web.Application:
        app = web.Application(client_max_size=256 * 1024 * 1024)

        async def ws(request: web.Request) -> web.WebSocketResponse:
            self.ws_headers = dict(request.headers)
            sock = web.WebSocketResponse()
            await sock.prepare(request)
            await sock.send_str('{"type":"status"}')
            async for msg in sock:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await sock.send_str("echo:" + msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await sock.send_bytes(b"echo:" + msg.data)
            return sock

        async def upload(request: web.Request) -> web.Response:
            digest = hashlib.sha256()
            size = 0
            async for chunk in request.content.iter_chunked(65536):
                digest.update(chunk)
                size += len(chunk)
            self.requests.append({"path": request.path_qs, "headers": dict(request.headers)})
            return web.json_response({"size": size, "sha256": digest.hexdigest()})

        async def any_path(request: web.Request) -> web.Response:
            self.requests.append({"path": request.path_qs, "headers": dict(request.headers),
                                  "method": request.method})
            if request.path == "/redirect":
                return web.Response(status=302, headers={"Location": f"http://{request.host}/elsewhere"})
            resp = web.Response(text=f"comfy {request.path}", content_type="text/html")
            resp.headers.add("Set-Cookie", "comfy_a=1; Path=/")
            resp.headers.add("Set-Cookie", "comfy_b=2; Path=/")
            return resp

        app.router.add_get("/ws", ws)
        app.router.add_post("/upload/image", upload)
        app.router.add_route("*", "/{tail:.*}", any_path)
        return app


class TicketTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempdir()
        self.clock = FakeClock()
        self.ctx = make_ctx(Path(self._tmp.name), clock=self.clock)

    def tearDown(self):
        self._tmp.cleanup()

    def now(self) -> int:
        return int(self.clock.now)

    def test_valid_ticket_once(self):
        ticket = ticket_for(self.now() + 60)
        self.assertEqual(self.ctx.verify_ticket(ticket), (True, "ok"))
        self.assertEqual(self.ctx.verify_ticket(ticket), (False, "already used"))

    def test_window_bounds(self):
        self.assertTrue(self.ctx.verify_ticket(ticket_for(self.now()))[0])
        self.assertTrue(self.ctx.verify_ticket(ticket_for(self.now() + 300))[0])
        self.assertEqual(self.ctx.verify_ticket(ticket_for(self.now() - 1)), (False, "expired"))
        self.assertEqual(self.ctx.verify_ticket(ticket_for(self.now() + 301)),
                         (False, "expiry too far in the future"))

    def test_bad_signatures_and_formats(self):
        exp = self.now() + 60
        good = ticket_for(exp)
        self.assertEqual(self.ctx.verify_ticket(ticket_for(exp, token="1" * 64)), (False, "bad signature"))
        self.assertEqual(self.ctx.verify_ticket(f"{exp}.{'0' * 64}"), (False, "bad signature"))
        for bad in (None, "", "abc", f"{exp}", good.upper(), good + "0", f"{exp}.{good.split('.')[1][:-1]}",
                    f"-{exp}.{good.split('.')[1]}", f" {good}"):
            with self.subTest(bad=bad):
                self.assertFalse(self.ctx.verify_ticket(bad)[0])
        # signature over a different exp string must not validate
        self.assertFalse(self.ctx.verify_ticket(f"0{exp}.{good.split('.')[1]}")[0])

    def test_signature_definition(self):
        import hmac as _hmac
        exp = "1800000100"
        expected = _hmac.new(TOKEN.encode(), b"comfy-login:" + exp.encode(), hashlib.sha256).hexdigest()
        self.assertEqual(self.ctx.expected_ticket_sig(exp), expected)

    def test_used_tickets_are_forgotten_after_expiry(self):
        ticket = ticket_for(self.now() + 10)
        self.assertTrue(self.ctx.verify_ticket(ticket)[0])
        self.clock.now += 11
        self.ctx.verify_ticket("junk")
        self.assertNotIn(ticket, self.ctx.used_tickets)

    def test_sessions_expire_after_12h(self):
        sid = self.ctx.new_session()
        self.assertEqual(len(sid), 64)
        self.assertTrue(self.ctx.session_valid(sid))
        self.clock.now += 12 * 3600 - 1
        self.assertTrue(self.ctx.session_valid(sid))
        self.clock.now += 2
        self.assertFalse(self.ctx.session_valid(sid))
        self.assertFalse(self.ctx.session_valid(None))
        self.assertFalse(self.ctx.session_valid("nope"))


class HeaderTests(unittest.TestCase):
    def test_upstream_headers_rewritten(self):
        incoming = CIMultiDictProxy(CIMultiDict([
            ("Host", "203.0.113.5:40001"),
            ("Origin", "https://203.0.113.5:40001"),
            ("Referer", "https://app.example/prompt"),
            ("Sec-Fetch-Site", "cross-site"),
            ("Sec-Fetch-Mode", "navigate"),
            ("Authorization", "Bearer secret"),
            ("Cookie", "pb_comfy=abc; theme=dark; other=1"),
            ("Accept", "text/html"),
            ("Connection", "keep-alive"),
        ]))
        out = G.ui_upstream_headers(incoming, "http://127.0.0.1:8188")
        self.assertEqual(out["Host"], "127.0.0.1:8188")
        self.assertEqual(out["Origin"], "http://127.0.0.1:8188")
        self.assertEqual(out["Cookie"], "theme=dark; other=1")
        self.assertEqual(out["Accept"], "text/html")
        for gone in ("Referer", "Sec-Fetch-Site", "Sec-Fetch-Mode", "Authorization", "Connection"):
            self.assertNotIn(gone, out)

    def test_only_session_cookie_dropped(self):
        self.assertEqual(G.strip_cookie("pb_comfy=1", "pb_comfy"), "")
        self.assertEqual(G.strip_cookie("a=1;pb_comfy=2; b=3", "pb_comfy"), "a=1; b=3")
        incoming = CIMultiDictProxy(CIMultiDict([("Cookie", "pb_comfy=x")]))
        self.assertNotIn("Cookie", G.ui_upstream_headers(incoming, "http://127.0.0.1:8188"))
        self.assertNotIn("Origin", G.ui_upstream_headers(incoming, "http://127.0.0.1:8188"))


class UiProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempdir()
        self.comfy = FakeComfy()
        self.comfy_srv = await RunningApp(self.comfy.app()).__aenter__()
        self.clock = FakeClock()
        self.ctx = make_ctx(Path(self._tmp.name), comfy_base=self.comfy_srv.url, clock=self.clock)
        self.ui = await RunningApp(G.create_ui_app(self.ctx)).__aenter__()
        # Plain-HTTP test server: the Secure cookie is sent by hand (see test_tls for the real cookie jar).
        self.client = aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar())

    async def asyncTearDown(self):
        await self.client.close()
        await self.ui.__aexit__(None, None, None)
        await self.comfy_srv.__aexit__(None, None, None)
        self._tmp.cleanup()

    async def login(self) -> str:
        ticket = ticket_for(int(self.clock.now) + 120)
        async with self.client.get(self.ui.url + f"/__pb_login?ticket={ticket}", allow_redirects=False) as resp:
            self.assertEqual(resp.status, 302)
            self.assertEqual(resp.headers["Location"], "/")
            self.set_cookie = resp.headers["Set-Cookie"]
        self.sid = self.set_cookie.split(";", 1)[0].split("=", 1)[1]
        return self.sid

    def cookie(self, extra: str = "") -> dict:
        return {"Cookie": f"pb_comfy={self.sid}" + (f"; {extra}" if extra else "")}

    async def test_login_sets_session_cookie(self):
        await self.login()
        cookie = self.set_cookie
        self.assertRegex(cookie, r"^pb_comfy=[0-9a-f]{64};")
        lowered = cookie.lower()
        for attr in ("httponly", "secure", "samesite=lax", "path=/", "max-age=43200"):
            self.assertIn(attr, lowered)
        self.assertEqual(len(self.ctx.sessions), 1)

    async def test_login_failures_are_401_html(self):
        now = int(self.clock.now)
        cases = [ticket_for(now - 5), ticket_for(now + 400), f"{now + 60}.{'a' * 64}", "garbage", ""]
        for ticket in cases:
            async with self.client.get(self.ui.url + f"/__pb_login?ticket={ticket}", allow_redirects=False) as resp:
                self.assertEqual(resp.status, 401)
                self.assertIn("text/html", resp.headers["Content-Type"])
                self.assertIn("/prompt", await resp.text())
                self.assertNotIn("Set-Cookie", resp.headers)
        self.assertEqual(self.ctx.sessions, {})

    async def test_replayed_ticket_rejected(self):
        ticket = ticket_for(int(self.clock.now) + 120)
        fresh = aiohttp.ClientSession()
        try:
            async with fresh.get(self.ui.url + f"/__pb_login?ticket={ticket}", allow_redirects=False) as resp:
                self.assertEqual(resp.status, 302)
        finally:
            await fresh.close()
        async with self.client.get(self.ui.url + f"/__pb_login?ticket={ticket}", allow_redirects=False) as resp:
            self.assertEqual(resp.status, 401)

    async def test_requires_session(self):
        async with self.client.get(self.ui.url + "/") as resp:
            self.assertEqual(resp.status, 401)
            self.assertIn("/prompt", await resp.text())
        async with self.client.get(self.ui.url + "/", headers={"Cookie": "pb_comfy=" + "0" * 64}) as resp:
            self.assertEqual(resp.status, 401)
        self.assertEqual(self.comfy.requests, [])

    async def test_proxy_after_login(self):
        await self.login()
        headers = {"Origin": "https://203.0.113.5:41000", "Referer": "https://app.example/prompt",
                   "Sec-Fetch-Site": "cross-site", "Authorization": "Bearer x", **self.cookie("theme=dark")}
        async with self.client.get(self.ui.url + "/some/page?q=1", headers=headers) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(await resp.text(), "comfy /some/page")
            self.assertEqual(len(resp.headers.getall("Set-Cookie")), 2)
        seen = self.comfy.requests[-1]
        self.assertEqual(seen["path"], "/some/page?q=1")
        self.assertEqual(seen["headers"]["Host"], f"127.0.0.1:{self.comfy_srv.port}")
        self.assertEqual(seen["headers"]["Origin"], self.comfy_srv.url)
        self.assertNotIn("Referer", seen["headers"])
        self.assertNotIn("Sec-Fetch-Site", seen["headers"])
        self.assertNotIn("Authorization", seen["headers"])
        self.assertEqual(seen["headers"].get("Cookie"), "theme=dark")

    async def test_absolute_upstream_redirect_is_rewritten(self):
        await self.login()
        async with self.client.get(self.ui.url + "/redirect", allow_redirects=False, headers=self.cookie()) as resp:
            self.assertEqual(resp.status, 302)
            self.assertEqual(resp.headers["Location"], "/elsewhere")

    async def test_large_upload_is_streamed(self):
        await self.login()
        blob = bytes(range(256)) * (8 * 1024 * 4)  # 8 MiB
        form = aiohttp.FormData()
        form.add_field("image", io.BytesIO(blob), filename="x.png", content_type="image/png")
        async with self.client.post(self.ui.url + "/upload/image", data=form, headers=self.cookie()) as resp:
            self.assertEqual(resp.status, 200)
            body = await resp.json()
        self.assertGreater(body["size"], len(blob))

    async def test_websocket_proxy(self):
        await self.login()
        async with self.client.ws_connect(self.ui.url.replace("http", "ws") + "/ws?clientId=abc",
                                          origin="https://203.0.113.5:41000", headers=self.cookie()) as sock:
            first = await asyncio.wait_for(sock.receive(), 5)
            self.assertEqual(first.data, '{"type":"status"}')
            await sock.send_str("hello")
            self.assertEqual((await asyncio.wait_for(sock.receive(), 5)).data, "echo:hello")
            await sock.send_bytes(b"\x00\x01")
            self.assertEqual((await asyncio.wait_for(sock.receive(), 5)).data, b"echo:\x00\x01")
        self.assertEqual(self.comfy.ws_headers.get("Origin"), self.comfy_srv.url)
        self.assertEqual(self.comfy.ws_headers.get("Host"), f"127.0.0.1:{self.comfy_srv.port}")
        self.assertNotIn("pb_comfy", self.comfy.ws_headers.get("Cookie", ""))

    async def test_websocket_requires_session(self):
        with self.assertRaises(aiohttp.WSServerHandshakeError) as cm:
            await self.client.ws_connect(self.ui.url + "/ws")
        self.assertEqual(cm.exception.status, 401)

    async def test_comfy_down_is_502(self):
        await self.login()
        self.ctx.comfy_base = "http://127.0.0.1:9"
        async with self.client.get(self.ui.url + "/", headers=self.cookie()) as resp:
            self.assertEqual(resp.status, 502)
            self.assertIn("not running", await resp.text())


if __name__ == "__main__":
    unittest.main()
