"""HTTPS gateway.

Port 8443 (API): bearer-token protected /health, /logs and a reverse proxy for
llama-server's OpenAI-compatible endpoints.
Port 8444 (UI): single-use login tickets create a cookie session; everything
else is reverse-proxied (HTTP and WebSocket) to ComfyUI.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import logging
import re
import secrets
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web
from multidict import CIMultiDict, CIMultiDictProxy

from . import constants as C
from .config import Config, Paths
from .logs import Redactor, tail_lines
from .state import BoxState

log = logging.getLogger("box.gateway")

HOP_BY_HOP = frozenset(
    h.lower()
    for h in (
        "Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization", "TE", "Trailer",
        "Trailers", "Transfer-Encoding", "Upgrade", "Proxy-Connection",
    )
)
_TICKET_RE = re.compile(r"([0-9]{1,12})\.([0-9a-f]{64})")
_STREAM_READ = 64 * 1024

UNAUTHORIZED_HTML = (
    "<!doctype html><html><head><meta charset=\"utf-8\"><title>Prompt box</title></head>"
    "<body style=\"font-family:sans-serif;margin:3em\"><h1>Not signed in</h1>"
    "<p>This ComfyUI session is missing or has expired. Open ComfyUI from the app's <b>/prompt</b> page.</p>"
    "</body></html>"
)
UPSTREAM_DOWN_HTML = (
    "<!doctype html><html><head><meta charset=\"utf-8\"><title>Prompt box</title></head>"
    "<body style=\"font-family:sans-serif;margin:3em\"><h1>ComfyUI is not running yet</h1>"
    "<p>The box is still starting ComfyUI. Try again in a moment.</p></body></html>"
)


@dataclass
class GatewayContext:
    token: str
    state: BoxState
    paths: Paths
    redactor: Redactor
    llama_base: str = f"http://{C.LOCAL_HOST}:{C.LLAMA_PORT}"
    comfy_base: str = f"http://{C.LOCAL_HOST}:{C.COMFY_PORT}"
    max_body: int = C.MAX_BODY_BYTES
    clock: Callable[[], float] = time.time
    used_tickets: dict[str, int] = field(default_factory=dict)
    sessions: dict[str, float] = field(default_factory=dict)
    max_sessions: int = 1000
    llama_http: Optional[aiohttp.ClientSession] = None
    comfy_http: Optional[aiohttp.ClientSession] = None

    # --- API auth ------------------------------------------------------------------------
    def check_bearer(self, header: Optional[str]) -> bool:
        if not header:
            return False
        scheme, _, value = header.strip().partition(" ")
        if scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(value.strip().encode("utf-8"), self.token.encode("utf-8"))

    # --- UI login --------------------------------------------------------------------------
    def expected_ticket_sig(self, exp: str) -> str:
        return hmac.new(self.token.encode("utf-8"), f"comfy-login:{exp}".encode("ascii"), hashlib.sha256).hexdigest()

    def verify_ticket(self, ticket: Optional[str]) -> tuple[bool, str]:
        now = int(self.clock())
        for used, exp in list(self.used_tickets.items()):
            if exp < now:
                del self.used_tickets[used]
        m = _TICKET_RE.fullmatch(ticket or "")
        if not m:
            return False, "malformed"
        exp_s, sig = m.groups()
        exp = int(exp_s)
        if exp < now:
            return False, "expired"
        if exp > now + C.TICKET_MAX_AGE_S:
            return False, "expiry too far in the future"
        if not hmac.compare_digest(sig, self.expected_ticket_sig(exp_s)):
            return False, "bad signature"
        if ticket in self.used_tickets:
            return False, "already used"
        self.used_tickets[ticket] = exp  # type: ignore[index]
        return True, "ok"

    def new_session(self) -> str:
        now = self.clock()
        for sid, exp in list(self.sessions.items()):
            if exp <= now:
                del self.sessions[sid]
        while len(self.sessions) >= self.max_sessions:
            del self.sessions[min(self.sessions, key=self.sessions.__getitem__)]
        sid = secrets.token_hex(32)
        self.sessions[sid] = now + C.SESSION_TTL_S
        return sid

    def session_valid(self, sid: Optional[str]) -> bool:
        if not sid:
            return False
        exp = self.sessions.get(sid)
        if exp is None:
            return False
        if exp <= self.clock():
            self.sessions.pop(sid, None)
            return False
        return True


# --- shared proxy helpers -------------------------------------------------------------------
def new_proxy_session(limit: int) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=limit, force_close=False),
        cookie_jar=aiohttp.DummyCookieJar(),
        auto_decompress=False,
        skip_auto_headers=("Accept-Encoding", "User-Agent"),
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None),
    )


def strip_cookie(header: str, name: str) -> str:
    kept = []
    for part in header.split(";"):
        key = part.split("=", 1)[0].strip()
        if key and key != name:
            kept.append(part.strip())
    return "; ".join(kept)


def copy_response_headers(
    upstream: aiohttp.ClientResponse, resp: web.StreamResponse, upstream_base: Optional[str] = None
) -> None:
    for key, value in upstream.headers.items():
        lk = key.lower()
        if lk in HOP_BY_HOP:
            continue
        if lk == "location" and upstream_base and value.startswith(upstream_base):
            value = value[len(upstream_base):] or "/"
        resp.headers.add(key, value)


async def stream_upstream(
    request: web.Request, upstream: aiohttp.ClientResponse, resp: web.StreamResponse
) -> web.StreamResponse:
    """Send ``upstream``'s body to the client chunk by chunk, as it arrives.

    If the client disconnects (write fails, or aiohttp cancels the handler) the
    upstream connection is closed, which makes llama-server abort the generation.
    """
    finished = False
    try:
        await resp.prepare(request)
        async for chunk in upstream.content.iter_any():
            await resp.write(chunk)
        finished = True
    except ConnectionResetError:
        log.info("client disconnected from %s", request.path)
        return resp
    finally:
        if finished:
            upstream.release()
        else:
            upstream.close()
    await resp.write_eof()
    return resp


async def _request_body_stream(request: web.Request) -> AsyncIterator[bytes]:
    async for chunk in request.content.iter_chunked(_STREAM_READ):
        yield chunk


class PayloadTooLarge(Exception):
    pass


async def read_limited(request: web.Request, limit: int) -> bytes:
    if request.content_length is not None and request.content_length > limit:
        raise PayloadTooLarge()
    body = bytearray()
    async for chunk in request.content.iter_any():
        body += chunk
        if len(body) > limit:
            raise PayloadTooLarge()
    return bytes(body)


# --- API app (8443) ---------------------------------------------------------------------------
def create_api_app(ctx: GatewayContext) -> web.Application:
    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        if not ctx.check_bearer(request.headers.get("Authorization")):
            return web.json_response({"error": "unauthorized"}, status=401, headers={"WWW-Authenticate": "Bearer"})
        return await handler(request)

    @web.middleware
    async def json_errors(request: web.Request, handler):
        try:
            return await handler(request)
        except web.HTTPException as exc:
            if exc.status >= 400 and exc.content_type != "application/json":
                code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status, "http_error")
                return web.json_response({"error": code}, status=exc.status)
            raise

    async def health(request: web.Request) -> web.Response:
        return web.json_response(ctx.state.to_health(), headers={"Cache-Control": "no-store"})

    async def logs(request: web.Request) -> web.Response:
        source = request.query.get("source", "box")
        if source not in C.LOG_SOURCES:
            return web.json_response({"error": "invalid_source", "allowed": list(C.LOG_SOURCES)}, status=400)
        raw = request.query.get("lines", str(C.LOG_LINES_DEFAULT))
        try:
            n = int(raw)
        except ValueError:
            return web.json_response({"error": "invalid_lines"}, status=400)
        n = max(1, min(n, C.LOG_LINES_MAX))
        lines = await asyncio.to_thread(tail_lines, ctx.paths.log(source), n)
        return web.json_response({"lines": [ctx.redactor(line) for line in lines]},
                                 headers={"Cache-Control": "no-store"})

    async def llama_proxy(request: web.Request) -> web.StreamResponse:
        if ctx.state.services.get("llama") != "ready" or ctx.llama_http is None:
            return web.json_response(
                {"error": "not_ready", "state": ctx.state.state, "llama": ctx.state.services.get("llama")},
                status=503,
            )
        body: Optional[bytes] = None
        if request.method == "POST":
            try:
                body = await read_limited(request, ctx.max_body)
            except PayloadTooLarge:
                return web.json_response({"error": "payload_too_large", "maxBytes": ctx.max_body}, status=413)
        headers = CIMultiDict()
        for key, value in request.headers.items():
            lk = key.lower()
            if lk in HOP_BY_HOP or lk in ("host", "authorization", "cookie", "content-length", "expect"):
                continue
            headers.add(key, value)
        data = io.BytesIO(body) if body is not None and len(body) > 1024 * 1024 else body
        try:
            upstream = await ctx.llama_http.request(
                request.method, ctx.llama_base + request.rel_url.path_qs, headers=headers, data=data,
                allow_redirects=False,
            )
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            log.warning("llama-server proxy error: %r", exc)
            return web.json_response({"error": "upstream_unavailable"}, status=502)
        resp = web.StreamResponse(status=upstream.status, reason=upstream.reason)
        copy_response_headers(upstream, resp)
        if upstream.headers.get("Content-Type", "").startswith("text/event-stream"):
            resp.headers.popall("Content-Length", None)
            resp.headers["Cache-Control"] = "no-cache"
            resp.headers["X-Accel-Buffering"] = "no"
        return await stream_upstream(request, upstream, resp)

    app = web.Application(middlewares=[auth_middleware, json_errors], client_max_size=ctx.max_body + 1)
    app.router.add_get("/health", health)
    app.router.add_get("/logs", logs)
    app.router.add_post("/v1/chat/completions", llama_proxy)
    app.router.add_get("/v1/models", llama_proxy)

    async def client_ctx(_app: web.Application):
        ctx.llama_http = new_proxy_session(limit=32)
        yield
        await ctx.llama_http.close()
        ctx.llama_http = None

    app.cleanup_ctx.append(client_ctx)
    return app


# --- UI app (8444) ----------------------------------------------------------------------------
def _html(text: str, status: int) -> web.Response:
    return web.Response(text=text, status=status, content_type="text/html",
                        headers={"Cache-Control": "no-store"})


def ui_upstream_headers(headers: "CIMultiDictProxy[str]", upstream_base: str) -> CIMultiDict:
    """Headers for ComfyUI: our cookie/auth removed, Host/Origin made same-origin.

    ComfyUI rejects requests whose Origin differs from a loopback Host and any
    request marked ``Sec-Fetch-Site: cross-site`` (the redirect after login is), so
    Origin is rewritten and Referer / Sec-Fetch-* are dropped.
    """
    parts = urlsplit(upstream_base)
    out: CIMultiDict = CIMultiDict()
    for key, value in headers.items():
        lk = key.lower()
        if (
            lk in HOP_BY_HOP
            or lk in ("host", "authorization", "origin", "referer", "content-length", "expect", "x-forwarded-for")
            or lk.startswith("sec-fetch-")
            or lk.startswith("sec-websocket-")
        ):
            continue
        if lk == "cookie":
            value = strip_cookie(value, C.SESSION_COOKIE)
            if not value:
                continue
        out.add(key, value)
    out["Host"] = parts.netloc
    if "Origin" in headers:
        out["Origin"] = f"{parts.scheme}://{parts.netloc}"
    return out


def create_ui_app(ctx: GatewayContext) -> web.Application:
    async def login(request: web.Request) -> web.Response:
        ok, reason = ctx.verify_ticket(request.query.get("ticket"))
        if not ok:
            if ctx.session_valid(request.cookies.get(C.SESSION_COOKIE)):
                return web.Response(status=302, headers={"Location": "/", "Cache-Control": "no-store"})
            log.warning("ComfyUI login rejected: %s", reason)
            return _html(UNAUTHORIZED_HTML, 401)
        sid = ctx.new_session()
        resp = web.Response(status=302, headers={"Location": "/", "Cache-Control": "no-store"})
        resp.set_cookie(
            C.SESSION_COOKIE, sid, max_age=C.SESSION_TTL_S, path="/", secure=True, httponly=True, samesite="Lax"
        )
        log.info("ComfyUI login accepted; %d active session(s)", len(ctx.sessions))
        return resp

    async def proxy_ws(request: web.Request, headers: CIMultiDict) -> web.StreamResponse:
        assert ctx.comfy_http is not None
        headers = CIMultiDict((k, v) for k, v in headers.items() if k.lower() not in ("host", "origin"))
        parts = urlsplit(ctx.comfy_base)
        try:
            upstream = await ctx.comfy_http.ws_connect(
                ctx.comfy_base + request.raw_path, headers=headers, origin=f"{parts.scheme}://{parts.netloc}",
                max_msg_size=0, autoping=True,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("ComfyUI websocket connect failed: %r", exc)
            return _html(UPSTREAM_DOWN_HTML, 502)
        client = web.WebSocketResponse(max_msg_size=0, autoping=True, heartbeat=30.0)
        await client.prepare(request)

        async def pump(src, dst) -> None:
            async for msg in src:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await dst.send_str(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await dst.send_bytes(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                    break

        tasks = [asyncio.create_task(pump(client, upstream)), asyncio.create_task(pump(upstream, client))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await upstream.close()
            await client.close()
        return client

    async def handler(request: web.Request) -> web.StreamResponse:
        if request.path == "/__pb_login" and request.method == "GET":
            return await login(request)
        if not ctx.session_valid(request.cookies.get(C.SESSION_COOKIE)):
            return _html(UNAUTHORIZED_HTML, 401)
        if not request.raw_path.startswith("/") or ctx.comfy_http is None:
            return _html("<h1>Bad request</h1>", 400)
        headers = ui_upstream_headers(request.headers, ctx.comfy_base)
        if web.WebSocketResponse().can_prepare(request).ok:
            return await proxy_ws(request, headers)
        data = None
        if request.body_exists:
            data = _request_body_stream(request)
            if request.content_length is not None:
                headers["Content-Length"] = str(request.content_length)
        try:
            upstream = await ctx.comfy_http.request(
                request.method, ctx.comfy_base + request.raw_path, headers=headers, data=data,
                allow_redirects=False,
            )
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            log.warning("ComfyUI proxy error: %r", exc)
            return _html(UPSTREAM_DOWN_HTML, 502)
        resp = web.StreamResponse(status=upstream.status, reason=upstream.reason)
        copy_response_headers(upstream, resp, ctx.comfy_base)
        return await stream_upstream(request, upstream, resp)

    app = web.Application(client_max_size=1024**3)
    app.router.add_route("*", "/{tail:.*}", handler)

    async def client_ctx(_app: web.Application):
        ctx.comfy_http = new_proxy_session(limit=100)
        yield
        await ctx.comfy_http.close()
        ctx.comfy_http = None

    app.cleanup_ctx.append(client_ctx)
    return app


# --- serving ------------------------------------------------------------------------------------
def make_ssl_context(cert_path: Path, key_path: Path) -> ssl.SSLContext:
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx


async def start_site(
    app: web.Application, host: str, port: int, ssl_context: Optional[ssl.SSLContext]
) -> web.AppRunner:
    runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port, ssl_context=ssl_context, reuse_address=True)
    await site.start()
    return runner


async def start_gateway(
    ctx: GatewayContext, ssl_context: ssl.SSLContext, host: str = "0.0.0.0",
    api_port: int = C.API_PORT, ui_port: int = C.UI_PORT,
) -> list[web.AppRunner]:
    api = await start_site(create_api_app(ctx), host, api_port, ssl_context)
    try:
        ui = await start_site(create_ui_app(ctx), host, ui_port, ssl_context)
    except BaseException:
        await api.cleanup()
        raise
    return [api, ui]


def context_from_config(cfg: Config, state: BoxState, redactor: Redactor) -> GatewayContext:
    return GatewayContext(
        token=cfg.token, state=state, paths=cfg.paths, redactor=redactor,
        llama_base=f"http://{C.LOCAL_HOST}:{C.LLAMA_PORT}", comfy_base=f"http://{C.LOCAL_HOST}:{C.COMFY_PORT}",
    )


__all__ = [
    "GatewayContext", "create_api_app", "create_ui_app", "start_gateway", "make_ssl_context",
    "context_from_config", "ui_upstream_headers", "strip_cookie",
]
