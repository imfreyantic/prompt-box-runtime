"""End-to-end orchestrator runs over real TLS with fake model files and fake services (offline)."""

import asyncio
import hashlib
import json
import os
import socket
import stat
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import aiohttp

from box import constants as C
from box import main as M
from box.config import load_config
from box.constants import PinnedFile
from tests.helpers import TOKEN, RunningApp, auth, b64, make_self_signed, tempdir, ticket_for
from tests.test_download import FakeHub, make_blob

FAKE_LLAMA = textwrap.dedent("""\
    import http.server, json, sys
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    with open(sys.argv[0] + ".argv.json", "w") as fh:
        json.dump(args, fh)
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            routes = {"/health": {"status": "ok"}, "/v1/models": {"data": [{"id": "qwen3.8-27b-aggressive"}]}}
            body = json.dumps(routes.get(self.path, {})).encode()
            self.send_response(200 if self.path in routes else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
""")

FAKE_COMFY_MAIN = textwrap.dedent("""\
    import http.server, json, sys
    port = int(sys.argv[sys.argv.index("--port") + 1])
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            routes = {"/system_stats": {"system": {}}, "/object_info": {"AAPromptLLM": {}, "PreviewAny": {}},
                      "/": {"page": "comfy"}}
            body = json.dumps(routes.get(self.path, {})).encode()
            self.send_response(200 if self.path in routes else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
""")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class OrchestratorTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempdir()
        self.root = Path(self._tmp.name)
        pems = make_self_signed(self.root)
        if pems is None:
            self.skipTest("openssl CLI not available to create a test certificate")
        self.cert, self.key = pems
        self.api_port, self.ui_port = free_port(), free_port()

    def tearDown(self):
        self._tmp.cleanup()

    def config(self, **env):
        base = {"PB_TOKEN": TOKEN, "PB_TLS_CERT_B64": b64(self.cert), "PB_TLS_KEY_B64": b64(self.key),
                "PB_ROOT": str(self.root / "box"), "PB_RUNTIME_COMMIT": "abc123"}
        base.update(env)
        cfg = load_config(base)
        cfg.paths.ensure()
        return cfg

    async def start(self, orch):
        self.orch = orch
        self.task = asyncio.create_task(orch.run())
        self.client = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False),
                                            cookie_jar=aiohttp.CookieJar(unsafe=True))
        for _ in range(100):
            try:
                async with self.client.get(self.api_url + "/health", headers=auth()) as resp:
                    if resp.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.05)
        self.fail("gateway did not come up")

    async def stop(self):
        await self.client.close()
        self.orch.stop_event.set()
        self.assertEqual(await asyncio.wait_for(self.task, 30), 0)

    @property
    def api_url(self):
        return f"https://127.0.0.1:{self.api_port}"

    @property
    def ui_url(self):
        return f"https://127.0.0.1:{self.ui_port}"

    async def health(self):
        async with self.client.get(self.api_url + "/health", headers=auth()) as resp:
            self.assertEqual(resp.status, 200)
            return await resp.json()


class ConfigErrorTests(OrchestratorTestCase):
    async def test_invalid_quant_reported_over_tls(self):
        cfg = self.config(PB_QUANT="Q3_K_P", PB_CTX="999999")
        await self.start(M.Orchestrator(cfg, host="127.0.0.1", api_port=self.api_port, ui_port=self.ui_port))
        try:
            body = await self.health()
            self.assertEqual((body["state"], body["stage"]), ("error", "error"))
            self.assertIn("PB_QUANT", body["error"])
            self.assertIn("PB_CTX", body["error"])
            self.assertEqual(body["versions"]["quant"], "Q3_K_P")
            self.assertEqual(body["versions"]["runtimeCommit"], "abc123")
            self.assertEqual([f["name"] for f in body["progress"]["download"]["files"]],
                             [C.MMPROJ_FILE.name, C.FASTMTP_FILE.name])
            async with self.client.get(self.api_url + "/health") as resp:
                self.assertEqual(resp.status, 401)
            async with self.client.get(self.api_url + "/logs?source=box", headers=auth()) as resp:
                self.assertEqual(resp.status, 200)
            key_path = cfg.paths.tls_dir / "key.pem"
            self.assertEqual(stat.S_IMODE(os.stat(key_path).st_mode), 0o600)
            # UI port: no session -> 401 page; ticket login -> Secure cookie accepted over HTTPS
            async with self.client.get(self.ui_url + "/") as resp:
                self.assertEqual(resp.status, 401)
            import time
            ticket = ticket_for(int(time.time()) + 60)
            async with self.client.get(self.ui_url + f"/__pb_login?ticket={ticket}", allow_redirects=False) as resp:
                self.assertEqual(resp.status, 302)
            self.assertIn("pb_comfy", {c.key for c in self.client.cookie_jar})
            async with self.client.get(self.ui_url + "/") as resp:
                self.assertEqual(resp.status, 502)  # session ok, ComfyUI not running
        finally:
            await self.stop()


class FullPipelineTests(OrchestratorTestCase):
    async def test_download_start_services_ready_and_proxy(self):
        text_data, mm_data, mtp_data = make_blob(300_000, 1), make_blob(70_000, 2), make_blob(50_000, 3)
        specs = {
            "text": PinnedFile("Qwen-test-Q8_K_P.gguf", len(text_data), hashlib.sha256(text_data).hexdigest()),
            "mm": PinnedFile("mmproj-test.gguf", len(mm_data), hashlib.sha256(mm_data).hexdigest()),
            "mtp": PinnedFile("fastmtp-test.gguf", len(mtp_data), hashlib.sha256(mtp_data).hexdigest()),
        }
        hub = FakeHub({specs["text"].name: text_data, specs["mm"].name: mm_data, specs["mtp"].name: mtp_data})
        hub_srv = await RunningApp(hub.hub_app()).__aenter__()
        cdn_srv = await RunningApp(hub.cdn_app()).__aenter__()
        hub.cdn_url = cdn_srv.url

        fake_bin = self.root / "fakebin"
        fake_bin.mkdir()
        llama = fake_bin / "llama-server"
        llama.write_text(f"#!{sys.executable}\n" + FAKE_LLAMA)
        llama.chmod(0o755)

        llama_port, comfy_port = free_port(), free_port()
        patches = mock.patch.multiple(
            C, QUANT_FILES={"Q8_K_P": specs["text"]}, MMPROJ_FILE=specs["mm"], FASTMTP_FILE=specs["mtp"],
            LLAMA_PORT=llama_port, COMFY_PORT=comfy_port,
        )
        patches.start()
        self.addCleanup(patches.stop)
        cfg = self.config()
        comfy_dir = cfg.paths.comfy_dir
        comfy_dir.mkdir(parents=True)
        (comfy_dir / "main.py").write_text(FAKE_COMFY_MAIN)

        async def fake_llama_build(cfg_, state, session, hf_base):
            state.build_state = "done"
            state.build_percent = 100
            return llama

        async def fake_comfy_install(cfg_, state):
            state.comfy_state = "done"
            return Path(sys.executable)

        with mock.patch.object(M, "ensure_llama", fake_llama_build), \
                mock.patch.object(M, "ensure_comfy", fake_comfy_install):
            await self.start(M.Orchestrator(cfg, hf_base=hub_srv.url, host="127.0.0.1",
                                            api_port=self.api_port, ui_port=self.ui_port))
            try:
                body = None
                for _ in range(300):
                    body = await self.health()
                    if body["state"] != "installing":
                        break
                    await asyncio.sleep(0.1)
                self.assertEqual(body["state"], "ready", body)
                self.assertEqual(body["stage"], "ready")
                self.assertEqual(body["message"], "Ready")
                self.assertIsInstance(body["readyAt"], int)
                self.assertEqual(body["services"], {"llama": "ready", "comfy": "ready"})
                dl = body["progress"]["download"]
                self.assertTrue(all(f["verified"] for f in dl["files"]))
                self.assertEqual(dl["doneBytes"], dl["totalBytes"])
                self.assertEqual(body["versions"]["ctx"], 131072)
                models = cfg.paths.models
                self.assertEqual((models / specs["text"].name).read_bytes(), text_data)
                argv = json.loads((fake_bin / "llama-server.argv.json").read_text())
                self.assertEqual(argv[argv.index("--model") + 1], str(models / specs["text"].name))
                self.assertEqual(argv[argv.index("--mmproj") + 1], str(models / specs["mm"].name))
                self.assertEqual(argv[argv.index("--spec-draft-model") + 1], str(models / specs["mtp"].name))
                self.assertEqual(argv[argv.index("--port") + 1], str(llama_port))
                async with self.client.get(self.api_url + "/v1/models", headers=auth()) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual((await resp.json())["data"][0]["id"], "qwen3.8-27b-aggressive")
                self.assertTrue((cfg.paths.run_dir / "llama.pid").exists())
                self.assertIn("starting llama-server", cfg.paths.log("llama").read_text())
            finally:
                await self.stop()
                await cdn_srv.__aexit__(None, None, None)
                await hub_srv.__aexit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
