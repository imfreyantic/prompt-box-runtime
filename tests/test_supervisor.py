import asyncio
import socket
import sys
import textwrap
import unittest
from pathlib import Path

import aiohttp

from box.procs import run_logged
from box.services import Supervisor
from tests.helpers import tempdir

SERVICE = textwrap.dedent("""
    import http.server, sys, threading, time
    port, lifetime = int(sys.argv[1]), float(sys.argv[2])
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    srv.allow_reuse_address = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("service up", flush=True)
    time.sleep(lifetime)
    sys.exit(3)
""")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempdir()
        self.tmp = Path(self._tmp.name)
        self.script = self.tmp / "svc.py"
        self.script.write_text(SERVICE)
        self.port = free_port()
        self.statuses: list[str] = []
        self.fatal: list[str] = []
        self.notes: list = []
        self.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        self._tmp.cleanup()

    async def healthy(self) -> bool:
        try:
            async with self.session.get(f"http://127.0.0.1:{self.port}/health",
                                        timeout=aiohttp.ClientTimeout(total=1)) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    def make(self, lifetime: float, **kwargs) -> Supervisor:
        defaults = dict(
            log_path=self.tmp / "svc.log", health=self.healthy, on_status=self.statuses.append,
            on_fatal=self.fatal.append, on_note=self.notes.append, pid_path=self.tmp / "svc.pid",
            backoff=(0.05,), start_timeout=10, poll_interval=0.05, label="svc",
        )
        defaults.update(kwargs)
        return Supervisor("svc", [sys.executable, str(self.script), str(self.port), str(lifetime)], **defaults)

    async def test_restarts_three_times_then_errors(self):
        sup = self.make(lifetime=0.4)
        await asyncio.wait_for(sup.run(), 30)
        self.assertEqual(self.statuses.count("ready"), 4)  # first start + 3 restarts
        self.assertEqual(self.statuses[-1], "error")
        self.assertEqual(len(self.fatal), 1)
        self.assertIn("restart limit (3)", self.fatal[0])
        self.assertEqual(sup.restarts, 3)
        self.assertTrue(any(n and "restarting" in n for n in self.notes))
        log = (self.tmp / "svc.log").read_text()
        self.assertEqual(log.count("service up"), 4)
        self.assertTrue((self.tmp / "svc.pid").read_text().strip().isdigit())

    async def test_never_healthy_counts_as_failed_start(self):
        sup = self.make(lifetime=0, health=lambda: asyncio.sleep(0, result=False))
        await asyncio.wait_for(sup.run(), 30)
        self.assertNotIn("ready", self.statuses)
        self.assertEqual(self.statuses[-1], "error")
        self.assertIn("failed to start", self.fatal[0])

    async def test_post_ready_failure_is_fatal_without_restart(self):
        async def bad_post_ready():
            raise RuntimeError("node AAPromptLLM missing")

        sup = self.make(lifetime=0.3, post_ready=bad_post_ready)
        await asyncio.wait_for(sup.run(), 30)
        self.assertEqual(self.fatal, ["node AAPromptLLM missing"])
        self.assertEqual(self.statuses[-1], "error")
        self.assertEqual(sup.restarts, 0)

    async def test_port_already_in_use_fails_fast(self):
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", self.port))
        blocker.listen()
        try:
            sup = self.make(lifetime=5, port=self.port)
            await asyncio.wait_for(sup.run(), 10)
        finally:
            blocker.close()
        self.assertEqual(self.statuses, ["error"])
        self.assertIn("already in use", self.fatal[0])
        self.assertIsNone(sup.proc)

    async def test_stop_terminates_service(self):
        sup = self.make(lifetime=60)
        task = asyncio.create_task(sup.run())
        for _ in range(200):
            if "ready" in self.statuses:
                break
            await asyncio.sleep(0.05)
        self.assertIn("ready", self.statuses)
        await sup.stop()
        await asyncio.wait_for(task, 10)
        self.assertIsNotNone(sup.proc.returncode)
        self.assertEqual(self.fatal, [])


class RunLoggedTests(unittest.IsolatedAsyncioTestCase):
    async def test_output_lines_and_exit_code(self):
        with tempdir() as tmp:
            log = Path(tmp) / "x.log"
            lines = []
            rc = await run_logged([sys.executable, "-c", "import sys; print('[1/4] a'); print('[2/4] b\\r[3/4] c');"
                                   "sys.exit(5)"], log, on_line=lines.append)
            self.assertEqual(rc, 5)
            self.assertEqual(lines, ["[1/4] a", "[2/4] b", "[3/4] c"])
            text = log.read_text()
            self.assertIn("[exit code 5]", text)
            self.assertIn("[1/4] a", text)

    async def test_missing_executable_and_timeout(self):
        with tempdir() as tmp:
            log = Path(tmp) / "x.log"
            self.assertEqual(await run_logged(["/nonexistent/binary"], log), 127)
            rc = await run_logged([sys.executable, "-c", "import time; time.sleep(30)"], log, timeout=0.5)
            self.assertEqual(rc, 124)


if __name__ == "__main__":
    unittest.main()
