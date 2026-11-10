import unittest

from box.state import FileProgress
from tests.helpers import FakeClock, make_state


class StageTests(unittest.TestCase):
    def setUp(self):
        self.state = make_state()
        self.state.files = [FileProgress("m.gguf", 100), FileProgress("p.gguf", 10)]

    def finish_downloads(self):
        for f in self.state.files:
            f.verified = True

    def test_progression(self):
        s = self.state
        self.assertEqual((s.state, s.stage), ("installing", "download"))
        self.finish_downloads()
        self.assertEqual(s.stage, "build")
        s.build_state = "done"
        self.assertEqual(s.stage, "comfy")
        s.comfy_state = "done"
        self.assertEqual(s.stage, "llama_start")
        s.set_service("llama", "ready")
        self.assertEqual(s.stage, "comfy_start")
        self.assertIsNone(s.ready_at)
        s.set_service("comfy", "ready")
        self.assertEqual((s.state, s.stage, s.message), ("ready", "ready", "Ready"))
        self.assertIsInstance(s.ready_at, int)

    def test_ready_is_sticky_and_restarts_are_visible(self):
        s = self.state
        self.finish_downloads()
        s.build_state = s.comfy_state = "done"
        s.set_service("llama", "ready")
        s.set_service("comfy", "ready")
        ready_at = s.ready_at
        s.set_service("llama", "starting")
        s.set_note("llama", "llama-server exited with code -9; restarting in 10s (1/3)")
        health = s.to_health()
        self.assertEqual((health["state"], health["stage"], health["readyAt"]), ("ready", "ready", ready_at))
        self.assertEqual(health["services"]["llama"], "starting")
        self.assertIn("restarting", health["message"])

    def test_error_wins(self):
        s = self.state
        s.fail("download", "disk full")
        s.fail("download", "disk full")
        s.fail("build", "cmake failed")
        self.assertEqual((s.state, s.stage), ("error", "error"))
        self.assertEqual(s.message, "download: disk full")
        self.assertEqual(s.to_health()["error"], "download: disk full | build: cmake failed")
        s.set_service("llama", "ready")
        s.set_service("comfy", "ready")
        self.assertIsNone(s.ready_at)

    def test_message_mentions_parallel_work(self):
        s = self.state
        s.files[0].done = 50
        s.build_state, s.build_percent = "running", 42
        s.comfy_state = "running"
        msg = s.message
        self.assertIn("Downloading model files", msg)
        self.assertIn("(45%)", msg)
        self.assertIn("Building llama.cpp (42%)", msg)
        self.assertIn("Installing ComfyUI", msg)

    def test_download_totals(self):
        s = self.state
        s.files[0].done = 30
        s.files[1].verified = True
        self.assertEqual(s.download_totals(), (40, 110))
        self.assertEqual(s.files[0].as_json(), {"name": "m.gguf", "doneBytes": 30, "totalBytes": 100,
                                                "verified": False})

    def test_started_at_uses_clock(self):
        from box.state import BoxState
        s = BoxState(versions={}, clock=FakeClock(1234.5))
        self.assertEqual(s.started_at, 1234500)


if __name__ == "__main__":
    unittest.main()
