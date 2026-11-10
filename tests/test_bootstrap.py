import os
import re
import shutil
import subprocess
import unittest

from tests.helpers import PROMPT_BOX_DIR


class BootstrapTests(unittest.TestCase):
    script = PROMPT_BOX_DIR / "bootstrap.sh"

    @unittest.skipIf(shutil.which("bash") is None, "bash not installed")
    def test_bash_syntax(self):
        proc = subprocess.run(["bash", "-n", str(self.script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_strict_mode_and_entrypoint(self):
        text = self.script.read_text()
        self.assertIn("set -euo pipefail", text)
        self.assertIn('VENV="$PB_ROOT/venv-box"', text)
        self.assertIn('exec "$VENV/bin/python" -m box.main', text)
        self.assertIn('PB_ROOT="${PB_ROOT:-/workspace/prompt-box}"', text)
        self.assertIn("aiohttp", (PROMPT_BOX_DIR / "requirements-box.txt").read_text())

    def test_requirements_are_pinned(self):
        for name in ("requirements-box.txt", "requirements-build.txt"):
            lines = [ln.strip() for ln in (PROMPT_BOX_DIR / name).read_text().splitlines()]
            pins = [ln for ln in lines if ln and not ln.startswith("#")]
            self.assertTrue(pins, name)
            for line in pins:
                self.assertRegex(line, r"^[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*$")
        build = (PROMPT_BOX_DIR / "requirements-build.txt").read_text()
        self.assertIn("cmake==", build)
        self.assertIn("ninja==", build)

    def test_shipped_files_contain_no_obvious_secrets(self):
        # tests/ holds deliberately fake key material for the redaction tests, so it is skipped.
        pattern = re.compile(r"hf_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----\s*\n\s*[A-Za-z0-9+/]{40,}|"
                             r"ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}")
        for path in PROMPT_BOX_DIR.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or "tests" in path.relative_to(PROMPT_BOX_DIR).parts:
                continue
            self.assertIsNone(pattern.search(path.read_text(errors="replace")), str(path))

    def test_executable_bit_not_required(self):
        # onstart runs `bash bootstrap.sh`, so the file mode does not matter, but keep it readable.
        self.assertTrue(os.access(self.script, os.R_OK))


if __name__ == "__main__":
    unittest.main()
