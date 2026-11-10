import logging
import unittest
from pathlib import Path

from box.logs import Redactor, RedactingFilter, tail_lines
from tests.helpers import TOKEN, tempdir

KEY_B64 = "LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tCk1JSUV2UUlCQURBTkJna3Foa2lHOXcwQkFRRUZBQVND"


class RedactionTests(unittest.TestCase):
    def setUp(self):
        self.redact = Redactor([TOKEN, KEY_B64, "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC", "short"])

    def test_known_secrets(self):
        line = f"env PB_TOKEN={TOKEN} key={KEY_B64} body MIIEvQIBADANBgkqhkiG9w0BAQEFAASC end"
        out = self.redact(line)
        self.assertNotIn(TOKEN, out)
        self.assertNotIn(KEY_B64, out)
        self.assertNotIn("MIIEvQIBADANBgkqhkiG9w0BAQEFAASC", out)
        self.assertIn("end", out)

    def test_short_strings_are_not_used_as_patterns(self):
        self.assertEqual(self.redact("a short line"), "a short line")

    def test_generic_patterns(self):
        cases = {
            "Authorization: Bearer abc.def-ghi_jkl": "abc.def-ghi_jkl",
            "authorization=Basic dXNlcjpwYXNz": "dXNlcjpwYXNz",
            "curl -H 'bearer 0123456789abcdefXYZ'": "0123456789abcdefXYZ",
            "Cookie: pb_comfy=deadbeefcafe; theme=dark": "deadbeefcafe",
            "GET /__pb_login?ticket=1800000000.abcdef0123 HTTP/1.1": "1800000000.abcdef0123",
            "PB_TLS_KEY_B64=Zm9vYmFyYmF6": "Zm9vYmFyYmF6",
            "PB_LLAMA_PREBUILT_URL=https://x/y?sig=1": "https://x/y?sig=1",
            "-----BEGIN PRIVATE KEY-----": "BEGIN PRIVATE KEY",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----": "MIIabc",
        }
        for text, secret in cases.items():
            with self.subTest(text=text):
                self.assertNotIn(secret, Redactor()(text))

    def test_harmless_text_unchanged(self):
        for text in ("llama_model_loader: loaded meta data", "srv  update_slots: all slots are idle",
                     "To see the GUI go to: http://127.0.0.1:8188", "the bearer of good news"):
            self.assertEqual(Redactor([TOKEN])(text), text)

    def test_logging_filter(self):
        logger = logging.getLogger("box.test-redaction")
        logger.propagate = False
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(self.format(record))

        handler = Capture()
        handler.addFilter(RedactingFilter(Redactor([TOKEN])))
        logger.addHandler(handler)
        try:
            logger.warning("token %s leaked", TOKEN)
        finally:
            logger.removeHandler(handler)
        self.assertEqual(records, ["token [REDACTED] leaked"])


class TailTests(unittest.TestCase):
    def test_tail_large_file(self):
        with tempdir() as tmp:
            path = Path(tmp) / "x.log"
            path.write_text("".join(f"line {i}\n" for i in range(100_000)))
            self.assertEqual(tail_lines(path, 3), ["line 99997", "line 99998", "line 99999"])
            self.assertEqual(len(tail_lines(path, 1000)), 1000)
            self.assertEqual(tail_lines(path, 1000)[0], "line 99000")

    def test_small_files_partial_last_line_and_ansi(self):
        with tempdir() as tmp:
            path = Path(tmp) / "x.log"
            path.write_bytes(b"a\r\n\x1b[32mgreen\x1b[0m\nno newline \xff")
            self.assertEqual(tail_lines(path, 10), ["a", "green", "no newline �"])
            self.assertEqual(tail_lines(path, 0), [])
            self.assertEqual(tail_lines(Path(tmp) / "missing.log", 5), [])

    def test_long_lines_truncated(self):
        with tempdir() as tmp:
            path = Path(tmp) / "x.log"
            path.write_text("x" * 20000 + "\nshort\n")
            lines = tail_lines(path, 2)
            self.assertLess(len(lines[0]), 9000)
            self.assertTrue(lines[0].endswith("[truncated]"))
            self.assertEqual(lines[1], "short")


if __name__ == "__main__":
    unittest.main()
