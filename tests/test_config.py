import os
import stat
import unittest
from pathlib import Path

from box import constants as C
from box.config import FatalConfigError, load_config, validate_token, write_tls_files
from tests.helpers import TOKEN, b64, tempdir

CERT = b"-----BEGIN CERTIFICATE-----\nMIIBfake\n-----END CERTIFICATE-----\n"
KEY = b"-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n-----END PRIVATE KEY-----\n"


def env(**overrides):
    base = {"PB_TOKEN": TOKEN, "PB_TLS_CERT_B64": b64(CERT), "PB_TLS_KEY_B64": b64(KEY), "PB_ROOT": "/tmp/pb-x"}
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


class TokenTests(unittest.TestCase):
    def test_valid_token(self):
        self.assertEqual(validate_token(TOKEN), TOKEN)

    def test_rejects_missing_or_malformed(self):
        for bad in (None, "", "abc", TOKEN.upper(), TOKEN[:-1], TOKEN + "0", "g" * 64, " " + TOKEN[1:]):
            with self.subTest(bad=bad):
                with self.assertRaises(FatalConfigError):
                    validate_token(bad)

    def test_load_config_refuses_bad_token(self):
        with self.assertRaises(FatalConfigError):
            load_config(env(PB_TOKEN="nothex"))
        with self.assertRaises(FatalConfigError):
            load_config(env(PB_TOKEN=None))

    def test_tls_material_required_and_must_be_pem(self):
        with self.assertRaises(FatalConfigError):
            load_config(env(PB_TLS_CERT_B64=None))
        with self.assertRaises(FatalConfigError):
            load_config(env(PB_TLS_KEY_B64="!!!notbase64"))
        with self.assertRaises(FatalConfigError):
            load_config(env(PB_TLS_KEY_B64=b64(b"just text")))

    def test_base64_with_line_wrapping_is_accepted(self):
        wrapped = "\n".join(b64(KEY)[i:i + 20] for i in range(0, len(b64(KEY)), 20))
        cfg = load_config(env(PB_TLS_KEY_B64=wrapped))
        self.assertEqual(cfg.tls_key_pem, KEY)


class SelectionTests(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config(env())
        self.assertEqual(cfg.errors, [])
        self.assertEqual(cfg.quant, "Q8_K_P")
        self.assertEqual(cfg.ctx, 131072)
        self.assertEqual(cfg.mtp, "fastmtp")
        self.assertEqual(cfg.text_model, C.QUANT_FILES["Q8_K_P"])
        self.assertFalse(cfg.use_prebuilt)

    def test_each_allowed_quant_selects_its_file(self):
        for quant, spec in C.QUANT_FILES.items():
            with self.subTest(quant=quant):
                cfg = load_config(env(PB_QUANT=quant))
                self.assertEqual(cfg.errors, [])
                self.assertEqual(cfg.text_model, spec)
                self.assertIn(quant, spec.name)

    def test_invalid_quant_is_reported(self):
        for bad in ("Q3_K_P", "IQ4_XS", "q8_k_p", "Q8_0"):
            with self.subTest(bad=bad):
                cfg = load_config(env(PB_QUANT=bad))
                self.assertIsNone(cfg.text_model)
                self.assertTrue(any("PB_QUANT" in e for e in cfg.errors))

    def test_ctx_bounds(self):
        for ok in ("8192", "131072", "262144", " 65536 "):
            with self.subTest(ok=ok):
                cfg = load_config(env(PB_CTX=ok))
                self.assertEqual(cfg.errors, [])
                self.assertEqual(cfg.ctx, int(ok))
        for bad in ("8191", "262145", "0", "-1", "abc", "1e5"):
            with self.subTest(bad=bad):
                self.assertTrue(any("PB_CTX" in e for e in load_config(env(PB_CTX=bad)).errors))

    def test_mtp_modes(self):
        for mode in ("fastmtp", "embedded", "off"):
            self.assertEqual(load_config(env(PB_MTP=mode)).mtp, mode)
        self.assertTrue(any("PB_MTP" in e for e in load_config(env(PB_MTP="eagle")).errors))

    def test_prebuilt_needs_both_values(self):
        sha = "ab" * 32
        cfg = load_config(env(PB_LLAMA_PREBUILT_URL="https://example.com/llama.tgz", PB_LLAMA_PREBUILT_SHA256=sha))
        self.assertTrue(cfg.use_prebuilt)
        cfg = load_config(env(PB_LLAMA_PREBUILT_URL="https://example.com/llama.tgz"))
        self.assertFalse(cfg.use_prebuilt)
        self.assertTrue(cfg.warnings)
        cfg = load_config(env(PB_LLAMA_PREBUILT_URL="https://example.com/x", PB_LLAMA_PREBUILT_SHA256="zz"))
        self.assertTrue(any("SHA256" in e for e in cfg.errors))

    def test_secrets_include_token_key_and_prebuilt_url(self):
        url = "https://bucket.example.com/llama.tgz?X-Amz-Signature=deadbeefdeadbeef"
        cfg = load_config(env(PB_LLAMA_PREBUILT_URL=url, PB_LLAMA_PREBUILT_SHA256="ab" * 32))
        secrets = cfg.secrets()
        self.assertIn(TOKEN, secrets)
        self.assertIn(b64(KEY), secrets)
        self.assertIn("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7", secrets)
        self.assertIn(url, secrets)


class TlsFileTests(unittest.TestCase):
    def test_written_with_0600(self):
        with tempdir() as tmp:
            cfg = load_config(env(PB_ROOT=tmp))
            cert, key = write_tls_files(cfg)
            self.assertEqual(cert.read_bytes(), CERT)
            self.assertEqual(key.read_bytes(), KEY)
            for path in (cert, key):
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(Path(tmp) / "tls").st_mode), 0o700)
            # rewriting an existing, looser file tightens it again
            os.chmod(key, 0o644)
            write_tls_files(cfg)
            self.assertEqual(stat.S_IMODE(os.stat(key).st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
