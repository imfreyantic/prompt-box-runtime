import re
import unittest

from box import constants as C


class PinTests(unittest.TestCase):
    def test_llama_and_patch_pins(self):
        self.assertEqual(C.LLAMA_COMMIT, "4df29be4f4c3673f428170fda944a5b19f743bb8")
        self.assertEqual(C.LLAMA_REPO_URL, "https://github.com/ggml-org/llama.cpp")
        self.assertEqual(C.PATCH_FILE.name, "HauhauCS-FastMTP-llama.cpp.patch")
        self.assertEqual(C.PATCH_FILE.size, 2445)
        self.assertEqual(C.PATCH_FILE.sha256, "981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615")
        self.assertEqual(
            C.LLAMA_CMAKE_ARGS,
            ("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=120", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"),
        )
        self.assertEqual(C.LLAMA_BUILD_TARGET, "llama-server")

    def test_comfy_pins(self):
        self.assertEqual(C.COMFY_TAG, "v0.44.0")
        self.assertEqual(C.COMFY_COMMIT, "2c75dc1b379a59e0880619d029607af5169a62ef")
        self.assertEqual(C.COMFY_REPO_URL, "https://github.com/Comfy-Org/ComfyUI")

    def test_hf_release_pins_and_url_form(self):
        self.assertEqual(C.HF_REPO, "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF")
        self.assertEqual(C.HF_REVISION, "993a5971fda8f30dd1b7eb2654792ba4415c7460")
        self.assertEqual(
            C.hf_url("x.gguf"),
            "https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/"
            "993a5971fda8f30dd1b7eb2654792ba4415c7460/x.gguf",
        )

    def test_model_manifest(self):
        expected = {
            "Q8_K_P": (31457990784, "4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc2"),
            "Q6_K_P": (25924151424, "70c8139f4dcd05d3818bc9843cde7301faa44f66321e5966965929baa9c24a6a"),
            "Q5_K_P": (20218177664, "a21e22af885bd2f7c430a72b550a78f5445966e6a72d3143b0950baa0bd6d411"),
            "Q4_K_P": (17923393664, "ba36dc3c2b2ff5e0aa5d71092a8894546996a6a119ae391803dda07cdc08516d"),
        }
        self.assertEqual(set(C.QUANT_FILES), set(expected))
        for quant, (size, sha) in expected.items():
            spec = C.QUANT_FILES[quant]
            self.assertEqual(spec.name, f"Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-{quant}.gguf")
            self.assertEqual((spec.size, spec.sha256), (size, sha))
        self.assertEqual(
            (C.MMPROJ_FILE.name, C.MMPROJ_FILE.size, C.MMPROJ_FILE.sha256),
            ("mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf", 931146624,
             "5681b690bcb8eb10cd28d62d078cb4e01521a3ea4880a3fc7d54de72de2dd142"),
        )
        self.assertEqual(
            (C.FASTMTP_FILE.name, C.FASTMTP_FILE.size, C.FASTMTP_FILE.sha256),
            ("Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf", 903453952,
             "115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b"),
        )
        for spec in [*C.QUANT_FILES.values(), C.MMPROJ_FILE, C.FASTMTP_FILE, C.PATCH_FILE]:
            self.assertRegex(spec.sha256, re.compile(r"^[0-9a-f]{64}$"))

    def test_limits(self):
        self.assertEqual((C.CTX_MIN, C.CTX_DEFAULT, C.CTX_MAX), (8192, 131072, 262144))
        self.assertEqual(C.MAX_BODY_BYTES, 32 * 1024 * 1024)
        self.assertEqual((C.API_PORT, C.UI_PORT, C.LLAMA_PORT, C.COMFY_PORT), (8443, 8444, 8080, 8188))
        self.assertEqual(C.LOG_SOURCES, ("box", "llama", "comfy", "bootstrap"))
        self.assertEqual((C.LOG_LINES_DEFAULT, C.LOG_LINES_MAX), (200, 1000))
        self.assertEqual((C.TICKET_MAX_AGE_S, C.SESSION_TTL_S, C.SESSION_COOKIE), (300, 43200, "pb_comfy"))


if __name__ == "__main__":
    unittest.main()
