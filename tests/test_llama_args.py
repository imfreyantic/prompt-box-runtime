import unittest
from pathlib import Path

from box.services import comfy_args, llama_server_args

BIN = Path("llama.cpp/build/bin/llama-server")
MODEL = Path("/m/model.gguf")
MMPROJ = Path("/m/mmproj.gguf")
DRAFT = Path("/m/fastmtp.gguf")

COMMON_TAIL = (
    "--parallel 1 --batch-size 2048 --ubatch-size 512 --n-gpu-layers all --split-mode none --flash-attn on "
    "--no-mmap --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 --jinja "
    "--reasoning on --reasoning-effort xhigh --reasoning-preserve --reasoning-format deepseek "
    "--alias qwen3.8-27b-aggressive --host 127.0.0.1 --port 8080"
)
SPEC_FLAGS = {"--spec-draft-model", "--spec-draft-ngl", "--spec-type", "--spec-draft-n-max", "--spec-draft-p-min"}


def build(mtp: str, ctx: int = 131072) -> list[str]:
    return llama_server_args(BIN, MODEL, MMPROJ, DRAFT, ctx, mtp)


class LlamaArgsTests(unittest.TestCase):
    def test_fastmtp_matches_contract_exactly(self):
        expected = (
            "llama.cpp/build/bin/llama-server --model /m/model.gguf --mmproj /m/mmproj.gguf "
            "--spec-draft-model /m/fastmtp.gguf --spec-draft-ngl all --spec-type draft-mtp --spec-draft-n-max 3 "
            "--spec-draft-p-min 0 --ctx-size 131072 " + COMMON_TAIL
        )
        self.assertEqual(" ".join(build("fastmtp")), expected)

    def test_embedded_drops_draft_model_and_uses_depth_2(self):
        args = build("embedded")
        self.assertNotIn("--spec-draft-model", args)
        self.assertNotIn("--spec-draft-ngl", args)
        self.assertNotIn(str(DRAFT), args)
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "2")
        self.assertEqual(
            " ".join(args),
            "llama.cpp/build/bin/llama-server --model /m/model.gguf --mmproj /m/mmproj.gguf "
            "--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0 --ctx-size 131072 " + COMMON_TAIL,
        )

    def test_off_has_no_speculative_flags(self):
        args = build("off")
        self.assertFalse(SPEC_FLAGS & set(args))
        self.assertEqual(
            " ".join(args),
            "llama.cpp/build/bin/llama-server --model /m/model.gguf --mmproj /m/mmproj.gguf --ctx-size 131072 "
            + COMMON_TAIL,
        )

    def test_ctx_value_and_bounds(self):
        for ctx in (8192, 65536, 262144):
            args = build("fastmtp", ctx)
            self.assertEqual(args[args.index("--ctx-size") + 1], str(ctx))
        for bad in (8191, 262145, 0):
            with self.assertRaises(ValueError):
                build("fastmtp", bad)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            build("eagle3")

    def test_listens_on_loopback_only(self):
        for mode in ("fastmtp", "embedded", "off"):
            args = build(mode)
            self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
            self.assertEqual(args[args.index("--port") + 1], "8080")

    def test_comfy_args(self):
        self.assertEqual(
            comfy_args(Path("/w/venv-comfy/bin/python")),
            ["/w/venv-comfy/bin/python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
        )


if __name__ == "__main__":
    unittest.main()
