import base64
import io
import json
import socket
import unittest
import urllib.error
from unittest import mock

from custom_nodes.aa_prompt_llm import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from custom_nodes.aa_prompt_llm import llm_client as L
from custom_nodes.aa_prompt_llm.nodes import AAPromptLLM

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover
    np = None
    Image = None


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingOpener:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append((req, timeout))
        if self.error is not None:
            raise self.error
        return FakeResponse(json.dumps(self.response).encode())


OK_RESPONSE = {"choices": [{"message": {"content": " A neon ramen stall. ", "reasoning_content": "thinking..."},
                            "finish_reason": "stop"}]}


class PayloadTests(unittest.TestCase):
    def test_thinking_off(self):
        p = L.build_payload(prompt="hi", thinking="off", max_tokens=64, temperature=0.5, top_p=0.9, top_k=10,
                            presence_penalty=1.0, seed=42)
        self.assertEqual(p["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(p["messages"], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertEqual((p["max_tokens"], p["temperature"], p["top_p"], p["top_k"], p["presence_penalty"], p["seed"]),
                         (64, 0.5, 0.9, 10, 1.0, 42))
        self.assertIs(p["stream"], False)
        self.assertEqual(p["model"], "qwen3.8-27b-aggressive")

    def test_thinking_levels(self):
        for level in ("low", "medium", "xhigh"):
            p = L.build_payload(prompt="x", thinking=level)
            self.assertEqual(p["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": level})
        with self.assertRaises(ValueError):
            L.build_payload(prompt="x", thinking="high")

    def test_system_and_images(self):
        urls = ["data:image/png;base64,AAA", "data:image/png;base64,BBB"]
        p = L.build_payload(prompt="describe", system="be terse", image_urls=urls)
        self.assertEqual(p["messages"][0], {"role": "system", "content": "be terse"})
        content = p["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "describe"})
        self.assertEqual(content[1:], [{"type": "image_url", "image_url": {"url": u}} for u in urls])

    def test_blank_system_is_omitted(self):
        for system in (None, "", "   \n"):
            p = L.build_payload(prompt="x", system=system)
            self.assertEqual([m["role"] for m in p["messages"]], ["user"])


@unittest.skipIf(np is None, "numpy/Pillow not installed")
class ImageEncodingTests(unittest.TestCase):
    def decode(self, url: str):
        prefix = "data:image/png;base64,"
        self.assertTrue(url.startswith(prefix))
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(url[len(prefix):]))))

    def test_float_rgb_roundtrip(self):
        img = np.random.default_rng(0).random((5, 7, 3), dtype=np.float32)
        out = self.decode(L.encode_png_data_url(img))
        self.assertEqual(out.shape, (5, 7, 3))
        self.assertTrue(np.array_equal(out, (img * 255).round().astype(np.uint8)))

    def test_rgba_gray_and_uint8(self):
        rgba = np.ones((4, 4, 4), dtype=np.float32) * 0.5
        self.assertEqual(self.decode(L.encode_png_data_url(rgba)).shape, (4, 4, 3))
        gray = np.zeros((4, 4, 1), dtype=np.float32)
        self.assertEqual(self.decode(L.encode_png_data_url(gray)).shape, (4, 4, 3))
        u8 = np.full((2, 3, 3), 200, dtype=np.uint8)
        self.assertTrue((self.decode(L.encode_png_data_url(u8)) == 200).all())
        clipped = np.full((2, 2, 3), 1.7, dtype=np.float32)
        self.assertTrue((self.decode(L.encode_png_data_url(clipped)) == 255).all())

    def test_batch_gives_one_url_per_image(self):
        batch = np.zeros((3, 8, 8, 3), dtype=np.float32)
        self.assertEqual(len(L.images_to_data_urls(batch)), 3)
        self.assertEqual(len(L.images_to_data_urls(batch[0])), 1)
        self.assertEqual(L.images_to_data_urls(None), [])

    def test_torch_like_tensor(self):
        class FakeTensor:
            def __init__(self, arr):
                self.arr = arr

            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def numpy(self):
                return self.arr

        urls = L.images_to_data_urls(FakeTensor(np.zeros((2, 4, 4, 3), dtype=np.float32)))
        self.assertEqual(len(urls), 2)

    def test_node_sends_every_image(self):
        opener = RecordingOpener(OK_RESPONSE)
        batch = np.zeros((2, 4, 4, 3), dtype=np.float32)
        with mock.patch.object(L, "_OPENER", mock.Mock(open=opener)):
            AAPromptLLM().generate("describe", "off", 128, 0.7, 0.8, 20, 1.5, 1, system="", image=batch)
        sent = json.loads(opener.calls[0][0].data)
        parts = sent["messages"][-1]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image_url", "image_url"])


class HttpTests(unittest.TestCase):
    def test_post_success(self):
        opener = RecordingOpener(OK_RESPONSE)
        out = L.post_chat_completion({"a": 1}, url="http://127.0.0.1:8080/v1/chat/completions", timeout=900,
                                     urlopen=opener)
        req, timeout = opener.calls[0]
        self.assertEqual(timeout, 900)
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.full_url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(req.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(req.data), {"a": 1})
        self.assertEqual(L.parse_completion(out), ("A neon ramen stall.", "thinking..."))

    def test_unreachable_server_message(self):
        opener = RecordingOpener(error=urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")))
        with self.assertRaises(L.LLMServerError) as cm:
            L.post_chat_completion({}, urlopen=opener)
        self.assertIn("Cannot reach llama-server", str(cm.exception))

    def test_http_error_message_includes_server_detail(self):
        body = io.BytesIO(json.dumps({"error": {"code": 400, "message": "context too long"}}).encode())
        err = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, body)
        with self.assertRaises(L.LLMServerError) as cm:
            L.post_chat_completion({}, urlopen=RecordingOpener(error=err))
        self.assertIn("HTTP 400", str(cm.exception))
        self.assertIn("context too long", str(cm.exception))

    def test_loading_and_timeout_messages(self):
        err = urllib.error.HTTPError("http://x", 503, "Unavailable", {}, io.BytesIO(b'{"error":{"message":"Loading model"}}'))
        with self.assertRaises(L.LLMServerError) as cm:
            L.post_chat_completion({}, urlopen=RecordingOpener(error=err))
        self.assertIn("still loading", str(cm.exception))
        with self.assertRaises(L.LLMServerError) as cm:
            L.post_chat_completion({}, timeout=5, urlopen=RecordingOpener(error=socket.timeout("timed out")))
        self.assertIn("within 5 s", str(cm.exception))

    def test_parse_completion_variants(self):
        self.assertEqual(L.parse_completion({"choices": [{"message": {"content": None}}]}), ("", ""))
        self.assertEqual(L.parse_completion({"choices": [{"message": {"content": [{"type": "text", "text": "a"},
                                                                                   {"type": "text", "text": "b"}]}}]}),
                         ("ab", ""))
        with self.assertRaises(L.LLMServerError):
            L.parse_completion({"error": "x"})

    def test_default_opener_bypasses_proxies(self):
        # build_opener(ProxyHandler({})) replaces the env-reading default ProxyHandler with one that
        # handles no scheme, so no proxy handler ends up registered at all.
        import urllib.request
        self.assertFalse(any(isinstance(h, urllib.request.ProxyHandler) for h in L._OPENER.handlers))


class NodeDefinitionTests(unittest.TestCase):
    def test_registration(self):
        self.assertIs(NODE_CLASS_MAPPINGS["AAPromptLLM"], AAPromptLLM)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["AAPromptLLM"], "AA Prompt LLM (Diddy)")
        self.assertEqual(AAPromptLLM.CATEGORY, "AA/Prompt")
        self.assertEqual(AAPromptLLM.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(AAPromptLLM.FUNCTION, "generate")

    def test_inputs(self):
        spec = AAPromptLLM.INPUT_TYPES()
        req, opt = spec["required"], spec["optional"]
        self.assertEqual(list(req), ["prompt", "thinking", "max_tokens", "temperature", "top_p", "top_k",
                                     "presence_penalty", "seed"])
        self.assertEqual(list(opt), ["system", "image"])
        self.assertEqual(req["prompt"][0], "STRING")
        self.assertTrue(req["prompt"][1]["multiline"])
        self.assertEqual(req["thinking"][0], ["off", "low", "medium", "xhigh"])
        self.assertEqual(req["thinking"][1]["default"], "off")
        self.assertEqual((req["max_tokens"][0], req["max_tokens"][1]["min"], req["max_tokens"][1]["max"],
                          req["max_tokens"][1]["default"]), ("INT", 16, 32768, 1024))
        self.assertEqual((req["temperature"][1]["min"], req["temperature"][1]["max"], req["temperature"][1]["default"]),
                         (0.0, 2.0, 0.7))
        self.assertEqual(req["top_p"][1]["default"], 0.8)
        self.assertEqual(req["top_k"][1]["default"], 20)
        self.assertEqual((req["presence_penalty"][1]["min"], req["presence_penalty"][1]["max"],
                          req["presence_penalty"][1]["default"]), (0.0, 2.0, 1.5))
        self.assertEqual((req["seed"][1]["min"], req["seed"][1]["max"], req["seed"][1]["default"]),
                         (0, 2**31 - 1, 0))
        self.assertEqual(opt["system"][0], "STRING")
        self.assertTrue(opt["system"][1]["multiline"])
        self.assertEqual(opt["image"][0], "IMAGE")

    def test_generate_returns_text_reasoning_and_ui(self):
        opener = RecordingOpener(OK_RESPONSE)
        with mock.patch.object(L, "_OPENER", mock.Mock(open=opener)):
            out = AAPromptLLM().generate("idea", "medium", 512, 0.6, 0.9, 30, 1.2, 7, system="sys")
        self.assertEqual(out["result"], ("A neon ramen stall.", "thinking..."))
        self.assertEqual(out["ui"], {"text": ["A neon ramen stall."]})
        sent = json.loads(opener.calls[0][0].data)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": "medium"})
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(opener.calls[0][1], 900)
        self.assertEqual(opener.calls[0][0].full_url, "http://127.0.0.1:8080/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
