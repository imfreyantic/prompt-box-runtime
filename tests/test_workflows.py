"""Static validation of the shipped ComfyUI workflows (UI format 0.4 and API format).

The UI format mirrors what ComfyUI frontend 1.57.6 saves (see its default graph and
the official LLM templates): ``inputs`` lists only socket inputs, ``widgets_values``
follows widget order, and INT inputs named ``seed`` get an extra
``control_after_generate`` value right after them.
"""

import json
import re
import unittest

from custom_nodes.aa_prompt_llm.nodes import AAPromptLLM
from tests.helpers import PROMPT_BOX_DIR

WF_DIR = PROMPT_BOX_DIR / "workflows"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}

# Core node shapes in ComfyUI v0.44.0 (comfy_extras/nodes_preview_any.py, nodes.py LoadImage).
CORE_NODES = {
    "PreviewAny": {"inputs": ["source"], "outputs": ["STRING"], "widgets": 0},
    "LoadImage": {"inputs": [], "outputs": ["IMAGE", "MASK"], "widgets": 2},  # image + upload button value
}


def node_widgets(cls) -> list[tuple[str, object]]:
    """(name, spec) for each widget the frontend creates, in order."""
    spec = cls.INPUT_TYPES()
    out = []
    for group in ("required", "optional"):
        for name, (typ, *rest) in spec.get(group, {}).items():
            opts = rest[0] if rest else {}
            if isinstance(typ, list) or typ in WIDGET_TYPES:
                if opts.get("forceInput"):
                    continue
                out.append((name, (typ, opts)))
                if typ == "INT" and (opts.get("control_after_generate") or name in ("seed", "noise_seed")):
                    out.append((name + ".control_after_generate", ("CONTROL", {})))
    return out


def node_sockets(cls) -> list[str]:
    spec = cls.INPUT_TYPES()
    return [name for group in ("required", "optional") for name, (typ, *_rest) in spec.get(group, {}).items()
            if not (isinstance(typ, list) or typ in WIDGET_TYPES)]


class UiWorkflowTests(unittest.TestCase):
    files = ["AA Prompt - Text to Prompt.json", "AA Prompt - Image to Prompt.json"]

    def load(self, name):
        return json.loads((WF_DIR / name).read_text())

    def test_top_level(self):
        for name in self.files:
            wf = self.load(name)
            self.assertEqual(wf["version"], 0.4)
            self.assertRegex(wf["id"], UUID_RE)
            for key in ("last_node_id", "last_link_id", "nodes", "links", "groups", "config", "extra"):
                self.assertIn(key, wf)
            self.assertEqual(wf["last_node_id"], max(n["id"] for n in wf["nodes"]))
            self.assertEqual(wf["last_link_id"], max(link[0] for link in wf["links"]))

    def test_nodes_and_links_are_consistent(self):
        for name in self.files:
            wf = self.load(name)
            nodes = {n["id"]: n for n in wf["nodes"]}
            self.assertEqual(len(nodes), len(wf["nodes"]))
            for n in wf["nodes"]:
                for key in ("id", "type", "pos", "size", "flags", "order", "mode", "properties"):
                    self.assertIn(key, n, (name, n["type"]))
                self.assertEqual(len(n["pos"]), 2)
                self.assertEqual(len(n["size"]), 2)
            for link_id, src, src_slot, dst, dst_slot, typ in wf["links"]:
                out = nodes[src]["outputs"][src_slot]
                inp = nodes[dst]["inputs"][dst_slot]
                self.assertIn(link_id, out["links"])
                self.assertEqual(inp["link"], link_id)
                self.assertEqual(out["type"], typ)
                self.assertIn(inp["type"], (typ, "*"))
            linked = {link[0] for link in wf["links"]}
            for n in wf["nodes"]:
                for inp in n.get("inputs", []):
                    if inp["link"] is not None:
                        self.assertIn(inp["link"], linked)
                for out in n.get("outputs", []):
                    for link_id in out.get("links") or []:
                        self.assertIn(link_id, linked)
            orders = sorted(n["order"] for n in wf["nodes"])
            self.assertEqual(orders, list(range(len(orders))))

    def check_llm_node(self, node):
        widgets = node_widgets(AAPromptLLM)
        values = node["widgets_values"]
        self.assertEqual(len(values), len(widgets))
        self.assertEqual([i["name"] for i in node["inputs"]], node_sockets(AAPromptLLM))
        self.assertEqual([o["name"] for o in node["outputs"]], list(AAPromptLLM.RETURN_NAMES))
        self.assertEqual([o["type"] for o in node["outputs"]], list(AAPromptLLM.RETURN_TYPES))
        for (wname, (typ, opts)), value in zip(widgets, values):
            with self.subTest(widget=wname):
                if typ == "CONTROL":
                    self.assertIn(value, CONTROL_VALUES)
                elif isinstance(typ, list):
                    self.assertIn(value, typ)
                elif typ == "STRING":
                    self.assertIsInstance(value, str)
                elif typ in ("INT", "FLOAT"):
                    self.assertIsInstance(value, (int, float))
                    self.assertGreaterEqual(value, opts.get("min", value))
                    self.assertLessEqual(value, opts.get("max", value))
                    if typ == "INT":
                        self.assertIsInstance(value, int)
        named = dict(zip([w for w, _ in widgets], values))
        self.assertEqual(named["thinking"], "off")
        self.assertEqual(named["max_tokens"], 1024)
        self.assertTrue(named["prompt"].strip())
        self.assertTrue(named["system"].strip())

    def test_text_to_prompt(self):
        wf = self.load(self.files[0])
        types = [n["type"] for n in wf["nodes"]]
        self.assertEqual(sorted(types), ["AAPromptLLM", "PreviewAny"])
        llm = next(n for n in wf["nodes"] if n["type"] == "AAPromptLLM")
        self.check_llm_node(llm)
        self.assertIsNone(llm["inputs"][0]["link"])

    def test_image_to_prompt(self):
        wf = self.load(self.files[1])
        nodes = {n["type"]: n for n in wf["nodes"]}
        self.assertEqual(sorted(nodes), ["AAPromptLLM", "LoadImage", "PreviewAny"])
        self.check_llm_node(nodes["AAPromptLLM"])
        load_link = nodes["LoadImage"]["outputs"][0]["links"][0]
        self.assertEqual(nodes["AAPromptLLM"]["inputs"][0], {"name": "image", "shape": 7, "type": "IMAGE",
                                                             "link": load_link})
        self.assertEqual(nodes["LoadImage"]["widgets_values"], ["example.png", "image"])
        self.assertIn("image", nodes["AAPromptLLM"]["widgets_values"][0].lower())

    def test_core_nodes_match_comfy(self):
        for name in self.files:
            for n in self.load(name)["nodes"]:
                if n["type"] in CORE_NODES:
                    shape = CORE_NODES[n["type"]]
                    self.assertEqual([i["name"] for i in n["inputs"]], shape["inputs"])
                    self.assertEqual([o["name"] for o in n["outputs"]], shape["outputs"])
                    self.assertEqual(len(n["widgets_values"]), shape["widgets"])
                    self.assertEqual(n["properties"]["Node name for S&R"], n["type"])


class ApiWorkflowTests(unittest.TestCase):
    def load(self, name):
        return json.loads((WF_DIR / "api" / name).read_text())

    def check_llm_inputs(self, inputs):
        spec = AAPromptLLM.INPUT_TYPES()
        for name in spec["required"]:
            self.assertIn(name, inputs)
        allowed = set(spec["required"]) | set(spec["optional"])
        self.assertTrue(set(inputs) <= allowed, set(inputs) - allowed)
        self.assertIn(inputs["thinking"], spec["required"]["thinking"][0])

    def test_text_api(self):
        wf = self.load("text-to-prompt.api.json")
        self.assertEqual({k: v["class_type"] for k, v in wf.items()}, {"1": "AAPromptLLM", "2": "PreviewAny"})
        self.check_llm_inputs(wf["1"]["inputs"])
        self.assertEqual(wf["2"]["inputs"], {"source": ["1", 0]})

    def test_image_api(self):
        wf = self.load("image-to-prompt.api.json")
        self.assertEqual({k: v["class_type"] for k, v in wf.items()},
                         {"1": "LoadImage", "2": "AAPromptLLM", "3": "PreviewAny"})
        self.assertEqual(wf["1"]["inputs"], {"image": "example.png"})
        self.check_llm_inputs(wf["2"]["inputs"])
        self.assertEqual(wf["2"]["inputs"]["image"], ["1", 0])
        self.assertEqual(wf["3"]["inputs"], {"source": ["2", 0]})

    def test_api_and_ui_values_agree(self):
        pairs = [("AA Prompt - Text to Prompt.json", "text-to-prompt.api.json", "1"),
                 ("AA Prompt - Image to Prompt.json", "image-to-prompt.api.json", "2")]
        names = [w for w, _ in node_widgets(AAPromptLLM)]
        for ui_name, api_name, node_id in pairs:
            ui = json.loads((WF_DIR / ui_name).read_text())
            llm = next(n for n in ui["nodes"] if n["type"] == "AAPromptLLM")
            ui_values = {k: v for k, v in zip(names, llm["widgets_values"]) if "." not in k}
            api_values = {k: v for k, v in self.load(api_name)[node_id]["inputs"].items() if k != "image"}
            self.assertEqual(ui_values, api_values)


if __name__ == "__main__":
    unittest.main()
