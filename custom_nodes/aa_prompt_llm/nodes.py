"""ComfyUI node that sends a prompt (and optional images) to the local llama-server."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .llm_client import (
    DEFAULT_TIMEOUT_S,
    DEFAULT_URL,
    THINKING_LEVELS,
    build_payload,
    images_to_data_urls,
    parse_completion,
    post_chat_completion,
)

log = logging.getLogger("aa_prompt_llm")


class AAPromptLLM:
    """Qwen3.8 27B (llama-server on this box) as a prompt writer / image describer."""

    DESCRIPTION = (
        "Sends the prompt (plus every image of the optional IMAGE batch) to the prompt box's "
        "llama-server and returns the answer and the model's reasoning."
    )
    CATEGORY = "AA/Prompt"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning")
    OUTPUT_TOOLTIPS = ("The model's answer.", "The model's reasoning (empty when thinking is off).")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False,
                                      "tooltip": "User message sent to the model."}),
                "thinking": (list(THINKING_LEVELS), {"default": "off",
                                                     "tooltip": "off = direct answer; low/medium/xhigh = reasoning effort."}),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 32768, "step": 1,
                                       "tooltip": "Maximum generated tokens (reasoning tokens count too)."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 1000, "step": 1}),
                "presence_penalty": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False,
                                      "tooltip": "Optional system message."}),
                "image": ("IMAGE", {"tooltip": "Optional image batch; every image is sent."}),
            },
        }

    def generate(
        self,
        prompt: str,
        thinking: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence_penalty: float,
        seed: int,
        system: Optional[str] = None,
        image: Any = None,
    ) -> dict[str, Any]:
        image_urls = images_to_data_urls(image) if image is not None else []
        payload = build_payload(
            prompt=prompt,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            seed=seed,
            system=system,
            image_urls=image_urls,
        )
        url = os.environ.get("AA_PROMPT_LLM_URL", DEFAULT_URL)
        response = post_chat_completion(payload, url=url, timeout=DEFAULT_TIMEOUT_S)
        text, reasoning = parse_completion(response)
        try:
            if response["choices"][0].get("finish_reason") == "length":
                log.warning("AA Prompt LLM: output hit max_tokens=%d and may be cut off", max_tokens)
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        return {"ui": {"text": [text]}, "result": (text, reasoning)}


NODE_CLASS_MAPPINGS = {"AAPromptLLM": AAPromptLLM}
NODE_DISPLAY_NAME_MAPPINGS = {"AAPromptLLM": "AA Prompt LLM (Qwen3.8)"}
