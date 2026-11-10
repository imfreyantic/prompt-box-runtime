"""Pure helpers for the AA Prompt LLM node: payload building, image encoding, HTTP call.

Only the standard library is needed at import time; numpy/Pillow are imported lazily
when images are encoded (ComfyUI ships both).
"""

from __future__ import annotations

import base64
import io
import json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Optional, Sequence

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_TIMEOUT_S = 900
MODEL_ALIAS = "qwen3.8-27b-aggressive"
THINKING_LEVELS = ("off", "low", "medium", "xhigh")

# Never route the loopback llama-server call through an HTTP proxy from the environment.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LLMServerError(RuntimeError):
    """Raised with a message that is shown to the ComfyUI user."""


def chat_template_kwargs(thinking: str) -> dict[str, Any]:
    if thinking == "off":
        return {"enable_thinking": False}
    if thinking in ("low", "medium", "xhigh"):
        return {"enable_thinking": True, "reasoning_effort": thinking}
    raise ValueError(f"thinking must be one of {THINKING_LEVELS}, got {thinking!r}")


def build_messages(prompt: str, system: Optional[str] = None, image_urls: Sequence[str] = ()) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
    messages.append({"role": "user", "content": content})
    return messages


def build_payload(
    *,
    prompt: str,
    thinking: str = "off",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    seed: int = 0,
    system: Optional[str] = None,
    image_urls: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "model": MODEL_ALIAS,
        "messages": build_messages(prompt, system, image_urls),
        "stream": False,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "presence_penalty": float(presence_penalty),
        "seed": int(seed),
        "chat_template_kwargs": chat_template_kwargs(thinking),
    }


def encode_png_data_url(image: Any) -> str:
    """Encode one HxWxC image (float 0..1 or uint8) as a PNG data URL."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected an HxWxC image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    channels = arr.shape[2]
    if channels == 1:
        pil = Image.fromarray(np.ascontiguousarray(arr[:, :, 0])).convert("RGB")
    elif channels in (3, 4):
        pil = Image.fromarray(np.ascontiguousarray(arr[:, :, :3]))  # alpha dropped
    else:
        raise ValueError(f"unsupported channel count {channels}")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def images_to_data_urls(images: Any) -> list[str]:
    """ComfyUI IMAGE batch ([B,H,W,C] torch tensor) -> one data URL per image."""
    if images is None:
        return []
    if hasattr(images, "detach"):
        images = images.detach().cpu().float().numpy()
    import numpy as np

    arr = np.asarray(images)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"expected an IMAGE batch [B,H,W,C], got shape {arr.shape}")
    return [encode_png_data_url(img) for img in arr]


def _server_error_detail(body: bytes) -> str:
    try:
        data = json.loads(body)
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err)
        return str(err)
    except (ValueError, AttributeError):
        return body.decode("utf-8", "replace")[:500]


def post_chat_completion(
    payload: dict[str, Any],
    url: str = DEFAULT_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
    urlopen: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    opener = urlopen or _OPENER.open
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with opener(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read() or b""
        except Exception:  # no body attached
            raw = b""
        detail = _server_error_detail(raw)
        if exc.code == 503:
            raise LLMServerError(f"llama-server is still loading the model (HTTP 503): {detail}") from None
        raise LLMServerError(f"llama-server returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise LLMServerError(f"llama-server did not answer within {timeout:.0f} s") from None
        raise LLMServerError(
            f"Cannot reach llama-server at {url} ({reason}). The prompt box may still be downloading or "
            "loading the model - check the box status in the app and try again."
        ) from None
    except (socket.timeout, TimeoutError):
        raise LLMServerError(f"llama-server did not answer within {timeout:.0f} s") from None
    try:
        return json.loads(body)
    except ValueError:
        raise LLMServerError("llama-server returned a response that is not JSON") from None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))
    return str(value)


def parse_completion(response: dict[str, Any]) -> tuple[str, str]:
    """(text, reasoning) from an OpenAI-style chat completion."""
    try:
        message = response["choices"][0].get("message") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        raise LLMServerError(f"unexpected llama-server response: {str(response)[:300]}") from None
    return _text(message.get("content")).strip(), _text(message.get("reasoning_content")).strip()
