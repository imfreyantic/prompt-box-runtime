"""Pinned versions, model manifest and fixed runtime parameters.

Everything the box downloads or builds is pinned here so a given runtime commit
always produces the same machine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedFile:
    name: str
    size: int
    sha256: str


# --- Hugging Face model release -------------------------------------------------
HF_BASE = "https://huggingface.co"
HF_REPO = "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
HF_REVISION = "993a5971fda8f30dd1b7eb2654792ba4415c7460"


def hf_url(filename: str, base: str = HF_BASE) -> str:
    return f"{base}/{HF_REPO}/resolve/{HF_REVISION}/{filename}"


QUANT_FILES: dict[str, PinnedFile] = {
    "Q8_K_P": PinnedFile(
        "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
        31457990784,
        "4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc2",
    ),
    "Q6_K_P": PinnedFile(
        "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf",
        25924151424,
        "70c8139f4dcd05d3818bc9843cde7301faa44f66321e5966965929baa9c24a6a",
    ),
    "Q5_K_P": PinnedFile(
        "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf",
        20218177664,
        "a21e22af885bd2f7c430a72b550a78f5445966e6a72d3143b0950baa0bd6d411",
    ),
    "Q4_K_P": PinnedFile(
        "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
        17923393664,
        "ba36dc3c2b2ff5e0aa5d71092a8894546996a6a119ae391803dda07cdc08516d",
    ),
}
DEFAULT_QUANT = "Q8_K_P"

MMPROJ_FILE = PinnedFile(
    "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
    931146624,
    "5681b690bcb8eb10cd28d62d078cb4e01521a3ea4880a3fc7d54de72de2dd142",
)
FASTMTP_FILE = PinnedFile(
    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf",
    903453952,
    "115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b",
)

# Parallel range connections per file (text model is the big one).
TEXT_MODEL_CONNECTIONS = 16
SIDECAR_CONNECTIONS = 4

# --- llama.cpp --------------------------------------------------------------------
LLAMA_REPO_URL = "https://github.com/ggml-org/llama.cpp"
LLAMA_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"
PATCH_FILE = PinnedFile(
    "HauhauCS-FastMTP-llama.cpp.patch",
    2445,
    "981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615",
)
LLAMA_CMAKE_ARGS: tuple[str, ...] = (
    "-DGGML_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=120",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_CURL=OFF",
)
LLAMA_BUILD_TARGET = "llama-server"
MODEL_ALIAS = "qwen3.8-27b-aggressive"

# --- ComfyUI ----------------------------------------------------------------------
COMFY_REPO_URL = "https://github.com/Comfy-Org/ComfyUI"
COMFY_TAG = "v0.44.0"
COMFY_COMMIT = "2c75dc1b379a59e0880619d029607af5169a62ef"
CUSTOM_NODE_PACKAGE = "aa_prompt_llm"
CUSTOM_NODE_CLASS = "AAPromptLLM"
CORE_PREVIEW_NODE = "PreviewAny"

# --- Ports and limits ---------------------------------------------------------------
LOCAL_HOST = "127.0.0.1"
LLAMA_PORT = 8080
COMFY_PORT = 8188
API_PORT = 8443
UI_PORT = 8444

CTX_DEFAULT = 131072
CTX_MIN = 8192
CTX_MAX = 262144
MTP_MODES: tuple[str, ...] = ("fastmtp", "embedded", "off")
DEFAULT_MTP = "fastmtp"

MAX_BODY_BYTES = 32 * 1024 * 1024
LOG_SOURCES: tuple[str, ...] = ("box", "llama", "comfy", "bootstrap")
LOG_LINES_DEFAULT = 200
LOG_LINES_MAX = 1000

TICKET_MAX_AGE_S = 300
SESSION_TTL_S = 12 * 3600
SESSION_COOKIE = "pb_comfy"

MAX_SERVICE_RESTARTS = 3
