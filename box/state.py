"""In-memory box state, rendered by GET /health."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

SERVICE_LABELS = {"llama": "llama-server", "comfy": "ComfyUI"}


def now_ms(clock: Callable[[], float] = time.time) -> int:
    return int(clock() * 1000)


@dataclass
class FileProgress:
    name: str
    total: int
    done: int = 0
    verified: bool = False
    phase: str = "pending"  # pending | downloading | verifying | done | error

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "doneBytes": self.total if self.verified else min(self.done, self.total),
            "totalBytes": self.total,
            "verified": self.verified,
        }


class BoxState:
    def __init__(self, versions: dict[str, Any], clock: Callable[[], float] = time.time):
        self._clock = clock
        self.started_at = now_ms(clock)
        self.ready_at: Optional[int] = None
        self.errors: list[str] = []
        self.files: list[FileProgress] = []
        self.build_state = "pending"
        self.build_percent: Optional[int] = None
        self.comfy_state = "pending"
        self.services: dict[str, str] = {"llama": "stopped", "comfy": "stopped"}
        self.versions = versions
        self.gpu: Optional[dict[str, Any]] = None
        self.notes: dict[str, str] = {}

    # --- mutation helpers -------------------------------------------------------------
    def fail(self, where: str, message: str) -> None:
        text = f"{where}: {message}" if where else message
        if text not in self.errors:
            self.errors.append(text)

    def set_service(self, name: str, status: str) -> None:
        self.services[name] = status
        self._maybe_ready()

    def set_note(self, key: str, text: Optional[str]) -> None:
        if text:
            self.notes[key] = text
        else:
            self.notes.pop(key, None)

    def _maybe_ready(self) -> None:
        if self.ready_at is None and not self.errors and all(s == "ready" for s in self.services.values()):
            self.ready_at = now_ms(self._clock)

    # --- derived fields ---------------------------------------------------------------
    @property
    def download_complete(self) -> bool:
        return bool(self.files) and all(f.verified for f in self.files)

    @property
    def state(self) -> str:
        if self.errors:
            return "error"
        if self.ready_at is not None:
            return "ready"
        return "installing"

    @property
    def stage(self) -> str:
        if self.errors:
            return "error"
        if self.ready_at is not None:
            return "ready"
        if not self.download_complete:
            return "download"
        if self.build_state != "done":
            return "build"
        if self.comfy_state != "done":
            return "comfy"
        if self.services["llama"] != "ready":
            return "llama_start"
        if self.services["comfy"] != "ready":
            return "comfy_start"
        return "ready"

    def download_totals(self) -> tuple[int, int]:
        done = sum(f.as_json()["doneBytes"] for f in self.files)
        total = sum(f.total for f in self.files)
        return done, total

    @property
    def message(self) -> str:
        if self.errors:
            return self.errors[0]
        parts: list[str] = []
        if self.files and not self.download_complete:
            verifying = [f.name for f in self.files if f.phase == "verifying"]
            done, total = self.download_totals()
            pct = int(done * 100 / total) if total else 0
            text = f"Downloading model files {done / 1e9:.1f}/{total / 1e9:.1f} GB ({pct}%)"
            if verifying:
                text += "; verifying SHA-256 of " + ", ".join(verifying)
            parts.append(text)
        if self.build_state == "running":
            parts.append("Building llama.cpp" + (f" ({self.build_percent}%)" if self.build_percent is not None else ""))
        if self.comfy_state == "running":
            parts.append("Installing ComfyUI")
        for name, status in self.services.items():
            if status == "starting":
                parts.append(f"Starting {SERVICE_LABELS[name]}")
        parts.extend(self.notes.values())
        if not parts:
            return "Ready" if self.ready_at is not None else "Starting"
        return "; ".join(parts)

    def to_health(self) -> dict[str, Any]:
        done, total = self.download_totals()
        return {
            "state": self.state,
            "stage": self.stage,
            "message": self.message,
            "error": " | ".join(self.errors) if self.errors else None,
            "startedAt": self.started_at,
            "readyAt": self.ready_at,
            "progress": {
                "download": {
                    "doneBytes": done,
                    "totalBytes": total,
                    "files": [f.as_json() for f in self.files],
                },
                "build": {"state": self.build_state, "percent": self.build_percent},
                "comfy": {"state": self.comfy_state},
            },
            "services": dict(self.services),
            "versions": dict(self.versions),
            "gpu": dict(self.gpu) if self.gpu is not None else None,
        }
