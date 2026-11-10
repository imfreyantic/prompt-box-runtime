"""Log redaction, tailing and logger setup."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

REDACTED = "[REDACTED]"
MAX_LINE_CHARS = 8192
_MAX_TAIL_BYTES = 32 * 1024 * 1024

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_GENERIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Authorization headers / bearer tokens
    (re.compile(r"(?i)(authorization\s*[:=]\s*)[^\r\n,;\"']+"), r"\1" + REDACTED),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1" + REDACTED),
    # Our session cookie and login tickets
    (re.compile(r"(pb_comfy=)[^;\s\"'&]+"), r"\1" + REDACTED),
    (re.compile(r"(ticket=)[^&\s\"']+"), r"\1" + REDACTED),
    # PB_* secrets echoed as KEY=value
    (re.compile(r"\b(PB_[A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD|CERT|URL)[A-Z0-9_]*\s*[=:]\s*)\S+"), r"\1" + REDACTED),
    # PEM private key material
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|$)", re.S),
     "[REDACTED PRIVATE KEY]"),
]


class Redactor:
    """Removes known secret strings and common credential patterns from text."""

    def __init__(self, secrets: Iterable[str] = ()):
        items = sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True)
        self._secret_re: Optional[re.Pattern[str]] = (
            re.compile("|".join(re.escape(s) for s in items)) if items else None
        )

    def __call__(self, text: str) -> str:
        if self._secret_re is not None:
            text = self._secret_re.sub(REDACTED, text)
        for pattern, repl in _GENERIC_PATTERNS:
            text = pattern.sub(repl, text)
        return text


class RedactingFilter(logging.Filter):
    def __init__(self, redactor: Redactor):
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed log call
            return True
        record.msg = self._redactor(message)
        record.args = None
        return True


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def tail_lines(path: Path, n: int, block_size: int = 64 * 1024) -> list[str]:
    """Return the last ``n`` lines of a file without reading all of it."""
    if n <= 0:
        return []
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        return []
    with fh:
        fh.seek(0, 2)
        pos = fh.tell()
        data = b""
        while pos > 0 and data.count(b"\n") <= n and len(data) < _MAX_TAIL_BYTES:
            step = min(block_size, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
    lines = data.splitlines()
    if pos > 0 and lines:
        lines = lines[1:]  # first line is probably partial
    out = []
    for raw in lines[-n:]:
        line = strip_ansi(raw.decode("utf-8", "replace")).rstrip("\r")
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + " …[truncated]"
        out.append(line)
    return out


def setup_logging(log_path: Path, redactor: Redactor) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    redact = RedactingFilter(redactor)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(redact)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    console.addFilter(redact)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    return logging.getLogger("box")
