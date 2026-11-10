#!/usr/bin/env bash
# Prompt box bootstrap.
#
# Started by the instance's onstart script from the runtime checkout:
#   cd /workspace/prompt-box/runtime && nohup bash bootstrap.sh >> /workspace/prompt-box/logs/bootstrap.log 2>&1 &
#
# Creates (idempotently) the orchestrator virtualenv and execs the orchestrator, which
# starts the HTTPS gateway first and then does all slow work (model download,
# llama.cpp build, ComfyUI install) while reporting progress on /health.
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PB_ROOT="${PB_ROOT:-/workspace/prompt-box}"
export PYTHONUNBUFFERED=1
VENV="$PB_ROOT/venv-box"
mkdir -p "$PB_ROOT/logs" "$PB_ROOT/run"

log() { printf '%s bootstrap: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# One bootstrap/orchestrator per box: fd 9 stays open (and locked) through the exec below.
if command -v flock >/dev/null 2>&1; then
  exec 9>"$PB_ROOT/run/bootstrap.lock"
  if ! flock -n 9; then
    log "another bootstrap or orchestrator is already running; exiting"
    exit 0
  fi
fi

find_python() {
  local c
  for c in "${PB_BASE_PYTHON:-}" python3 python; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1 &&
      "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      command -v "$c"
      return 0
    fi
  done
  return 1
}

PY="$(find_python)" || { log "ERROR: no Python >= 3.10 found"; exit 1; }
export PB_BASE_PYTHON="$PY"
log "runtime $RUNTIME_DIR; base python $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"

venv_ok() { [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; }

apt_install() {
  if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = 0 ]; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq &&
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$@"
  else
    return 1
  fi
}

if ! venv_ok; then
  log "creating $VENV"
  if ! "$PY" -m venv "$VENV"; then
    # Ubuntu's python3 has no ensurepip unless python3-venv is installed.
    log "python -m venv failed; creating it without pip and seeding pip from the system pip"
    "$PY" -m venv --without-pip "$VENV" || true
    "$PY" -m pip --python "$VENV/bin/python" install --disable-pip-version-check --no-input -q pip || true
  fi
  if ! venv_ok; then
    log "installing python3-venv with apt-get"
    apt_install python3-venv || true
    "$PY" -m venv "$VENV" || true
  fi
  venv_ok || { log "ERROR: could not create $VENV"; exit 1; }
fi

REQ="$RUNTIME_DIR/requirements-box.txt"
STAMP="$VENV/.pb-requirements.sha256"
WANT="$("$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$REQ")"
if [ "$(cat "$STAMP" 2>/dev/null || true)" != "$WANT" ]; then
  log "installing orchestrator requirements"
  for attempt in 1 2 3 4 5; do
    if "$VENV/bin/python" -m pip install --disable-pip-version-check --no-input -q -r "$REQ"; then
      echo "$WANT" >"$STAMP"
      break
    fi
    if [ "$attempt" = 5 ]; then
      log "ERROR: pip install -r requirements-box.txt failed"
      exit 1
    fi
    log "pip install failed (attempt $attempt); retrying in $((attempt * 10))s"
    sleep $((attempt * 10))
  done
fi

cd "$RUNTIME_DIR"
log "starting orchestrator"
exec "$VENV/bin/python" -m box.main
