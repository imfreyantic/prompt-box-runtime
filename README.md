# Prompt box

GPU-side runtime for a self-hosted **Qwen3.8-27B (HauhauCS Aggressive, GGUF)** chat and prompt-writing server.
One long-running process (`python -m box.main`) on a freshly rented GPU instance:

1. starts an **HTTPS gateway first**, so the controlling app can watch progress from the first second;
2. **downloads the pinned model files** from Hugging Face (parallel range requests, resumable, SHA-256 verified);
3. **builds llama.cpp** at a pinned commit with the HauhauCS FastMTP patch (or installs a verified prebuilt);
4. **installs ComfyUI v0.44.0** with the `AA Prompt LLM` node and two example workflows;
5. runs **llama-server** (`127.0.0.1:8080`) and **ComfyUI** (`127.0.0.1:8188`), restarting them if they die.

Steps 2–4 run in parallel. Every step is idempotent: after a reboot finished work is skipped (marker files written
after verification) and interrupted downloads resume.

Target machine: Vast.ai on-demand instance with one NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120) running
`pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel`. That image is Ubuntu 24.04 with the system Python 3.12, PyTorch
and the CUDA 13.0 toolkit, but **without git, g++, make, curl and python3-venv**; the box installs what it needs
(apt-get for git/g++, pip for cmake/ninja, a pip-seeding fallback for venvs). Rent at least **80 GB of disk**
(the Q8_K_P model alone is 33 GB with projector and sidecar).

## Starting it

The app's onstart script clones this repository at a pinned commit into `/workspace/prompt-box/runtime` and runs
`bootstrap.sh` in the background. Because the image has no git, the script must install it (or fetch a tarball).
Example:

```bash
#!/bin/bash
set -e
mkdir -p /workspace/prompt-box/logs
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
R=/workspace/prompt-box/runtime
[ -d "$R/.git" ] || { git init -q "$R" && git -C "$R" remote add origin https://github.com/<owner>/<repo>.git; }
git -C "$R" fetch -q --depth 1 origin <runtime-commit-sha>
git -C "$R" checkout -q --force --detach <runtime-commit-sha>
cd "$R" && nohup bash bootstrap.sh >> /workspace/prompt-box/logs/bootstrap.log 2>&1 &
```

Vast runs onstart on every container start, so this is safe to repeat. Pass the environment variables below as
instance environment (`-e`), and expose ports 8443 and 8444 (`-p 8443:8443 -p 8444:8444`); Vast maps them to
random public ports that the app reads from the instance details.

`bootstrap.sh` creates `/workspace/prompt-box/venv-box`, installs `requirements-box.txt` (pinned aiohttp) and
`exec`s the orchestrator; the build step later adds the pinned cmake/ninja from `requirements-build.txt`. A second
bootstrap or orchestrator exits immediately while one is running (`flock` on `run/bootstrap.lock`, plus
`run/box.lock`).

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `PB_TOKEN` | required | 64 lowercase hex characters. Bearer token for port 8443 and HMAC key for ComfyUI login tickets. Missing or malformed: the orchestrator exits (message in `bootstrap.log`). |
| `PB_TLS_CERT_B64`, `PB_TLS_KEY_B64` | required | Base64 of the PEM certificate / private key (line-wrapped base64 is fine). Written to `$PB_ROOT/tls/{cert,key}.pem` with mode 0600 (directory 0700) and used for both ports. Missing or not PEM: the orchestrator exits. |
| `PB_QUANT` | `Q8_K_P` | Text model: `Q8_K_P`, `Q6_K_P`, `Q5_K_P` or `Q4_K_P`. |
| `PB_CTX` | `131072` | llama-server context size, 8192–262144. |
| `PB_MTP` | `fastmtp` | `fastmtp` (FastMTP sidecar, depth 3), `embedded` (model's own MTP head, depth 2) or `off`. |
| `PB_LLAMA_PREBUILT_URL`, `PB_LLAMA_PREBUILT_SHA256` | unset | If both are set: download that `.tar.gz`, verify its SHA-256 and skip compiling (see below). If only one is set it is ignored with a warning. |
| `PB_BUILD_JOBS` | auto | Override `-j` for the llama.cpp build (auto = nproc capped by the container CPU quota and by memory at 2.5 GB per job). |
| `PB_ROOT` | `/workspace/prompt-box` | Working directory. |
| `PB_RUNTIME_COMMIT` | unset | Reported as `versions.runtimeCommit` when the runtime checkout has no `.git`. |

An invalid `PB_QUANT`, `PB_CTX`, `PB_MTP` or prebuilt setting does not stop the gateway: `/health` reports
`state: "error"` with the reason and nothing is downloaded.

The secrets never appear in logs written by the box, and every line returned by `/logs` is passed through a
redactor (token, key material, bearer headers, session cookies, login tickets, `PB_*KEY/TOKEN/URL=` values).

## Layout on the box

```
/workspace/prompt-box/
  runtime/        this repository (cloned by onstart)
  venv-box/       orchestrator venv          venv-comfy/   ComfyUI venv (--system-site-packages, reuses the image's torch)
  models/         *.gguf, *.gguf.verified (hash marker), *.part + *.part.json while downloading
  llama.cpp/      pinned checkout + patch, build/bin/llama-server, build/.pb-build.json marker
  ComfyUI/        v0.44.0 checkout, custom_nodes/aa_prompt_llm, user/default/workflows/AA Prompt - *.json
  logs/           box.log, llama.log, comfy.log, bootstrap.log
  tls/ run/ state/ patches/ downloads/
```

## API gateway — port 8443 (HTTPS)

Every request needs `Authorization: Bearer <PB_TOKEN>` (compared with `hmac.compare_digest`); otherwise
`401 {"error":"unauthorized"}`. Unknown paths return `404 {"error":"not_found"}`.

### `GET /health`

```json
{"state": "installing|ready|error",
 "stage": "download|build|comfy|llama_start|comfy_start|ready|error",
 "message": "Downloading model files 12.3/33.3 GB (37%); Building llama.cpp (54%); Installing ComfyUI",
 "error": null,
 "startedAt": 1794346342441, "readyAt": null,
 "progress": {"download": {"doneBytes": 0, "totalBytes": 0,
                           "files": [{"name": "...", "doneBytes": 0, "totalBytes": 0, "verified": false}]},
              "build": {"state": "pending|running|done|error", "percent": null},
              "comfy": {"state": "pending|running|done|error"}},
 "services": {"llama": "stopped|starting|ready|error", "comfy": "stopped|starting|ready|error"},
 "versions": {"llamaCommit": "4df29be4…", "patchSha256": "98128540…", "comfy": "v0.44.0",
              "quant": "Q8_K_P", "ctx": 131072, "mtp": "fastmtp", "runtimeCommit": "…" },
 "gpu": {"name": "…", "vramMiB": 97887, "powerLimitW": 600.0, "powerMaxW": 600.0, "driver": "…", "cuda": "13.0"}}
```

* `state` is the lifecycle: `installing` until both services first become healthy, then `ready`; `error` is sticky
  and `error` holds every failure joined by ` | ` (`message` shows the first).
* `stage` is the earliest unfinished step in the order download → build → comfy → llama_start → comfy_start, even
  though the first three run in parallel; `progress` has the detail.
* `services` is live. After `ready`, a crashed service shows `starting` while it is restarted (up to 3 restarts per
  service, with 10/30/60 s backoff; `message` mentions the restart) and `error` once the budget is spent, which also
  sets `state: "error"`. Chat requests only need `services.llama == "ready"`.
* `gpu` is `null` if `nvidia-smi` is unavailable; a field nvidia-smi reports as `[N/A]` is `null`.
* `runtimeCommit` is `git rev-parse HEAD` of the runtime checkout (or `PB_RUNTIME_COMMIT`, else `null`).

### `GET /logs?source=box|llama|comfy|bootstrap&lines=N`

Returns `{"lines": [...]}`: the last N lines (default 200, clamped to 1..1000) of that log, ANSI colours stripped,
secrets redacted, very long lines truncated. `400 {"error":"invalid_source"}` / `{"error":"invalid_lines"}` for bad
parameters. `llama.log` holds the clone/patch/cmake/build output followed by llama-server; `comfy.log` holds the
clone/pip output followed by ComfyUI; `box.log` is the orchestrator; `bootstrap.log` is the onstart redirect.

### `POST /v1/chat/completions`, `GET /v1/models`

Reverse proxy to llama-server (`http://127.0.0.1:8080`, same path and query). The client's `Authorization` (and
cookies) are not forwarded. Request bodies over 32 MiB get `413 {"error":"payload_too_large","maxBytes":33554432}`.
Responses are streamed chunk by chunk; for `text/event-stream` (`"stream": true`) the gateway sets
`Cache-Control: no-cache` and `X-Accel-Buffering: no`. If the client disconnects, the upstream connection is closed
and llama-server stops generating. When llama-server is not ready:
`503 {"error":"not_ready","state":"installing","llama":"starting"}`. If it cannot be reached:
`502 {"error":"upstream_unavailable"}`.

The server alias is `qwen3.8-27b-aggressive`. Server defaults are the instruct profile (temperature 0.7, top-p 0.8,
top-k 20, presence penalty 1.5) with reasoning on at `xhigh`; override per request, e.g.
`"chat_template_kwargs": {"enable_thinking": false}` or `{"enable_thinking": true, "reasoning_effort": "low"}`.
Reasoning is returned in `message.reasoning_content` (`--reasoning-format deepseek`). Images go in OpenAI
`image_url` parts as data URLs.

## ComfyUI — port 8444 (HTTPS, for a human in a browser)

The app's `/prompt` page sends the user to a one-time login URL:

```js
// Node.js, in the app
const exp = Math.floor(Date.now() / 1000) + 120;                       // now <= exp <= now + 300
const sig = crypto.createHmac('sha256', Buffer.from(PB_TOKEN, 'utf8'))   // key = the 64-char hex string's bytes
                  .update(`comfy-login:${exp}`).digest('hex');
const url = `https://${publicIp}:${port8444}/__pb_login?ticket=${exp}.${sig}`;
```

`GET /__pb_login?ticket=<exp>.<sig>` accepts a ticket once (used tickets are remembered until they expire), sets
`pb_comfy=<64 hex>; Secure; HttpOnly; SameSite=Lax; Path=/; Max-Age=43200` (session kept in memory for 12 h; lost
when the orchestrator restarts) and redirects to `/`. An invalid, expired or reused ticket gets a small 401 page
telling the user to open ComfyUI from the app's /prompt page — unless the browser already holds a valid session,
in which case it is simply redirected to `/`. Every other path needs the session cookie (same 401 page otherwise)
and is reverse-proxied to `http://127.0.0.1:8188`, including the `/ws` WebSocket. `Host` and `Origin` are rewritten
to `127.0.0.1:8188`; `Referer`, `Sec-Fetch-*`, `Authorization` and the `pb_comfy` cookie are removed. (ComfyUI
answers 403 to `Sec-Fetch-Site: cross-site`, which the browser sends on the redirect that follows the login.)
Bodies are streamed both ways. The certificate is whatever the app supplied, so a self-signed one triggers a browser
warning; tickets are deterministic per `exp`, so generate a fresh one for each click.

## The ComfyUI node and workflows

`custom_nodes/aa_prompt_llm` registers **AA Prompt LLM (Diddy)** (`AAPromptLLM`, category `AA/Prompt`). Inputs:
`prompt`, `thinking` (`off`/`low`/`medium`/`xhigh`), `max_tokens` (16–32768), `temperature`, `top_p`, `top_k`,
`presence_penalty`, `seed`, optional `system` and optional `image` (every image of the batch is sent as a PNG data
URL). It POSTs a non-streaming request to `http://127.0.0.1:8080/v1/chat/completions` (900 s timeout, no proxy;
override the URL with `AA_PROMPT_LLM_URL`) and returns `text` and `reasoning`, plus `{"ui": {"text": [text]}}` for
the history. It uses only the standard library, numpy and Pillow and does not import ComfyUI code.

`workflows/` holds two UI-format workflows (copied to `ComfyUI/user/default/workflows/`, so they appear in the
Workflows sidebar; a copy the user edited in place is not overwritten) and their API-format equivalents in
`workflows/api/`:

* **AA Prompt - Text to Prompt**: AA Prompt LLM → Preview as Text (`PreviewAny`).
* **AA Prompt - Image to Prompt**: Load Image → AA Prompt LLM (`image`) → Preview as Text.

## Prebuilt llama.cpp (optional fast path)

Set `PB_LLAMA_PREBUILT_URL` and `PB_LLAMA_PREBUILT_SHA256`. The archive is downloaded to `downloads/`, its SHA-256
checked, extracted safely, and the directory that contains `llama-server` (with its shared libraries, if any) is
moved into `llama.cpp/build/bin/`; `llama-server --version` must succeed. A box that built from source can produce
one with `tar -C /workspace/prompt-box/llama.cpp/build/bin -czf llama-server-sm120.tar.gz .`. The URL may be a
pre-signed link: it is never logged with its query string. llama-server runs with `LD_LIBRARY_PATH` pointing at
`build/bin`.

## Build and run details

* Model files come from `https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/993a5971…/<file>`.
  The resolve URL is turned into the signed CDN URL once (refreshed after 30 min or on 401/403/404/410) and 64 MiB
  chunks are fetched over 16 connections for the text model and 4 for the projector and sidecar, all three files in
  parallel. Each completed chunk is recorded after an `fdatasync`; after the last one the file is hashed, renamed
  and a `<file>.verified` marker holding the hash is written. A hash mismatch puts the box in `error` and keeps the
  data (delete the `.part` files to retry). Free disk space is checked before downloading.
* llama.cpp: shallow fetch of exactly `4df29be4f4c3673f428170fda944a5b19f743bb8` (full fetch as fallback), `HEAD`
  verified, patch `HauhauCS-FastMTP-llama.cpp.patch` downloaded from the same HF revision and verified
  (2445 bytes, SHA-256 `98128540…`), applied unless `git apply --reverse --check` shows it is already applied, then
  `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`
  and `cmake --build build --config Release -j N --target llama-server` (cmake and the Ninja generator come from
  pip, g++ from apt-get).
* ComfyUI: `git clone --depth 1 --branch v0.44.0`, `HEAD` must be `2c75dc1b379a59e0880619d029607af5169a62ef`;
  `pip install -r requirements.txt` with the image's torch/torchvision/torchaudio pinned as constraints.
* llama-server command line (`PB_MTP=fastmtp`):
  `llama-server --model <quant> --mmproj <mmproj> --spec-draft-model <FastMTP> --spec-draft-ngl all --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --ctx-size <PB_CTX> --parallel 1 --batch-size 2048 --ubatch-size 512 --n-gpu-layers all --split-mode none --flash-attn on --no-mmap --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 --jinja --reasoning on --reasoning-effort xhigh --reasoning-preserve --reasoning-format deepseek --alias qwen3.8-27b-aggressive --host 127.0.0.1 --port 8080`.
  `embedded` drops `--spec-draft-model/--spec-draft-ngl` and uses `--spec-draft-n-max 2`; `off` drops all
  `--spec-*` flags.
* ComfyUI: `venv-comfy/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch`; ready when
  `/system_stats` answers and `/object_info` contains `AAPromptLLM` and `PreviewAny`.
* The box never shuts the machine down and never deletes downloaded or built files because of an error; on an
  unrecoverable failure `/health` and `/logs` keep working.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install aiohttp==3.14.5 numpy pillow
.venv/bin/python -m unittest discover -s tests -t .
```

The tests are offline: fake Hugging Face/CDN servers, fake llama-server/ComfyUI processes, local git repositories
and a throwaway self-signed certificate (the git and openssl CLIs are needed for those tests; they are skipped
otherwise; the image-encoding tests skip without numpy/Pillow).

## Credits and licences

* **Qwen3.8-27B** by the Qwen team — Apache-2.0.
* **Qwen3.8-27B-Uncensored-HauhauCS-Aggressive** GGUF release (K_P quants, BF16 projector, FastMTP sidecar and the
  FastMTP llama.cpp patch) by **HauhauCS** — released under Apache-2.0 per its model card. Downloaded at runtime;
  not redistributed here.
* **llama.cpp** by ggml-org and contributors — MIT. Cloned and built at runtime.
* **ComfyUI** by Comfy-Org — GPL-3.0. Cloned at runtime only; nothing from ComfyUI is included in this repository.
