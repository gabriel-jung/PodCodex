# PodCodex

**Transcribe, translate, search your podcasts.**

Turn audio into a searchable knowledge base. Ingest podcasts, YouTube channels, lectures, or interviews — anything with speech — and build a structured, multilingual library you can query by meaning.

PodCodex is a **local-first desktop app** that transcribes, diarizes, corrects, translates, and indexes your audio into an embedded LanceDB vector store. A bundled **Discord bot** lets anyone search the library with slash commands, and an **MCP server** exposes the same retrieval to Claude Desktop / Claude Code.

---

## What it does

Point it at a podcast RSS feed, a YouTube channel, or a folder of recordings and it will:

1. **Ingest** — pull episodes from RSS, YouTube, or local files. Manage multiple shows with per-show config.
2. **Transcribe** — WhisperX + pyannote speaker diarization, with word-level timestamps.
3. **Correct** — fix transcription errors with an LLM (Ollama, OpenAI, Anthropic, Mistral, or manual copy/paste).
4. **Translate** — any target language, via the same LLM backends.
5. **Synthesize** (optional) — Qwen3-TTS voice cloning for dubbed versions.
6. **Index & search** — vectorize into a local LanceDB index with hybrid retrieval (vector ANN + Tantivy FTS), then search by meaning (semantic), keywords (exact), or random sampling.

All steps share a segment editor (inline editing, speaker mapping, timestamp snapping) and a global audio player (WaveSurfer waveform, per-episode speed, segment-level playback). Everything runs on your machine — no cloud lock-in, no external vector DB.

## Features

### Shows & ingest

- Add podcasts by Apple Podcasts search, RSS URL, YouTube channel/playlist, or existing folder
- Per-show pipeline defaults (Whisper model, LLM provider, translation target, …)
- Batch download with shift-select, rate-limit backoff for YouTube
- Auto-refresh artwork, feed metadata, and speaker registry
- Removed-from-feed flag: keeps local copies of episodes that disappear from the live RSS/YouTube source

### Pipeline

- WhisperX (tiny → large-v3-turbo) + pyannote diarization
- LLM correct and translate via Ollama, OpenAI-compatible APIs, or manual copy/paste
- Voice cloning with Qwen3-TTS (voice sample extraction, segment generation, episode assembly)
- Batch pipeline: select N episodes, run any steps in order, skip what's already done (provenance-based)
- Global task bar with per-episode logs, progress, cancellation

### Editing & playback

- Shared segment editor (virtualized, scales to 10k+ segments): inline edit, speaker dropdown, timestamp snap, split-at-time, tail-merge
- Flagged-segment detection (unknown speakers, low density), pagination, filters
- Global audio player persists across pages — WaveSurfer waveform, per-episode speed, drag-to-seek
- Inline "now playing" segment overlay on the audio bar; transcript editor syncs once on open, with a manual re-sync button (no auto-follow during edits)
- Word-level diff view between original and corrected segments
- Version history per step with full provenance (model, params, timestamp, content hash); atomic writes for config, manifest, and versions

### Search

- Semantic (BGE-M3 hybrid or E5), exact, or random across indexed segments
- Show / episode / speaker / source / pub-date filters, unified across frontend, Discord bot, and MCP
- Per-episode search panel and show-wide search tab
- Command palette (Cmd+K) searches transcripts across every indexed show
- Export transcripts as text, SRT, VTT, or ZIP archive

### Integrations

- **Discord bot** — `/search`, `/exact`, `/random`, `/speakers`, `/stats`, `/episodes`; simple/advanced command split; password-gated per-server show access
- **MCP server** — expose `search`, `exact`, `list_shows`, `get_context` to Claude Desktop (one-toggle setup from Settings) or Claude Code via stdio/HTTP

## Tech stack

| Layer          | Technology                                          |
|----------------|-----------------------------------------------------|
| Desktop shell  | Tauri v2 (Rust)                                     |
| Frontend       | React 19, Vite, TypeScript, Tailwind CSS, shadcn/ui |
| State          | Zustand, TanStack Query, TanStack Router            |
| Backend        | FastAPI (REST + WebSocket, background tasks)        |
| Transcription  | WhisperX, pyannote-audio                            |
| LLM            | Ollama (local), OpenAI, Anthropic, Mistral          |
| Voice cloning  | Qwen3-TTS                                           |
| Search         | LanceDB, BGE-M3 / E5 embeddings, Chonkie            |
| Audio          | WaveSurfer.js, ffmpeg, sox                          |

---

## Install

Prerequisites: Python 3.12, Node.js (LTS), ffmpeg, and [uv](https://docs.astral.sh/uv/). For YouTube auto-generated subtitles, [deno](https://deno.com/) is also required (`brew install deno` on macOS).

```bash
git clone https://github.com/gabriel-jung/podcodex
cd podcodex

# Python deps (pick the extras you need)
uv sync --extra desktop --extra pipeline --extra rag --extra youtube

# Frontend deps
cd frontend && npm install && cd ..

# Run in browser (API on :18811, Vite on :5173)
make dev-no-tauri
```

For a native window (requires Rust + GTK/WebKit on Linux), run `make dev` instead. See [Makefile](Makefile) for all targets.

### Installing a pre-built release

Grab the latest `.dmg` (macOS) or `.msi` (Windows) from the [Releases](https://github.com/gabriel-jung/podcodex/releases) page.

**macOS — first-launch quarantine.** The DMG is not yet code-signed or notarized, so Gatekeeper marks the copied app as quarantined and refuses to open it with:

> "PodCodex.app" is damaged and can't be opened. You should move it to the Bin.

The app is not actually damaged. After dragging it into `/Applications`, strip the quarantine attribute once:

```bash
xattr -dr com.apple.quarantine /Applications/PodCodex.app
```

Then open it normally. This is a one-off — subsequent launches don't need it. Signing + notarization is on the roadmap.

### Build a standalone .dmg / .msi

The desktop build freezes the Python backend with PyInstaller into a single
sidecar binary, fetches static `ffmpeg` + `yt-dlp`, then asks Tauri to bundle
everything into a native installer (macOS DMG, Windows MSI). Linux runs from
source via `make dev`. macOS arm64 verified; Windows not yet smoke-tested.

```bash
make setup-pyinstaller    # one-time — adds PyInstaller to .venv
make bundle               # ~5 min — freezes backend, fetches sidecars,
                          # builds frontend, runs cargo tauri build
```

`make bundle` chains `bundle-server` → `bundle-natives` → `npm run build` →
`cargo tauri build`. Use the individual `make bundle-*` targets when
iterating on a single layer. Clean rebuild: `make clean && make bundle`.

macOS outputs:
`src-tauri/target/release/bundle/macos/PodCodex.app` (~497 MB) and
`src-tauri/target/release/bundle/dmg/PodCodex_<version>_<arch>.dmg`
(~459 MB). ML weights download on first use to
`~/Library/Application Support/podcodex/models/`.

Shipped installer is CPU-only; an optional CUDA backend is downloaded
in-app on NVIDIA hosts — see [Phase M](ROADMAP.md#phase-m--standalone-distribution-v010).

Full per-OS guide (macOS, Windows native + signing, notarization,
troubleshooting): see [`deploy/BUILD.md`](deploy/BUILD.md).

| Target               | What it does                                                  |
|----------------------|---------------------------------------------------------------|
| `make setup`         | Install Python + frontend deps                                |
| `make dev`           | FastAPI + Vite + Tauri (hot reload)                           |
| `make dev-no-tauri`  | FastAPI + Vite only (browser at localhost:5173)               |
| `make bundle-server` | PyInstaller-freeze the backend                                |
| `make bundle-natives`| Download ffmpeg + yt-dlp                                      |
| `make bundle`        | Full standalone .app/.dmg                                     |
| `make bundle-sign`   | Sign + notarize (needs `APPLE_SIGNING_IDENTITY` + notary kc)  |
| `make clean`         | Remove build artifacts                                        |
| `make types`         | Regenerate frontend TS types from Pydantic                    |
| `make test`          | Run Python tests                                              |

### Extras

| Extra      | Installs                                           | Needed for                           |
|------------|----------------------------------------------------|--------------------------------------|
| `desktop`  | fastapi, uvicorn                                   | Desktop app backend                  |
| `pipeline` | whisperx, pyannote-audio, ollama, openai, qwen-tts | Transcription, correction, synthesis |
| `rag`      | torch, sentence-transformers, chonkie, lancedb     | Embeddings & semantic search         |
| `youtube`  | yt-dlp                                             | YouTube ingest                       |
| `bot`      | discord.py, openai                                 | Discord bot                          |

> **YouTube subtitles:** Manual subtitles download instantly. Auto-generated subtitles require deno (yt-dlp uses it to solve YouTube's JS challenges) and take ~60 seconds per episode due to rate limiting.

### Environment variables

Create `.env` at the repo root, set only what you need:

```env
HF_TOKEN=your_huggingface_token   # speaker diarization (pyannote)
API_KEY=your_api_key              # any OpenAI-compatible provider
DISCORD_TOKEN=your_bot_token      # Discord bot
PODCODEX_INDEX=/path/to/index     # optional override of LanceDB location
```

`HF_TOKEN` requires accepting the terms for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

---

## Discord bot

Search your transcripts from Discord with slash commands.

| Command     | Description                                             |
|-------------|---------------------------------------------------------|
| `/search`   | Semantic / hybrid search by meaning                     |
| `/exact`    | Literal substring match (like Ctrl+F), accent-aware, 1-edit fuzzy tier |
| `/random`   | Random quote, with show/episode/speaker/source filters  |
| `/speakers` | List speakers for a show with episode counts            |
| `/stats`    | Index overview (shows, episodes, duration)              |
| `/episodes` | List episodes for a show                                |
| `/unlock` · `/lock` · `/changepassword` | Per-server show access control (admin) |
| `/setup`    | Per-server defaults (admin)                             |

`/search` and `/exact` each have an `-advanced` variant exposing alpha / model / chunker / top_k — keeps the default surface clean.

Run locally:

```bash
uv sync --extra bot --extra rag
DISCORD_TOKEN=... uv run podcodex-bot
```

Full install guide (uv and Docker paths, token setup, access control, VPS deploy): see [`deploy/BOT.md`](deploy/BOT.md).

---

## Claude Desktop / MCP

An MCP server exposes your index to Claude Desktop (or any MCP-capable client) so Claude can search your transcripts directly during a conversation. The server does retrieval only — the client LLM reads the chunks and synthesizes the answer.

Fastest path: **Settings → Claude Desktop → Enable Claude Desktop integration** in the desktop app. PodCodex writes the `claude_desktop_config.json` entry, resolves the absolute path to the bundled `podcodex-mcp` stdio binary, and handles WSL (Linux binary wrapped via `wsl.exe` for Windows Claude Desktop) automatically.

Tools exposed: `search`, `exact`, `list_shows`, `get_context` — plus editable slash prompts (`/brief`, `/speaker`, `/quote`, `/compare`, `/timeline`).

Full guide (manual stdio config, Claude Code registration, prompts, troubleshooting): see [`deploy/MCP.md`](deploy/MCP.md).

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│  Tauri shell — native window, file system access    │
│  ┌───────────────────────────────────────────────┐  │
│  │  React frontend (Vite + TypeScript)           │  │
│  │  └── Zustand stores, TanStack Query           │  │
│  └───────────────────────────────────────────────┘  │
│             ↕ HTTP + WebSocket                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  FastAPI backend (Python)                     │  │
│  │  └── podcodex.core.* pipeline modules         │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

- **Frontend** (`frontend/`) — React 19, Vite, Tailwind, shadcn/ui. Talks to the backend over REST + WebSocket.
  - Pipeline step registry (`PipelineSteps.tsx`) drives sidebar, status badges, and info tab.
  - Shared hooks: `usePipelineTask` (task lifecycle), `useLLMConfig` / `buildLLMRequest` (LLM settings), `usePipelineDefaults` (step-status comparison).
  - API client factory: `createVersionApi` (segments + versions CRUD) and `createLLMPipelineApi` (adds start/manual-prompts/apply-manual).
- **Backend** (`src/podcodex/api/`) — FastAPI, exposes the pipeline as HTTP endpoints with background tasks.
  - Shared `LLMRequest` base model, `_batch_llm_step()` unified handler, `_resolve_source_segments()` for version DB lookups.
- **Core** (`src/podcodex/core/`) — transcribe, correct, translate, synthesize, versioning, per-show pipeline DB.
  - `run_llm_pipeline()` — single LLM dispatch function shared by correct and translate.
- **RAG** (`src/podcodex/rag/`) — chunking, embedding, LanceDB index store, hybrid retrieval (vector ANN + BM25 FTS).
- **Bot** (`src/podcodex/bot/`) — Discord slash commands over the shared retriever, password-gated show access.
- **MCP** (`src/podcodex/mcp/`) — stdio + HTTP MCP server exposing `search` / `exact` / `list_shows` / `get_context` plus user-editable prompts.
- **Tauri** (`src-tauri/`) — thin Rust shell, auto-spawns backend, native file dialogs.

Every pipeline save (transcribe, correct, translate, manual edit) is archived as a **version** under `.versions/{step}/` with full provenance (model, params, content hash). Episode status is tracked in a per-show `pipeline.db` SQLite. All embeddings live in a single LanceDB index under the platform's app-data directory (`<data_dir>/index`; e.g. `~/Library/Application Support/podcodex/index` on macOS, `%APPDATA%\podcodex\index` on Windows, `~/.local/share/podcodex/index` on Linux). Collection names follow `{show}__{model}__{chunker}`.

## Roadmap

See [ROADMAP.md](ROADMAP.md). Next up: semi-automatic speaker mapping via voice embeddings, then standalone `.dmg` / `.msi` distribution.

## Notes

- WhisperX does not yet support MPS — transcription runs on CPU on Apple Silicon.
- YouTube auto-generated subtitles need deno installed. yt-dlp delegates JS challenge solving to deno at runtime. Without it, YouTube returns 429 errors for auto-generated captions. Manual subtitles work without deno.
- Ollama correct/translate may not work reliably with small models. Larger models are recommended but this needs more testing.
- Qwen3-TTS is GPU-heavy — CUDA recommended for synthesis.

## License

MIT. See [LICENSE](LICENSE).
