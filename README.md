# PodCodex

**Transcribe, translate, search your podcasts.**

Turn audio into a searchable knowledge base. Ingest podcasts, YouTube channels, lectures, or interviews — anything with speech — and build a structured, multilingual library you can query by meaning.

PodCodex is a **local-first desktop app** that transcribes, diarizes, corrects, translates, and indexes your audio into an embedded LanceDB vector store. A bundled **Discord bot** lets anyone search the library with slash commands.

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

### Pipeline

- WhisperX (tiny → large-v3-turbo) + pyannote diarization
- LLM correct and translate via Ollama, OpenAI-compatible APIs, or manual copy/paste
- Voice cloning with Qwen3-TTS (voice sample extraction, segment generation, episode assembly)
- Batch pipeline: select N episodes, run any steps in order, skip what's already done (provenance-based)
- Global task bar with per-episode logs, progress, cancellation

### Editing & playback

- Shared segment editor: inline edit, speaker dropdown, timestamp snap to playback position
- Flagged-segment detection (unknown speakers, low density), pagination, filters
- Global audio player persists across pages — WaveSurfer waveform, per-episode speed, drag-to-seek
- Word-level diff view between original and corrected segments
- Version history per step with full provenance (model, params, timestamp, content hash)

### Search

- Semantic (BGE-M3 hybrid or E5), exact, or random across indexed segments
- Episode / speaker / source filters
- Per-episode search panel and show-wide search tab
- Export transcripts as text, SRT, VTT, or ZIP archive

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
| Search         | SQLite + numpy, BGE-M3 / E5 embeddings, Chonkie     |
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

### Extras

| Extra      | Installs                                           | Needed for                           |
|------------|----------------------------------------------------|--------------------------------------|
| `desktop`  | fastapi, uvicorn                                   | Desktop app backend                  |
| `pipeline` | whisperx, pyannote-audio, ollama, openai, qwen-tts | Transcription, correction, synthesis |
| `rag`      | torch, sentence-transformers, chonkie, bm25s       | Embeddings & semantic search         |
| `youtube`  | yt-dlp                                             | YouTube ingest                       |

> **YouTube subtitles:** Manual subtitles download instantly. Auto-generated subtitles require deno (yt-dlp uses it to solve YouTube's JS challenges) and take ~60 seconds per episode due to rate limiting.
| `bot`      | discord.py                                         | Discord bot                          |

### Environment variables

Create `.env` at the repo root, set only what you need:

```env
HF_TOKEN=your_huggingface_token   # speaker diarization (pyannote)
API_KEY=your_api_key              # any OpenAI-compatible provider
DISCORD_TOKEN=your_bot_token      # Discord bot
PODCODEX_DB=/path/to/vectors.db   # optional override of SQLite location
```

`HF_TOKEN` requires accepting the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

---

## Discord bot

Search your transcripts from Discord with slash commands.

| Command     | Description                                             |
|-------------|---------------------------------------------------------|
| `/search`   | Semantic / hybrid search by meaning                     |
| `/exact`    | Literal substring match (like Ctrl+F)                   |
| `/random`   | Random quote, with show/episode/speaker/source filters  |
| `/stats`    | Index overview (shows, episodes, duration)              |
| `/episodes` | List episodes for a show                                |
| `/setup`    | Per-server defaults (admin)                             |

Run locally:

```bash
uv sync --extra bot --extra rag
DISCORD_TOKEN=... podcodex-bot --model bge-m3 --chunking semantic --top-k 5
```

Deploy to a VPS: the [`deploy/`](deploy/) directory ships a Docker Compose setup (bot-only image, ~3 GB, auto-restarts). See [`deploy/MULTI_SHOW.md`](deploy/MULTI_SHOW.md) for multi-show config.

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
- **RAG** (`src/podcodex/rag/`) — chunking, embedding, SQLite store, hybrid retrieval.
- **Tauri** (`src-tauri/`) — thin Rust shell, auto-spawns backend, native file dialogs.

Every pipeline save (transcribe, correct, translate, manual edit) is archived as a **version** under `.versions/{step}/` with full provenance (model, params, content hash). Episode status is tracked in a per-show `pipeline.db` SQLite. All embeddings live in a single `vectors.db` — no external vector store, search is numpy cosine similarity. Collection names follow `{show}__{model}__{chunker}`.

## Roadmap

See [ROADMAP.md](ROADMAP.md). Next up: semi-automatic speaker mapping via voice embeddings, then a zero-config "simple mode" for non-technical users, then standalone `.app`/`.deb`/`.exe` distribution.

## Notes

- WhisperX does not yet support MPS — transcription runs on CPU on Apple Silicon.
- YouTube auto-generated subtitles need deno installed. yt-dlp delegates JS challenge solving to deno at runtime. Without it, YouTube returns 429 errors for auto-generated captions. Manual subtitles work without deno.
- Ollama correct/translate may not work reliably with small models. Larger models are recommended but this needs more testing.
- Qwen3-TTS is GPU-heavy — CUDA recommended for synthesis.

## License

MIT. See [LICENSE](LICENSE).
