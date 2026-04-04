# podcodex

Turn audio into a searchable knowledge base. Ingest podcasts, YouTube channels, lectures, interviews - anything with speech - and build a structured, multilingual library you can search by meaning.

The **desktop app** is the main interface: add shows from RSS, YouTube, or local folders, then transcribe, correct, translate, and index - all from one window. The **Discord bot** lets anyone search your library. The **pipeline** can also be driven from the CLI or Python.

## What is PodCodex?

PodCodex is a local-first app that turns audio sources into a searchable knowledge database. Point it at a podcast RSS feed, a YouTube channel, or a folder of recordings - it transcribes, identifies speakers, corrects errors with an LLM, translates to other languages, and indexes everything for semantic search.

The end result is a **structured, queryable archive** of everything that was said: who said it, when, in what episode, in any language. You can search across hundreds of hours by meaning, not just keywords.

The pipeline has three layers:

**Ingest** - pull episodes from RSS feeds, YouTube channels/playlists, or local audio files. Manage multiple shows with per-show configuration.

**Process** - transcribe with WhisperX + pyannote speaker diarization, correct with an LLM (Ollama, OpenAI, Anthropic, Mistral), translate to any language, and optionally clone voices with Qwen3-TTS for dubbed versions.

**Search** - vectorize transcripts into a local SQLite store, then search by meaning (semantic), keywords (exact), or random sampling. Deploy a Discord bot so others can search too.

All steps share a segment editor (inline editing, speaker mapping, timestamp snapping) and a global audio player (WaveSurfer waveform, per-episode speed, segment-level playback).

## Features

**Show management:**
- Search and add podcasts by name (Apple Podcasts directory) or RSS URL
- Add YouTube channels or playlists - download audio, import subtitles as transcripts
- Import existing show folders
- Browse episodes with status indicators (downloaded, transcribed, polished, translated, synthesized, indexed)
- Per-step status chips (none/outdated/done) comparing provenance against pipeline defaults
- Filter episodes by duration (min/max) and title (include/exclude)
- Sortable episode list (date, title, duration, number) with list and card views
- Download episodes from RSS feeds or YouTube (single, batch, or shift-select)
- Move show folder with optional file relocation
- Show-level speaker registry (define known speakers for better diarization and LLM context)
- Per-show pipeline configuration (Whisper model, LLM provider, translation target, etc.)
- Delete audio files
- Export all episode files as ZIP

**Transcription (WhisperX):**
- WhisperX for speech-to-text with word-level timestamps (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo)
- pyannote for speaker diarization
- Background processing with real-time progress bar
- Upload existing transcript files (JSON, SRT, VTT)
- Import transcripts from local files or YouTube subtitles
- Speaker name mapping after diarization

**LLM correction:**
- Fix transcription errors using a local LLM (Ollama) or cloud API (OpenAI, Anthropic, Mistral)
- Manual mode: generate prompts to paste into any chat (ChatGPT, Claude, etc.), then apply the corrections
- Word-level diff view comparing original transcript to corrected version

**Translation (LLM):**
- Translate to any language using the same LLM modes
- Multiple target languages per episode
- Side-by-side view of source and translated text

**Voice cloning (Qwen3-TTS):**
- Extract voice samples from source audio per speaker
- TTS generation segment-by-segment with cloned voices
- Episode assembly from generated segments

**Batch pipeline:**
- Select multiple episodes and run pipeline steps (transcribe → diarize → assign → export → polish → translate → index) in one go
- Per-step configuration dialogs - customize model, language, and provider before running
- Global task bar with real-time progress, per-episode status (pending/running/done/failed), and expandable logs
- Idempotent - skips steps already completed for each episode (provenance-based comparison)
- Cancel at any time, with result summary (completed/skipped/failed)
- GPU-safe sequential execution with per-episode locking
- Duration-based LLM batching for efficient polish/translate
- YouTube rate-limit detection with automatic backoff

**Indexing & search:**
- Choose embedding model and chunking strategy
- Vectorize episodes into local SQLite store
- Semantic and exact-match search across indexed segments
- Random quote with optional filters

**Segment editor (shared across all steps):**
- Inline text editing with auto-resize
- Speaker dropdown per segment
- Timestamp editing with "snap to current playback position"
- Insert / delete segments
- Flagged segment detection (unknown speakers, low speech density)
- Pagination, search, speaker/flagged/changed filters
- "Estimate timestamps" for transcripts without timing data
- Active segment highlighting during audio playback

**Audio player:**
- Global player persists across page navigation
- Play from any segment with one click
- WaveSurfer.js waveform with drag-to-seek, skip +/-15s, volume control
- Per-episode playback speed (saved and restored automatically)
- Toggleable time display (remaining / elapsed / total)
- Current segment text overlay

**Settings & export:**
- Model cache management (list models, disk usage, delete)
- VRAM monitoring (GPU device, used/free/reserved)
- Export transcripts as plain text, SRT subtitles, WebVTT, or ZIP archive

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop App | Tauri (Rust) |
| Frontend | React 19, TypeScript, Tailwind CSS, shadcn/ui |
| State | Zustand, TanStack Query |
| Backend | FastAPI (Python) |
| Transcription | WhisperX, pyannote (speaker diarization) |
| LLM | Ollama (local), OpenAI, Anthropic, Mistral |
| Voice Cloning | Qwen3-TTS |
| Search | SQLite, numpy, BGE-M3 / E5 embeddings |
| Audio | WaveSurfer.js, ffmpeg, sox |

---

## Discord Bot

Search your podcast transcripts from Discord with slash commands.

**Commands:**

- `/search <question>` - find relevant passages by meaning (semantic) or keywords
- `/exact <query>` - literal text search, like Ctrl+F
- `/random` - random quote with optional show/episode/speaker filters
- `/stats` - overview of indexed shows, episodes, and duration
- `/episodes [show]` - list episodes (auto-selects if only one show)
- `/setup` - configure server defaults (admin)
- `/sync` - manually sync the command tree (admin)

All search commands support optional `show`, `episode`, `speaker`, `source`, and `compact` filters.

**Admin commands:**

- `/unlock <show> <password>` - unlock a show for this server
- `/lock <show>` - remove a show from this server
- `/help` - list available commands and usage

### Setting up the Discord application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**
2. Under **Bot**, click **Reset Token** and copy it - this is your `DISCORD_TOKEN`
3. Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent** (needed for the bot to read messages if you add message-based features later)
4. Under **OAuth2 → URL Generator**:
   - **Scopes**: select `bot` and `applications.commands`
   - **Bot Permissions**: select `Send Messages`, `Embed Links`, `Use Slash Commands`
5. Copy the generated URL and open it in your browser to invite the bot to your server

### Running the bot locally

```bash
# Required env vars
DISCORD_TOKEN=your_bot_token
# Start
podcodex-bot --model bge-m3 --chunking semantic --top-k 5
podcodex-bot --dev-guild 123456789  # instant command sync for development
```

### Deploying to a server

The `deploy/` directory contains everything needed to run the bot on a VPS (tested on Ubuntu 4GB RAM). Only Docker is required on the server - no Python, no GPU.

#### 1. Clone and configure

```bash
ssh user@vps
git clone https://github.com/gabriel-jung/podcodex ~/podcodex
cd ~/podcodex
cp deploy/.env.example deploy/.env.production
# Edit deploy/.env.production and set DISCORD_TOKEN
```

#### 2. Copy your vectors database

```bash
# From your local machine
scp /path/to/vectors.db user@vps:~/podcodex/deploy/data/
```

#### 3. Build and start

```bash
cd ~/podcodex

# Build the bot image (includes BGE-M3 model download - takes a few minutes)
docker compose -f deploy/docker-compose.yml build

# Start the bot
docker compose -f deploy/docker-compose.yml up -d
```

The bot image installs only core + bot dependencies (no pipeline), keeping it under 3GB. Search uses numpy over SQLite - no external services needed. Logs rotate automatically (50MB x 3 files). The bot auto-restarts on crash or reboot.

#### Updating

**New code:**

```bash
cd ~/podcodex
git pull
docker compose -f deploy/docker-compose.yml build bot
docker compose -f deploy/docker-compose.yml up -d
```

**New episodes indexed locally:**

```bash
# From your local machine
scp /path/to/vectors.db user@vps:~/podcodex/deploy/data/

# On the VPS
docker compose -f deploy/docker-compose.yml run --rm --entrypoint podcodex bot \
    sync --db /app/data/vectors.db --show "My Podcast"
```

No manual stop needed - `up -d` replaces the running container automatically.

**Checking logs:**

```bash
docker compose -f deploy/docker-compose.yml logs bot --tail 50
docker compose -f deploy/docker-compose.yml logs -f bot        # follow live
docker compose -f deploy/docker-compose.yml ps                 # container status
```

---

## Pipeline

The processing pipeline can be used from Python or the CLI, independently of the desktop app.

```text
Audio → transcribe → correct → translate → synthesize
                       ↓
                  vectorize → search
```

### System requirements

```bash
# Ubuntu/Debian/WSL
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Environment variables

Create a `.env` file at the root. Only set what you need:

```env
# Bot
DISCORD_TOKEN=your_bot_token      # required for the Discord bot

# Search
PODCODEX_DB=/path/to/vectors.db   # optional, overrides default SQLite location

# Pipeline
HF_TOKEN=your_huggingface_token   # required for speaker diarization
API_KEY=your_api_key              # any OpenAI-compatible provider (Mistral, OpenAI, etc.)
```

`HF_TOKEN` requires accepting the pyannote model terms on Hugging Face:

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Transcription

```python
from podcodex.core import transcribe

transcribe.transcribe_file("episode.mp3", output_dir="ep01/")
transcribe.diarize_file("episode.mp3", output_dir="ep01/")
transcribe.assign_speakers("episode.mp3", output_dir="ep01/")

transcribe.save_speaker_map("episode.mp3", {
    "SPEAKER_00": "Alice",
    "SPEAKER_01": "Bob",
}, output_dir="ep01/")

transcript = transcribe.export_transcript("episode.mp3", output_dir="ep01/",
                                          show="My Podcast", episode="ep01")

# Check pipeline status
transcribe.processing_status("episode.mp3", output_dir="ep01/")
```

### LLM correction

```python
from podcodex.core import polish

segments = transcribe.load_transcript("episode.mp3", output_dir="ep01/")

# Via API
polished = polish.polish_segments(
    segments, mode="api", source_lang="French",
    context="French podcast about film music, hosted by Alice and Bob",
    model="mistral-small-latest",
    api_base_url="https://api.mistral.ai/v1",
)

# Via Ollama
polished = polish.polish_segments(
    segments, mode="ollama", model="qwen3:14b", source_lang="French",
)

polish.save_polished_raw("episode.mp3", polished, output_dir="ep01/")
# After review:
polish.save_polished("episode.mp3", polished, output_dir="ep01/")
```

### Translation

```python
from podcodex.core import translate

segments = polish.load_polished("episode.mp3", output_dir="ep01/")

# Via API
translated = translate.translate_segments(
    segments, mode="api",
    source_lang="French", target_lang="English",
    context="French podcast about film music",
    model="mistral-small-latest",
    api_base_url="https://api.mistral.ai/v1",
)

# Via Ollama
translated = translate.translate_segments(
    segments, mode="ollama", model="qwen3:14b",
    source_lang="French", target_lang="English",
)

# Manual copy/paste workflow
batches = translate.build_manual_prompts_batched(
    segments, source_lang="French", target_lang="English",
)
# ... paste each batch into an LLM, collect results ...
translated = translate.translate_segments(json_from_llm, mode="manual")

translate.save_translation_raw("episode.mp3", translated, "english", output_dir="ep01/")
```

#### Translation modes

| Mode | Description | Best for |
|------|-------------|----------|
| `ollama` | Local LLM via Ollama | Privacy, offline - use 14B+ models |
| `api` | OpenAI-compatible API | Best quality (Mistral, OpenAI, etc.) |
| `manual` | Copy/paste via LLM UI | Full control, best quality |

### Voice cloning

```python
from podcodex.core import synthesize

voice_samples = synthesize.extract_voice_samples(
    "episode.mp3", translated, output_dir="ep01/", min_duration=5.0, top_k=3,
)

model = synthesize.load_tts_model(model_size="1.7B")
clone_prompts = synthesize.build_clone_prompts(model, voice_samples, sample_index=0)

# Generate one segment at a time
for i, seg in enumerate(translated):
    output_path = Path(f"ep01/tts_segments/{i:04d}_{seg['speaker']}.wav")
    result = synthesize.generate_segment(
        model, seg, clone_prompts, output_path, language="English",
    )

# Assemble into final episode
episode_path = synthesize.assemble_episode(
    generated, "episode.mp3", output_dir="ep01/", strategy="original_timing",
)
```

### Indexing & search

Indexing saves everything locally in a SQLite file (`vectors.db`). Search uses numpy brute-force cosine similarity over the SQLite store - no external services needed.

```bash
# Embed and store a transcript
podcodex vectorize ep01/ep01.transcript.json --show "My Podcast"
podcodex vectorize ep01/ep01.transcript.json --show "My Podcast" --model bge-m3 --chunking semantic

# Query
podcodex query "film music composer" --show "My Podcast"
podcodex query "film music composer" --show "My Podcast" --top-k 3 --alpha 0.7

# Manage collections
podcodex list
podcodex list --show "My Podcast"
podcodex delete my_podcast__bge-m3__semantic
```

#### Embedding models

| Model key | Embedder | Dim | Search |
|-----------|----------|-----|--------|
| `bge-m3` | BAAI/bge-m3 | 1024 | hybrid (dense + sparse via RRF) |
| `e5-small` | multilingual-e5-small | 384 | dense cosine |
| `e5-large` | multilingual-e5-large | 1024 | dense cosine |
| `pplx` | pplx-embed-context | 1024 | dense cosine |

#### Chunking strategies

| Strategy | Description |
|----------|-------------|
| `semantic` | Semantic similarity splitting (Chonkie) - recommended default |
| `speaker` | One chunk per speaker turn - fast, no extra deps |

Collection names follow the format `{show}__{model}__{chunker}`, e.g. `my_podcast__bge-m3__semantic`.

#### In Python

```python
from podcodex.rag.localstore import LocalStore
from podcodex.rag.store import collection_name
from podcodex.rag.retriever import Retriever

local = LocalStore(db_path="/path/to/vectors.db")

col = collection_name("My Podcast", model="bge-m3", chunker="semantic")
retriever = Retriever(model="bge-m3", local=local)
results = retriever.retrieve("film music", col, top_k=5, alpha=0.5)

# Filter by episode, speaker, or source
results = retriever.retrieve(
    "film music", col, top_k=5, episode="ep01", speaker="Alice",
)
```

### CLI - RSS and import

```bash
# Fetch RSS feed and list episodes (uses show.toml for the URL, or --rss to override)
podcodex rss <show_folder>
podcodex rss <show_folder> --rss https://example.com/feed.xml
podcodex rss <show_folder> --download          # also download audio files

# Import an external transcript
podcodex import transcript.json <show_folder>
podcodex import transcript.json <show_folder> --episode ep01 --show "My Podcast"
```

### Streamlit app (legacy)

> **Note:** The Streamlit app is being replaced by the desktop app. It remains functional for quick prototyping.

```bash
uv sync --extra app
streamlit run streamlit/app.py
```

---

## Development

### Quick start

```bash
git clone https://github.com/gabriel-jung/podcodex
cd podcodex

# Prerequisites - Node.js (pick one)
# Linux/WSL (via nvm, recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
nvm install --lts
# macOS
brew install node

# Install Python dependencies
uv sync --extra desktop

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start (API + frontend in browser)
make dev-no-tauri
# Open http://localhost:5173
```

Optionally, for a native window (requires Rust + system GTK/WebKit on Linux):

```bash
# One-time Rust setup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli --version "^2"

# Linux only - GTK/WebKit system deps
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev \
  libayatana-appindicator3-dev librsvg2-dev libssl-dev pkg-config

# Start everything (API + Vite + Tauri window)
make dev
```

### Installation extras

```bash
# Desktop app (lightweight - browse, upload, edit transcripts)
uv sync --extra desktop

# Add automatic transcription, TTS, LLM correction/translation
uv sync --extra desktop --extra pipeline

# Add semantic search & indexing
uv sync --extra desktop --extra rag

# Bot only (search + Discord)
uv sync --extra bot --extra rag

# Full install (everything)
uv sync --extra bot --extra pipeline --extra rag --extra desktop --extra app
```

| Extra | What it installs | Use case |
|-------|-----------------|----------|
| *(none)* | feedparser, loguru, python-dotenv | Core (RSS parsing, logging) |
| `desktop` | fastapi, uvicorn, python-multipart | Desktop app backend |
| `pipeline` | whisperx, pyannote-audio, ollama, openai, qwen-tts, etc. | Transcription, correction, translation, synthesis |
| `rag` | torch, sentence-transformers, chonkie, bm25s | Semantic search & indexing |
| `bot` | discord.py | Discord bot |
| `app` | streamlit | Legacy Streamlit UI |

### Make targets

| Target | Description |
|--------|-------------|
| `make setup` | One-time: install all Python and frontend deps |
| `make dev` | Start API + Vite + Tauri (native window, hot-reload) |
| `make dev-no-tauri` | Start API + Vite only (use browser at localhost:5173) |
| `make build` | Production `.app` / `.exe` bundle |
| `make test` | Run Python test suite |
| `make clean` | Remove build artifacts |

### Architecture

```text
┌──────────────────────────────────────────────────┐
│  Tauri shell (native window, file system access)  │
│  ┌──────────────────────────────────────────────┐ │
│  │  React frontend (Vite + TypeScript)          │ │
│  │  └── Zustand stores, TanStack Query           │ │
│  └──────────────────────────────────────────────┘ │
│         ↕ HTTP + WebSocket                        │
│  ┌──────────────────────────────────────────────┐ │
│  │  FastAPI backend (Python)                    │ │
│  │  └── podcodex.core.* pipeline modules        │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

- **Frontend** (`frontend/`): React 19, Vite, Tailwind CSS, shadcn/ui. REST API + WebSocket for real-time progress.
- **Backend** (`src/podcodex/api/`): FastAPI server exposing the pipeline as HTTP endpoints. Background tasks with progress over WebSocket.
- **Tauri** (`src-tauri/`): Thin Rust shell - auto-spawns backend, health-checks, native file dialogs. Standalone distribution (`.app`/`.deb`/`.exe`) is in development.

### API endpoints

The FastAPI backend exposes these route groups (all under `/api`):

| Prefix | Description |
|--------|-------------|
| `/health` | Status and capability check |
| `/config` | App configuration (show folders, save path, pipeline constants) |
| `/shows` | List shows, register folders, show metadata, unified episode lists, move/delete folder |
| `/rss` | RSS feed parsing, episode download |
| `/youtube` | YouTube video list, audio download, subtitle import |
| `/batch` | Multi-episode pipeline execution with progress tracking |
| `/transcribe` | Load/save segments, start transcription, speaker map, upload |
| `/polish` | Load/save segments, start correction, manual prompts |
| `/translate` | Load/save segments, start translate, manual prompts, language list |
| `/audio` | Serve audio files, extract clips, delete files |
| `/fs` | Directory browser for folder picker |
| `/synthesize` | Voice synthesis: samples, TTS generation, assembly |
| `/index` | Vectorize episodes, manage embeddings |
| `/search` | Semantic and keyword search across episodes |
| `/tasks` | List active tasks, cancel running tasks |
| `/ws` | WebSocket for real-time task progress |
| `/export` | Export segments as text, SRT, VTT, or ZIP archive |
| `/models` | Model cache management, VRAM monitoring |

### Modules

| Module | Description |
|--------|-------------|
| `podcodex.api` | FastAPI backend (REST + WebSocket, background tasks) |
| `podcodex.bot` | Discord bot (`/search`, `/exact`, `/random`, `/stats`, `/episodes`) |
| `podcodex.rag` | Chunking, embedding, vector storage (SQLite), hybrid retrieval |
| `podcodex.cli` | CLI: `init / rss / import / vectorize / sync / query / list / delete / validate / enrich` |
| `podcodex.core.transcribe` | WhisperX transcription + alignment + speaker diarization |
| `podcodex.core.polish` | LLM-based transcript correction (proper nouns, spelling, punctuation) |
| `podcodex.core.translate` | LLM-based translation (Ollama, OpenAI-compatible API, or manual) |
| `podcodex.core.synthesize` | Qwen3-TTS voice cloning + episode assembly |
| `podcodex.core.versions` | Generation versioning - save, load, list, prune pipeline output versions |
| `podcodex.core.pipeline_db` | Per-show SQLite DB for episode status and version metadata |
| `podcodex.ingest` | Folder scanning, RSS feed parsing, transcript import, show metadata |

### Output files

Outputs are organised per episode under the show folder. Every pipeline save (transcribe, polish, translate, manual edit) creates a **version** - a timestamped JSON snapshot stored in `.versions/`. Metadata (provenance, content hash, model, params) lives in `pipeline.db`. The most recent version is shown by default; users can browse and restore any version from the History dropdown.

Legacy flat files (`{stem}.transcript.json`, etc.) are still written alongside versions for backward compatibility.

```text
/shows/my_podcast/
├── show.toml                          ← show settings (name, RSS, language, speakers)
├── pipeline.db                        ← per-show SQLite (episode status + version metadata)
├── .feed_cache.json                   ← cached RSS feed data
├── ep01.mp3
├── ep01/
│   ├── .episode_meta.json             ← RSS episode metadata (title, pub date, etc.)
│   ├── ep01.segments.parquet          ← WhisperX raw segments
│   ├── ep01.segments.meta.json
│   ├── ep01.diarization.parquet       ← pyannote diarization
│   ├── ep01.diarization.meta.json
│   ├── ep01.diarized_segments.parquet
│   ├── ep01.speaker_map.json
│   ├── .versions/                     ← versioned pipeline outputs (primary store)
│   │   ├── transcript/
│   │   │   ├── 20260401T103000Z_raw.json
│   │   │   └── 20260401T120000Z_validated.json
│   │   ├── polished/
│   │   │   └── ...
│   │   └── english/
│   │       └── ...
│   ├── ep01.transcript.raw.json       ← legacy copy (pipeline export)
│   ├── ep01.transcript.json           ← legacy copy (validated)
│   ├── ep01.polished.raw.json         ← legacy copy
│   ├── ep01.polished.json             ← legacy copy
│   ├── ep01.english.raw.json          ← legacy copy
│   ├── ep01.english.json              ← legacy copy
│   ├── ep01.synthesized.wav           ← assembled episode
│   ├── voice_samples/
│   └── tts_segments/
├── vectors.db                         ← SQLite (embedded chunks, show-level)
└── ep02/
    └── ...
```

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the detailed plan.

| Feature | Status |
|---------|--------|
| Multi-store Zustand, UI component library, WaveSurfer waveform | Done |
| Platform abstraction, model cache, export (text/SRT/VTT/ZIP) | Done |
| Episode duration filter, per-episode playback speed | Done |
| Tauri backend sidecar (dev mode) | Done |
| Batch pipeline, global task bar, per-step config, speakers panel, move folder | Done |
| Generation versioning - N versions per pipeline step with provenance | Done |
| YouTube support - channel/playlist import, audio download, subtitle transcript | Done |
| Pipeline DB - per-show SQLite, step status comparison, provenance tracking | Done |
| **Semi-automatic speaker mapping** - voice embeddings for auto speaker ID | Next |
| **Simple mode** - zero-config end-to-end pipeline for non-technical users | Planned |
| **Standalone distribution** - PyInstaller sidecar for `.app`/`.deb`/`.exe` | Planned |
| **Timeline editor** - multi-track assembly with jingle/music insertion | Future |

## Notes

- Transcription runs on CPU on Apple Silicon (WhisperX does not support MPS yet)
- Diarization requires a valid `HF_TOKEN` and model access on Hugging Face
- Synthesis (Qwen3-TTS) is GPU-heavy - recommended on CUDA for production use
- Ollama translation works best with models >= 14B for reliable JSON output
- All embeddings are saved locally in `vectors.db` (SQLite) - search uses numpy brute-force over this store
