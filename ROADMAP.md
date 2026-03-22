# Roadmap

## Semi-automatic speaker mapping

**Goal**: auto-generate `speaker_map.json` for new episodes without manual UI intervention.

### What's already in place
- `extract_voice_samples()` (`core/synthesize.py`) — extracts per-speaker audio clips from a diarized episode, ready to feed a reference database
- `save_speaker_map()` / `load_speaker_map()` (`core/transcribe.py`) — the map is already wired into the pipeline
- `assign_speakers()` — `SPEAKER_XX` labels are ready to match against

### What needs to be built (`core/identify.py`)
- **Embedding computation** — Resemblyzer or pyannote `SpeakerEmbedding`
- **Reference database** — `{speaker_name: embedding}` built from manually-labeled episodes
- **Matching logic** — cosine similarity + confidence threshold → `{SPEAKER_00: "Name", ...}`
- **Entry point** — takes a diarized episode + reference DB, writes `speaker_map.json`

### Bootstrapping
A few manually-labeled episodes are needed first to build the reference database.
For podcasts with a small fixed cast this is a one-time cost.

---

## What's Done ✓

| Item | Files |
|------|-------|
| RAG module: chunker, embedder, store, retriever | `src/podcodex/rag/` |
| SQLite LocalStore (source of truth, skip re-embed) | `src/podcodex/rag/localstore.py` |
| CLI: `podcodex vectorize / sync / query / list / delete` | `src/podcodex/cli.py` |
| Discord bot: `/search`, `/exact`, `/stats`, `/episodes`, `/setup`, `/sync` | `src/podcodex/bot/bot.py` |
| Paginated results, rich embeds, expand-in-context view | `src/podcodex/bot/ui.py` |
| Full test suite passing (239 tests) | `tests/` |
| Notebook `rag_dev.ipynb` updated to current API | `../Notebooks/rag_dev.ipynb` |

---

## Discord Bot Improvements

### Done ✓
- Richer embeds with score bars, highlighted matches, speaker info
- Paginated results with prev/next buttons
- Expand-in-context view (surrounding chunks)
- `/exact` mention counts ("N mentions in M chunks")
- `/random` command (WIP, uncommitted)

### Still open
- Conversation context: stateless today; follow-up questions would be nice
- LLM-synthesised answers on top of retrieved chunks
- Multi-show cross-collection search

---

## App Replacement (Streamlit → Desktop)

**Why:** Streamlit was a useful prototype shell. Not suited for a proper product: no real state management, poor UX control, hard to share.

### Architecture (settled)

| Layer | Choice | Notes |
|---|---|---|
| Desktop shell | Tauri | Thin launcher — spawns backend, opens webview |
| Frontend | React + TypeScript | Tailwind, WaveSurfer for waveforms |
| Backend | FastAPI | Wraps the full pipeline; works local or remote |
| Storage | SQLite + Qdrant (optional) | Qdrant only needed for RAG |
| Apple Silicon | MLX for inference | Better perf than PyTorch on M-series |

The FastAPI backend is API-first: the Tauri app points at `localhost` by default, but a user can point it at a remote GPU server instead. No code changes needed to support either mode.

**Reference:** [Voicebox](https://github.com/jamiepine/voicebox) (MIT) uses this exact stack. We won't fork it, but use it as a reference for Tauri config, Makefile, and React shell.

### Install tiers

Users install only what they need via `uv sync --extra`:

| What they want | Extras | Additional requirements |
|---|---|---|
| Pipeline only (transcribe, translate, synthesize) | `pipeline` | HuggingFace token (pyannote licence) |
| + RAG / vectorize / search | `pipeline,rag` | Docker (Qdrant) |
| + Discord bot | `bot` | A server to run it on |

The app detects which extras are installed at startup and shows/hides features accordingly.

### First-time setup flow

1. Download and install Docker Desktop _(optional — only for RAG)_
2. Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Download PodCodex release zip, move app to Applications
4. Double-click `Setup.command` — runs `uv sync`, pulls Qdrant image if Docker present, creates `.env` template
5. Add HuggingFace token to `.env` _(one-time; requires accepting pyannote licence on HF website)_
6. Double-click `PodCodex.app` — starts backend, opens UI

### Day-to-day

Double-click `PodCodex.app`. Starts Qdrant container (if installed), starts FastAPI backend, opens webview. Quit app → cleans up.

### Screens (replacing Streamlit)

- Episode management / status dashboard _(new)_
- Transcription + diarization
- Speaker map editor
- Translation / polish
- TTS synthesis
- RAG search _(hidden if `rag` extra not installed)_

### Dev setup

Makefile orchestrates everything (following Voicebox pattern):

```
make setup   # uv sync + frontend deps + Rust toolchain check
make dev     # starts FastAPI + Vite + Tauri in watch mode
make build   # production .app bundle
```

---

## RSS Ingest

**Goal:** handle shows at the RSS feed level — fetch episodes directly from a feed URL instead of managing audio files manually.

### What this enables
- Auto-discover new episodes from a feed, download audio, queue for transcription
- Populate episode metadata (title, date, description, guests) from RSS tags
- Pre-fill `show_name`, `episode` fields used in transcript metadata and RAG
- Potentially auto-trigger the pipeline for new episodes (background job)

### What needs to be built (`core/rss.py` or `ingest/rss.py`)
- Feed fetch + parse (title, enclosure URL, pub date, description, guid)
- Episode download with resume support (large audio files)
- Deduplication by guid against already-downloaded episodes
- Integration with `scan_folder()` / `EpisodeInfo` so the dashboard reflects RSS state

**Design not settled yet — discuss before implementing.**

---

## RAG Layer Polish ✓

All items resolved:

- ~~**Incremental sync**~~ ✓ — `podcodex sync` works
- ~~**`--db` UX**~~ ✓ — covered by `PODCODEX_DB` env var + `--db` flag
- ~~**BM25 persistence**~~ ✓ — BM25 index cached in retriever
- ~~**Multi-episode deduplication**~~ — not a real issue; `source` filter handles it

---

## Key Design Decisions Pending

1. **Bot conversation model** — thread-based vs stateful per-user? (unblocks bot context feature)
