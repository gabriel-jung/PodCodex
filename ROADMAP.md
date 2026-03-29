# Roadmap

## In Progress: Desktop App Improvements

8 improvements organized into 7 phases. Phases A–F are complete; Phase G is next.

### Completed

| Phase | What | Status |
|-------|------|--------|
| A | **Multi-store Zustand** — split single `useAppStore` into 8 domain-sliced stores (audio, episode, show, config, ui, pipeline, editor, search) | Done |
| B | **Eliminate prop drilling** — pipeline panels read episode/showMeta from `episodeStore` instead of props | Done |
| C | **UI patterns + accent color** — CircleButton, SettingRow/Section components, EmptyState dashed variant, accent color with distinct chroma | Done |
| D | **WaveSurfer.js waveform** — bar-style waveform in AudioBar replacing the seek slider, theme-aware colors via MutationObserver | Done |
| E | **Platform abstraction** — `PlatformProvider` with interfaces for FS, window, lifecycle; web fallbacks; Tauri detection ready | Done |
| F | **Backend — model cache + export** — centralized model cache (`~/.podcodex/models/`), VRAM monitoring, SRT/VTT/text/ZIP export endpoints | Done |

### Next Up

| Phase | What | Status |
|-------|------|--------|
| G | **Frontend — model panel + export UI** — Settings page with ModelCachePanel (cache dir, VRAM bar, model table), export dropdown in EditorToolbar, ZIP download in episode header | Done |
| H | **Episode filtering** — min-duration filter (persisted), hide short bonus episodes | Done |

---

## Semi-automatic Speaker Mapping

**Goal**: auto-generate `speaker_map.json` for new episodes without manual UI intervention.

### What's already in place
- `extract_voice_samples()` (`core/synthesize.py`) — extracts per-speaker audio clips from a diarized episode
- `save_speaker_map()` / `load_speaker_map()` (`core/transcribe.py`) — the map is already wired into the pipeline
- `assign_speakers()` — `SPEAKER_XX` labels are ready to match against

### What needs to be built (`core/identify.py`)
- **Embedding computation** — Resemblyzer or pyannote `SpeakerEmbedding`
- **Reference database** — `{speaker_name: embedding}` built from manually-labeled episodes
- **Matching logic** — cosine similarity + confidence threshold → `{SPEAKER_00: "Name", ...}`
- **Entry point** — takes a diarized episode + reference DB, writes `speaker_map.json`
- **Show-level speaker registry** — first-class Speaker entity with name, voice embedding, metadata

### Bootstrapping
A few manually-labeled episodes are needed first to build the reference database.
For podcasts with a small fixed cast this is a one-time cost.

---

## Generation Versioning

**Goal**: keep N versions of each pipeline step output with full provenance.

### What to track per version
- Timestamp, model name/size, pipeline parameters
- Content hash (detect what actually changed)
- Manual edit flag + editor identity
- Parent version (for diff/rollback)

### Design considerations
- Store versions alongside current output (e.g. `.polished.v1.json`, `.polished.v2.json`) or in a manifest
- UI needs a version picker per pipeline step
- Requires backend schema changes — deeper rework, not just endpoint additions

---

## Timeline Editor

**Goal**: multi-track assembly with jingle/music reinsertion for final episode production.

### What this enables
- Drag-and-drop segment reordering
- Insert jingles, intros, outros, music beds between segments
- Per-segment volume/fade controls
- Export assembled episode as single audio file

### Dependencies
- WaveSurfer.js regions plugin (Phase D provides the foundation)
- Audio mixing backend (ffmpeg or similar)
- Deeper frontend state management for multi-track editing

---

## App Architecture (Settled)

| Layer | Choice | Notes |
|---|---|---|
| Desktop shell | Tauri | Thin launcher — spawns backend, opens webview |
| Frontend | React + TypeScript | Tailwind, WaveSurfer for waveforms |
| Backend | FastAPI | Wraps the full pipeline; works local or remote |
| Storage | SQLite + Qdrant (optional) | Qdrant only needed for RAG |
| Apple Silicon | MLX for inference | Better perf than PyTorch on M-series |

The FastAPI backend is API-first: the Tauri app points at `localhost` by default, but a user can point it at a remote GPU server instead.

### Install tiers

Users install only what they need via `uv sync --extra`:

| What they want | Extras | Additional requirements |
|---|---|---|
| Pipeline only (transcribe, translate, synthesize) | `pipeline` | HuggingFace token (pyannote licence) |
| + RAG / vectorize / search | `pipeline,rag` | Docker (Qdrant) |
| + Discord bot | `bot` | A server to run it on |

The app detects which extras are installed at startup and shows/hides features accordingly.

### Dev setup

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

---

## What's Done

| Item | Files |
|------|-------|
| RAG module: chunker, embedder, store, retriever | `src/podcodex/rag/` |
| SQLite LocalStore (source of truth, skip re-embed) | `src/podcodex/rag/localstore.py` |
| CLI: `podcodex vectorize / sync / query / list / delete` | `src/podcodex/cli.py` |
| Discord bot: `/search`, `/exact`, `/stats`, `/episodes`, `/setup`, `/sync` | `src/podcodex/bot/bot.py` |
| Paginated results, rich embeds, expand-in-context view | `src/podcodex/bot/ui.py` |
| Full test suite passing (239 tests) | `tests/` |
| RAG layer polish (incremental sync, BM25 persistence, dedup) | `src/podcodex/rag/` |
| Model cache management (`~/.podcodex/models/`) | `src/podcodex/core/cache.py` |
| Export endpoints (text, SRT, VTT, ZIP) | `src/podcodex/api/routes/export.py` |

---

## Discord Bot Improvements

### Done
- Richer embeds with score bars, highlighted matches, speaker info
- Paginated results with prev/next buttons
- Expand-in-context view (surrounding chunks)
- `/exact` mention counts ("N mentions in M chunks")

### Still open
- Conversation context: stateless today; follow-up questions would be nice
- LLM-synthesised answers on top of retrieved chunks
- Multi-show cross-collection search

---

## Key Design Decisions Pending

1. **Bot conversation model** — thread-based vs stateful per-user? (unblocks bot context feature)
