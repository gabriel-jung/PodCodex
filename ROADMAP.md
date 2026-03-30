# Roadmap

## Completed: Desktop App (Phases A–J)

10 UI/desktop improvement phases — all complete.

| Phase | What |
|-------|------|
| A | **Multi-store Zustand** — 8 domain-sliced stores (audio, episode, show, config, ui, pipeline, editor, search) |
| B | **Eliminate prop drilling** — panels read from stores instead of props |
| C | **UI patterns + accent color** — CircleButton, SettingRow/Section, EmptyState, accent color |
| D | **WaveSurfer.js waveform** — bar-style waveform in AudioBar, theme-aware |
| E | **Platform abstraction** — PlatformProvider with web + Tauri implementations |
| F | **Backend — model cache + export** — centralized cache, VRAM monitoring, SRT/VTT/text/ZIP export |
| G | **Frontend — model panel + export UI** — ModelCachePanel, export dropdown, ZIP download |
| H | **Episode filtering + AudioBar UX** — duration filter, toggleable time, per-episode speed, blurred artwork |
| I | **Frontend cleanup** — component extraction, shared hooks, dead code removal |
| J | **Tauri integration** — backend sidecar, health-check, window lifecycle, CORS |
| K | **Batch pipeline** — multi-episode operations, global task bar, sortable episode list, move folder, duration-based LLM batching |

---

## Next Up

### Phase L: Speaker Entity + Auto-Mapping

**Goal**: auto-generate `speaker_map.json` for new episodes without manual intervention.

#### Speaker as first-class entity
- Show-level speaker registry: name, language, avatar, voice samples, synthesis config
- Cross-episode identity — same speaker recognized across all episodes of a show
- Backend data model in `core/identify.py`

#### Auto-mapping pipeline
- **Embedding computation** — Resemblyzer or pyannote SpeakerEmbedding
- **Reference database** — `{speaker_name: embedding}` built from manually-labeled episodes
- **Matching logic** — cosine similarity + confidence threshold → `{SPEAKER_00: "Name", ...}`
- **Entry point** — takes diarized episode + reference DB, writes `speaker_map.json`

#### What's already in place
- `extract_voice_samples()` (`core/synthesize.py`)
- `save_speaker_map()` / `load_speaker_map()` (`core/transcribe.py`)
- `assign_speakers()` — `SPEAKER_XX` labels ready to match

#### Bootstrapping
A few manually-labeled episodes build the reference database. One-time cost for fixed-cast podcasts.

### Phase M: Standalone Distribution

PyInstaller sidecar to bundle the Python backend. `make build` produces shareable `.app`/`.deb`/`.exe`.

### Phase N: Frontend Polish

- **Service registry for LLM providers** — typed provider objects instead of growing conditionals in LLMControls
- **Auto-generate TS types from Pydantic** — prevent API type drift
- **Command palette (cmdk)** — quick navigation to episodes, pipeline steps, shows
- **Drag-and-drop file import** — Tauri file drop events to add episodes

---

## Future

### Generation Versioning

**Goal**: keep N versions of each pipeline step output with full provenance.

- Timestamp, model name/size, pipeline parameters
- Content hash, manual edit flag, parent version link
- Version picker per pipeline step in UI
- Significant schema change — do after speaker entity settles the data model

### Timeline Editor

**Goal**: multi-track assembly with jingle/music reinsertion for final episode production.

- Drag-and-drop segment reordering
- Insert jingles, intros, outros, music beds
- Per-segment volume/fade controls
- Export assembled episode as single audio file

### Discord Bot Improvements

- Conversation context (thread-based or stateful per-user)
- LLM-synthesised answers on top of retrieved chunks
- Multi-show cross-collection search

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

| What they want | Extras | Additional requirements |
|---|---|---|
| Pipeline only (transcribe, translate, synthesize) | `pipeline` | HuggingFace token (pyannote licence) |
| + RAG / vectorize / search | `pipeline,rag` | Docker (Qdrant) |
| + Discord bot | `bot` | A server to run it on |

### Dev setup

```
make setup   # uv sync + frontend deps + Rust toolchain check
make dev     # starts FastAPI + Vite + Tauri in watch mode
make build   # production .app bundle
```

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
| Window state persistence | `tauri-plugin-window-state` |
| RSS ingest (feed parse, download, dedup by guid) | `src/podcodex/api/routes/rss.py`, `src/podcodex/ingest/` |
