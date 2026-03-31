# Roadmap

## Completed

**Core architecture** — Zustand stores, platform abstraction (web + Tauri), FastAPI backend with WebSocket progress

**UI/UX** — WaveSurfer waveform, segment editor, audio player with per-episode speed, episode filtering (duration/title), list + card views, model cache management, VRAM monitoring, export (text/SRT/VTT/ZIP)

**Tauri integration (dev mode)** — backend sidecar, health-check, window state persistence, CORS

**Batch pipeline** — multi-episode operations with per-step config dialogs, global task bar with per-episode logs, show speakers panel, move/rename folder, pipeline config store, duration-based LLM batching, task cancellation + result summaries

**Search & bot** — RAG module (Qdrant + SQLite, hybrid retrieval), CLI (`vectorize / sync / query / list / delete`), Discord bot (`/search`, `/exact`, `/random`, `/stats`, `/episodes`)

**Ingest** — RSS feed parsing, episode download, folder scanning, transcript import

---

## Next Up

### Phase L: Generation Versioning

**Goal**: keep N versions of each pipeline step output with full provenance.

- Timestamp, model name/size, pipeline parameters per version
- Content hash, manual edit flag, parent version link
- Version picker per pipeline step in UI
- Schema change that other features (speaker entity, simple mode) can build on

### Phase M: Speaker Entity + Auto-Mapping

**Goal**: auto-generate `speaker_map.json` for new episodes without manual intervention.

- Show-level speaker registry: name, language, avatar, voice samples, synthesis config
- Cross-episode identity — same speaker recognized across all episodes of a show
- Voice embedding computation (Resemblyzer or pyannote SpeakerEmbedding)
- Reference database built from manually-labeled episodes
- Cosine similarity matching with confidence threshold
- Bootstrapping: a few labeled episodes build the reference DB — one-time cost for fixed-cast podcasts

### Phase N: Simple Mode

**Goal**: "I have audio, give me searchable knowledge" — zero-config end-to-end pipeline for non-technical users.

- Hardware auto-detection (GPU/CPU, VRAM) to pick optimal Whisper model and settings
- One-click flow: drop audio → transcribe → export → index
- Searchable transcripts ready to query with no manual configuration
- Optional opt-in to advanced steps (diarization, polish, translate, synthesis)

### Phase O: Frontend Polish

- **Service registry for LLM providers** — typed provider objects instead of growing conditionals in LLMControls
- **Auto-generate TS types from Pydantic** — prevent API type drift
- **Command palette (cmdk)** — quick navigation to episodes, pipeline steps, shows
- **Drag-and-drop file import** — Tauri file drop events to add episodes

### Phase P: Standalone Distribution

PyInstaller sidecar to bundle the Python backend. `make build` produces shareable `.app`/`.deb`/`.exe`.

---

## Parallel: Discord Bot Improvements

Can happen alongside any phase.

- Conversation context (thread-based or stateful per-user)
- LLM-synthesized answers on top of retrieved chunks
- Multi-show cross-collection search

---

## Future

### Timeline Editor

**Goal**: multi-track assembly with jingle/music reinsertion for final episode production.

- Drag-and-drop segment reordering
- Insert jingles, intros, outros, music beds
- Per-segment volume/fade controls
- Export assembled episode as single audio file
