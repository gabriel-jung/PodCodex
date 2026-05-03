# PodCodex Architecture

Non-obvious wiring. For folder layout and what each module contains, run `ls src/podcodex/`.

## Process topology

```
┌─────────────────────────────────────────────────────────┐
│  Tauri shell (Rust, src-tauri/)                         │
│  ─ Native window, file dialogs, IPC                     │
│  ─ Spawns sidecar in a process group (command-group)    │
└────────────────┬────────────────────────────────────────┘
                 │ stdout/stderr + HTTP :18811
                 ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI sidecar (PyInstaller-frozen `podcodex-server`) │
│  ─ Routes (api/), WebSocket progress channel            │
│  ─ Owns pipeline DB, version archive, Lance index       │
│  ─ Forks subprocesses for heavy steps                   │
└────────────────┬────────────────────────────────────────┘
                 │ multiprocessing.Queue (prog_q)
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Step worker subprocesses                               │
│  ─ transcribe / diarize / correct / translate / synth   │
│  ─ Re-exec into a fresh Python so torch state is clean  │
│  ─ Loguru lines forwarded to parent via prog_q          │
└─────────────────────────────────────────────────────────┘
```

**Why subprocess workers:** torch + CUDA leak GPU memory across runs in-process. Re-execing isolates each step. The Tauri process group ensures workers die when the app quits — without it, orphaned torch processes survive the shell.

**Bootstrap order matters.** `bootstrap.py` patches must run before any `torch.*` import. Required env vars (`PODCODEX_DATA_DIR`, `HF_HOME`, `TORCH_HOME`) must be set before `bootstrap_for_*()`.

## Data layout

`<data_dir>` is platform-resolved by `core/app_paths.py`:

| Platform | `<data_dir>` |
|---|---|
| macOS | `~/Library/Application Support/podcodex/` |
| Windows | `%APPDATA%\podcodex\` |
| Linux | `~/.local/share/podcodex/` |

User config (`secrets.env`, etc.) is separate and lives at `~/.config/podcodex/` on **all platforms** (`config_dir()` deliberately ignores XDG defaults to keep paths symmetric — config is small, data is big).

Each show is a self-contained folder under a user-chosen root:

```
<show_root>/<show>/
├── .feed_cache.json             RSS / YouTube feed metadata (all known episodes)
├── audio/                       Source audio per episode
├── <stem>/                      One folder per episode
│   ├── .episode_meta.json       Per-episode RSSEpisode (indexer's RSS source)
│   ├── <stem>.transcript.json   Latest transcript (ASR output)
│   ├── <stem>.corrected.json    Latest LLM-corrected transcript
│   ├── <stem>.<lang>.json       Latest translation per language
│   ├── <stem>.synth.<lang>.wav  Latest synthesized audio
│   └── .versions/
│       ├── transcript/<id>.json   Every save, content-hashed
│       ├── corrected/<id>.json
│       ├── translation/<id>.json
│       └── synth/<id>.wav
├── pipeline.db                  Per-show SQLite (episodes + versions)
└── show.json                    Show config (RSS URL, defaults)
```

Files at the episode root are pointers to the latest version. `.versions/{step}/<id>.json` is the truth: every save (auto or manual) is archived with model, params, content hash, timestamp.

`.episode_meta.json` is the indexer's RSS-metadata source (title, pub_date, description, episode_number, artwork_url). It mirrors a single `RSSEpisode` from `.feed_cache.json`. Whenever a richer extraction lands (per-video YouTube call, RSS refetch, one-shot backfill) the merge goes through `fill_empty_fields()` in `ingest/rss.py` — three call sites pre-consolidation each rolled their own and drifted on which keys counted. Don't add a fourth.

### `pipeline.db` schema (per show)

```sql
episodes (
  stem TEXT PRIMARY KEY,
  audio_path TEXT,
  transcribed INTEGER, corrected INTEGER,
  indexed INTEGER, synthesized INTEGER,
  translations TEXT,             -- JSON array of language codes
  provenance TEXT,               -- JSON
  updated_at REAL
)
versions (
  id TEXT, stem TEXT, step TEXT,
  timestamp TEXT, type TEXT,     -- "auto" | "manual"
  model TEXT, params TEXT,       -- JSON
  manual_edit INTEGER,
  content_hash TEXT, input_hash TEXT,
  segment_count INTEGER,
  PRIMARY KEY (id, stem, step)
)
```

Step status (`transcribed`, `corrected`, …) is a count flag, not a boolean — increments on each save. `versions.input_hash` chains a step to the version it was derived from, enabling the version tree UI.

## RAG layer

All embeddings for all shows live in **one** LanceDB index at `<data_dir>/index/`. Collections within the index are named:

```
{show}__{model}__{chunker}
```

Example: `myshow__bge-m3__semantic`.

This means changing the embedding model or chunker creates a new collection rather than overwriting — old collections stick around until explicitly removed. The desktop app's Index step writes here; the bot and MCP server read.

**Truth-of-record:** indexed status comes from LanceDB itself, not from filesystem markers. `lance_indexed_stems()` returns the set of stems present in the index; `unified_episodes()` reconciles this against the per-show `pipeline.db` on each call. There is no `.rag_indexed` marker file.

**Hybrid retrieval:** vector ANN (cosine on embeddings) + BM25 full-text on the raw segment text, fused with reciprocal rank. Both indexes are maintained inside the single LanceDB table per collection.

## Frontend ↔ backend type sync

Pydantic request/response models in `src/podcodex/api/` are the source of truth. Run `make types` to regenerate `frontend/src/api/types.ts`. The frontend's API client (`createVersionApi`, `createLLMPipelineApi`) consumes these types.

Don't hand-edit `frontend/src/api/types.ts` — it's overwritten by `make types`. Pydantic models inherit from `LLMRequest` for any endpoint that talks to an LLM; that base carries model, params, and provider routing.

## Bot and MCP

Both consume the same shared retriever. They are read-only — neither builds the index. The bot resolves the index path via `_resolve_default_index_path()` in `rag/index_store.py` (PODCODEX_INDEX env > `<data_dir>/index/` > `./deploy/index/` > `./index/`). MCP server runs over stdio for Claude Desktop and over HTTP at `/mcp` on the same uvicorn process for other clients.

Detailed deploy guides: `deploy/BOT.md`, `deploy/MCP.md`.
