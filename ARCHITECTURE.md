# PodCodex Architecture

Non-obvious wiring. For folder layout and what each module contains, run `ls src/podcodex/`.

## Process topology

```
┌─────────────────────────────────────────────────────────┐
│  Tauri shell (Rust, src-tauri/)                         │
│  - Native window, file dialogs, IPC                     │
│  - Spawns sidecar in a process group (command-group)    │
└────────────────┬────────────────────────────────────────┘
                 │ stdout/stderr + HTTP :18811
                 ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI sidecar (PyInstaller-frozen `podcodex-server`) │
│  - Routes (api/), WebSocket progress channel            │
│  - Owns pipeline DB, version archive, Lance index       │
│  - Forks subprocesses for heavy steps                   │
└────────────────┬────────────────────────────────────────┘
                 │ multiprocessing.Queue (prog_q)
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Step worker subprocesses                               │
│  - transcribe (+diarize) / synthesize / index           │
│  - Re-exec into a fresh Python so torch state is clean  │
│  - Loguru lines forwarded to parent via prog_q          │
└─────────────────────────────────────────────────────────┘
```

LLM correct and translate run in the API process: PodCodex only makes HTTP calls, to a hosted API or to a local Ollama daemon. With Ollama the model may well run on the local GPU, but inside the Ollama process, which manages its own memory, so there is no torch state here to isolate.

**Why subprocess workers:** torch + CUDA leak GPU memory across runs in-process. Re-execing isolates each step. The Tauri process group ensures workers die when the app quits; without it, orphaned torch processes survive the shell.

**Bootstrap order matters.** Every entry point (sidecar, MCP stdio, dev server, bot, step worker) calls one `bootstrap_for_*()` before any `torch.*` import. Its first step, `core/cache.py:wire_model_caches`, sets the HF and torch cache vars, because `huggingface_hub` reads them once at import. `PODCODEX_DATA_DIR` is set by the Tauri shell and optional elsewhere; if used, it must be in the environment before bootstrap.

## Data layout

`<data_dir>` is platform-resolved by `core/app_paths.py`; `PODCODEX_DATA_DIR` overrides every row (the Tauri shell sets it):

| Platform | `<data_dir>` |
|---|---|
| macOS | `~/Library/Application Support/podcodex/` |
| Windows | `%APPDATA%\podcodex\` |
| Linux | `$XDG_DATA_HOME/podcodex/`, else `~/.local/share/podcodex/` |

User config (`secrets.env`, etc.) is separate and lives at `$XDG_CONFIG_HOME/podcodex/`, else `~/.config/podcodex/`, on **all platforms**, Windows included (`config_dir()` skips the per-OS app-support dirs to keep paths symmetric: config is small, data is big). The Tauri shell and the Vite proxy resolve the same path.

Each show is a self-contained folder under a user-chosen root:

```
<show_root>/<show>/
├── .feed_cache.json                       RSS / YouTube feed metadata (all known episodes)
├── <stem>/                                One folder per episode
│   ├── <stem>.mp3                         Source audio (may live here or alongside)
│   ├── .episode_meta.json                 Per-episode RSSEpisode (indexer's RSS source)
│   ├── voice_samples/                     Reference clips per speaker (for TTS cloning)
│   ├── tts_segments/                      Per-segment generated audio + manifest.json (scratch dir during assemble)
│   ├── llm_failures.json                  Rejected LLM batches per step (correct / translate)
│   ├── transcript/<id>.json               Every transcript save (raw ASR or validated export)
│   ├── transcript/segments/<id>.parquet   Word-level ASR segments (parquet substep)
│   ├── transcript/diarization/<id>.parquet        Pyannote diarization output (parquet substep)
│   ├── transcript/diarized_segments/<id>.parquet  Segments + speaker assignment merged (parquet substep)
│   ├── speaker_map/<id>.json              Diarization label to speaker name mapping
│   ├── corrected/<id>.json                Every LLM-corrected save
│   ├── <lang>/<id>.json                   Every translation save per language (e.g. english/)
│   └── synthesize/<id>.wav                Every assembled episode synthesis
├── pipeline.db                            Per-show SQLite (episodes + versions)
└── show.toml                              Show config (stable show id, RSS URL, defaults)
```

`show.toml` carries the stable show `id` (`{slug}_{8 hex}`) minted by `save_show_meta`. That id, not the display `name`, is what collections, bot passwords and `.podcodex` manifests key on, so renaming a show is a label change only.

Every step uses the same storage layout: `{ep_dir}/{step}/{version_id}.{json|parquet|wav}` resolved by `version_path(base, step, id)` in `core/versions.py`, except the parquet substeps, which nest under `transcript/`. Versions are content-hashed; metadata (model, params, timestamp, segment count, input hash for lineage) lives in the `versions` table of `pipeline.db`. The `versions` table is the truth; the directory listing is incidental.

`.episode_meta.json` is the indexer's RSS-metadata source (title, pub_date, description, episode_number, artwork_url). It mirrors a single `RSSEpisode` from `.feed_cache.json`. Whenever a richer extraction lands (per-video YouTube call, RSS refetch, one-shot backfill), the merge goes through `fill_empty_fields()` in `ingest/rss.py`. Three call sites pre-consolidation each rolled their own and drifted on which keys counted. Don't add a fourth.

### `pipeline.db` schema (per show)

```sql
episodes (
  stem TEXT PRIMARY KEY,
  audio_path TEXT,
  transcribed INTEGER, corrected INTEGER,
  indexed INTEGER, synthesized INTEGER,
  translations TEXT,             -- JSON array of language codes
  provenance TEXT,               -- JSON
  verified_step TEXT,            -- verified pointer: the version the user
  verified_version_id TEXT,      --   marked as the reference for this episode
  updated_at REAL
)
versions (
  id TEXT, stem TEXT, step TEXT,
  timestamp TEXT, type TEXT,     -- "raw" | "validated" (validated = hand-edited)
  model TEXT, params TEXT,       -- JSON
  manual_edit INTEGER,
  content_hash TEXT, input_hash TEXT,
  segment_count INTEGER,
  PRIMARY KEY (id, stem, step)
)
```

Step status (`transcribed`, `corrected`, `synthesized`, and entries in the `translations` JSON array) is a boolean flag: a step counts as done when it has a version file on disk. `versions.input_hash` chains a step to the version it was derived from, enabling the version tree UI. The canonical source for downstream steps is resolved by `resolve_canonical_ref`: the verified pointer if set, else the edited-first newest `corrected` version, else the newest transcript.

The journal mode is `DELETE`, not WAL: WAL needs shared memory next to the database, which is unsafe when the show folder sits in a synced or network folder.

**Version lifecycle is symmetric across all steps.** Every save flows through `save_version` (or `save_synthesize_version` for the `.wav` content-hash variant); every delete flows through `delete_version`. `delete_version` removes the on-disk file at `version_path`, removes the DB row via `pipeline_db.delete_versions`, then runs `_refresh_status_after_delete` which demotes the matching boolean flag (or trims the `translations` array) once no versions remain for that step. `shows.py` `_load_status_context` (behind `/unified` and `/status`) also reconciles every `STEP_FLAG` flag and the `translations` list in both directions on "has a version file on disk", skipping stems whose scan was incomplete, which guards against any path that bypassed the helper. Files, not rows: a DB bootstrapped from a folder scan has files and no rows yet, and a row whose file is gone (backfill keeps it through a grace period) cannot be read. Status flags promote AND demote. Adding a new step means: an entry in `PIPELINE_STEPS` (a step missing there is treated as a translation language), in `STEP_FLAG` if it has a status flag, in `PARQUET_STEPS` or `WAV_STEPS` if it is not JSON, and the four touchpoints (path, save, delete, status refresh).

## Environment variables

Every variable the backend reads. None is required for a desktop install; the Tauri shell sets the ones it needs.

| Variable | Read by | Effect |
|---|---|---|
| `PODCODEX_DATA_DIR` | `core/app_paths.py` | Overrides the platform data dir. Set by the Tauri shell and the bot's Dockerfile. |
| `PODCODEX_CACHE_DIR` | `core/cache.py` | Moves the whole model cache (HF, torch, sentence-transformers) out of `<data_dir>/models`. |
| `PODCODEX_HF_OFFLINE` | `core/cache.py` | `1` sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`, skipping Hub round-trips when every model is cached. |
| `PODCODEX_INDEX` | `rag/index_store.py`, `core/recovery.py` | Uses another LanceDB index instead of `<data_dir>/index`. |
| `PODCODEX_DEVICE` | `core/device.py` | `auto`, `cpu` or `cuda`; see CLAUDE.md. |
| `PODCODEX_MACHINE_ID` | `core/machine_id.py` | Fixes the machine identity used for index ownership (containers without a persistent data dir). |
| `PODCODEX_FFMPEG_EXE` | `core/_ffmpeg.py` | Path to the ffmpeg binary; the Tauri shell points it at the bundled one. |
| `PODCODEX_API_TOKEN` | `api/api_token.py` | Loopback API token; overrides the persisted one in the config dir (tests pin it). |
| `PODCODEX_API_PORT`, `PODCODEX_PARENT_PID` | `api/app.py`, `api/server.py` | Sidecar port, and the shell PID the sidecar watches so it exits with the app. Set by Tauri. |
| `PODCODEX_GPU_MANIFEST_URL` | `api/gpu_backend.py` | Alternative manifest for the optional CUDA backend download. |
| `OLLAMA_HOST` | `core/llm.py` | Ollama daemon URL, default `http://localhost:11434`. |
| `HF_TOKEN` | `core/transcribe.py` | Hugging Face token for the gated pyannote diarization model. |
| `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `SENTENCE_TRANSFORMERS_HOME` | libraries | Set by `wire_model_caches` at bootstrap; see ML_RUNTIME.md. |

## RAG layer

All embeddings for all shows live in **one** LanceDB index at `<data_dir>/index/`. A collection name is built from the show **id** (`{show_id}__{model}__{chunker}`, e.g. `myshow_3f2a9c11__bge-m3__semantic`), so a rename never orphans one. LanceDB OSS cannot rename a table, so collections created before ids existed keep their name-derived table names and carry a `show_id` column in the `_collections` metadata table instead; the one-time migration in `rag/show_id_migration.py` backfills that column on the first `IndexStore` open.

Because of those two shapes, a table name is never something a caller reconstructs. `rag/store.collection_name` is internal to `IndexStore`; readers go through `resolve_collection`, writers through `ensure_collection_for_show` (which resolves before creating). Rebuilding the name from a show's display name is exactly the bug the id migration exists to close.

Changing the embedding model or chunker creates a new collection rather than overwriting; old collections stick around until explicitly removed. The desktop app's Index step writes here; the bot and MCP server read.

**Truth-of-record:** indexed status comes from LanceDB itself, not from filesystem markers. `lance_indexed_stems()` returns the set of stems present in the index; `unified_episodes()` reconciles this against the per-show `pipeline.db` on each call. There is no `.rag_indexed` marker file.

**Hybrid retrieval:** vector ANN (cosine on embeddings) + BM25 full-text on the raw segment text, fused with reciprocal rank. Both indexes are maintained inside the single LanceDB table per collection.

**Compaction:** Lance tables are copy-on-write, so every re-index (a delete plus an add per episode), every `pub_date`/`episode_title` backfill and every metadata heal leaves the previous data files behind. `IndexStore.compact()` runs `optimize()` over the chunk tables the process wrote to, reclaiming them while keeping a week of history so an open reader elsewhere is unaffected. It is called at the end of an index job, of a batch index, and of `podcodex-reindex`, never on a read path.

**Shared search service:** all three query surfaces (the desktop app's HTTP API, the Discord bot, the MCP server) resolve shows to collections and fan queries across them through one module, `podcodex.rag.search_service`. Surfaces keep their own transport, access control, and response shaping; the service owns collection picking, per-model query encoding, cross-collection merging, and result ordering. `resolve_collections()` picks one collection per show from `IndexStore.get_all_collection_info()`, in this precedence, each rung skipped when no collection matches: an explicit override (a caller-supplied model/chunker, e.g. a user's request params) beats the show's `show.toml` RAG preference (`load_show_rag_prefs()`) beats a caller-supplied default beats the global `DEFAULT_MODEL`/`DEFAULT_CHUNKING` combo beats the first collection by sorted name. That last rung keeps a show reachable even when it's indexed only under a non-default model. `hybrid_search()`, `exact_search()`, and `random_quote()` then query the resolved collections; a `ValueError` from the retriever (bad filter, dim mismatch) re-raises, any other per-collection failure is logged and skipped so one broken table never blanks the whole answer.

## Frontend ↔ backend type sync

Pydantic request/response models in `src/podcodex/api/` are the source of truth. Run `make types` to regenerate `frontend/src/api/generated-types.ts`. The frontend's API client (`createVersionApi`, `createLLMPipelineApi`) consumes these types.

Don't hand-edit `frontend/src/api/generated-types.ts`; it's overwritten by `make types`. `frontend/src/api/types.ts` is the hand-maintained layer next to it: it re-exports the generated names and defines the frontend-only types that have no Pydantic model, so that is the file to edit when a type is not backed by the API. Pydantic models inherit from `LLMRequest` for any endpoint that talks to an LLM; that base carries model, params, and provider routing.

## Bot and MCP

Both consume the same shared search service (see **Shared search service** above). They are read-only; neither builds the index. The bot resolves the index path via `_resolve_default_index_path()` in `rag/index_store.py` (PODCODEX_INDEX env > `<data_dir>/index/` > `./deploy/index/` > `./index/`). MCP server runs over stdio for Claude Desktop. The same uvicorn process also exposes HTTP at `/mcp` for other clients.

Detailed deploy guides: `deploy/BOT.md`, `deploy/MCP.md`.
