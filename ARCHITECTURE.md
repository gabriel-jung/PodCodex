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
│  - transcribe / diarize / correct / translate / synth   │
│  - Re-exec into a fresh Python so torch state is clean  │
│  - Loguru lines forwarded to parent via prog_q          │
└─────────────────────────────────────────────────────────┘
```

**Why subprocess workers:** torch + CUDA leak GPU memory across runs in-process. Re-execing isolates each step. The Tauri process group ensures workers die when the app quits; without it, orphaned torch processes survive the shell.

**Bootstrap order matters.** `bootstrap.py` patches must run before any `torch.*` import. Required env vars (`PODCODEX_DATA_DIR`, `HF_HOME`, `TORCH_HOME`) must be set before `bootstrap_for_*()`.

## Data layout

`<data_dir>` is platform-resolved by `core/app_paths.py`:

| Platform | `<data_dir>` |
|---|---|
| macOS | `~/Library/Application Support/podcodex/` |
| Windows | `%APPDATA%\podcodex\` |
| Linux | `~/.local/share/podcodex/` |

User config (`secrets.env`, etc.) is separate and lives at `~/.config/podcodex/` on **all platforms** (`config_dir()` deliberately ignores XDG defaults to keep paths symmetric: config is small, data is big).

Each show is a self-contained folder under a user-chosen root:

```
<show_root>/<show>/
├── .feed_cache.json                       RSS / YouTube feed metadata (all known episodes)
├── <stem>/                                One folder per episode
│   ├── <stem>.mp3                         Source audio (may live here or alongside)
│   ├── .episode_meta.json                 Per-episode RSSEpisode (indexer's RSS source)
│   ├── voice_samples/                     Reference clips per speaker (for TTS cloning)
│   ├── tts_segments/                      Per-segment generated audio + manifest.json (scratch dir during assemble)
│   ├── transcript/<id>.json               Every transcript save (raw ASR or validated export)
│   ├── segments/<id>.parquet              Word-level ASR segments (parquet substep)
│   ├── diarization/<id>.parquet           Pyannote diarization output (parquet substep)
│   ├── diarized_segments/<id>.parquet     Segments + speaker assignment merged (parquet substep)
│   ├── speaker_map/<id>.json              Diarization label to speaker name mapping
│   ├── corrected/<id>.json                Every LLM-corrected save
│   ├── <lang>/<id>.json                   Every translation save per language (e.g. english/)
│   └── synthesize/<id>.wav                Every assembled episode synthesis
├── pipeline.db                            Per-show SQLite (episodes + versions)
└── show.toml                              Show config (stable show id, RSS URL, defaults)
```

`show.toml` carries the stable show `id` (`{slug}_{8 hex}`) minted by `save_show_meta`. That id, not the display `name`, is what collections, bot passwords and `.podcodex` manifests key on, so renaming a show is a label change only.

Every step uses the same storage layout: `{ep_dir}/{step}/{version_id}.{json|parquet|wav}` resolved by `version_path(base, step, id)` in `core/versions.py`. Versions are content-hashed; metadata (model, params, timestamp, segment count, input hash for lineage) lives in the `versions` table of `pipeline.db`. The `versions` table is the truth; the directory listing is incidental.

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

Step status (`transcribed`, `corrected`, `synthesized`, and entries in the `translations` JSON array) is a boolean flag derived from the presence of versions for that step. `versions.input_hash` chains a step to the version it was derived from, enabling the version tree UI.

**Version lifecycle is symmetric across all steps.** Every save flows through `save_version` (or `save_synthesize_version` for the `.wav` content-hash variant); every delete flows through `delete_version`. `delete_version` removes the on-disk file at `version_path`, removes the DB row via `pipeline_db.delete_versions`, then runs `_refresh_status_after_delete` which demotes the matching boolean flag (or trims the `translations` array) once no versions remain for that step. The `shows.py` `unified_episodes` reconcile pass also demotes a stale `synthesized=True` if the versions table reports no rows, guarding against any path that bypassed the helper. Status flags promote AND demote: adding a new step means wiring all four touchpoints (path, save, delete, status refresh), nothing more.

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
