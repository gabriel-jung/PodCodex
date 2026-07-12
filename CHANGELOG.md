# Changelog

## [0.2.6] - 2026-07-12

### One search engine for app, bot, and MCP

The app API, the Discord bot, and the MCP server now resolve shows and run
searches through a single shared service, so behavior and results match
across all three surfaces.

- Per-show RAG preferences in `show.toml` (`rag_model`, `rag_chunker`) are
  honored everywhere; previously only the MCP server read them. Selection
  order per show: explicit request, show preference, server default, global
  default, then the first indexed collection, so a show indexed under a
  non-default model always stays reachable.
- Requesting a model a show is not indexed under falls back through that
  chain instead of returning nothing.

### Fuzzy search fix

Exact search was silently incomplete on text indexes built by older
releases: typo tolerance did nothing, and matches inside longer words were
missed ("William" returned fewer results than "Williams"). Affected indexes
are rebuilt once per collection on the first search after upgrading (about a
second per show, logged); newly indexed collections are unaffected.

### Discord bot: /stats redesign

- Per-show breakdown, newest episode first: episode count, indexed duration,
  and newest episode date per show, with totals in a single line and
  humanized durations ("46h 12m").
- A single-show scope (only one show indexed, or one picked via the show
  filter) also shows the top speakers with talk time and the show artwork.

## [0.2.5] - 2026-07-11

### Episode speakers

Speakers are now visible without opening a transcript.

- Episode overview shows the speaker list with per-speaker talk-time share,
  computed from the canonical transcript (verified version first, then the
  best corrected, then the newest transcript). Music and gaps are not
  attributed, so shares can sum to under 100%.
- The show's episode list gains a speakers column (list and card views),
  ordered by talk time, fed by one cached roster scan shared with the
  Speakers tab.
- Renaming speakers, saving edits, deleting versions, finishing a
  transcribe/correct run, or moving the verified pointer refreshes every
  speaker view immediately.

### Episode navigation

Previous/next arrows in the episode page header walk the show's episodes in
date order, matching the episode list's default ordering.

### Broadcast numbers in the app

The `broadcast_number_pattern` regex is now editable in the show settings,
with a live preview against the show's latest episode title: extracted
number, no-match hint, and explicit errors for invalid patterns or a missing
capture group. Reindex episodes to apply a changed pattern.

### Discord bot

- Lean result cards: quote first, show as author, engine numbers behind an
  ephemeral Details button; `/exact` opens as a list, `/search` as a card,
  with a persistent toggle between the two.
- Media-aware results: cover thumbnails, timestamped YouTube jump links, and
  RSS listen links (media metadata is baked at index time; reindex to light
  them up on existing shows).
- `/announcements`: opt-in channel for new-episode and bot-version
  announcements, with per-guild access filtering.

### Fixes

- Timestamps (`start_hms`) now truncate to the second instead of rounding,
  so a citation never points past the passage it marks and matches
  floor-based external conventions.
- Speaker aliases (introduced in 0.2.4) are removed: wrong names are fixed
  at the speaker map step instead of being remapped at read time.
- The speaker roster resolves each episode's canonical transcript (verified
  pointer included) via two bulk queries and is cached server-side.

## [0.2.4] - 2026-07-11

### MCP retrieval

New tools and options that cut the cost of programmatic transcript mining
(entity counts, timestamp lookups, whole-episode reads) without pulling full
chunk text.

- `exact_count`: count literal phrase occurrences per episode across a batch of
  queries, with no chunk text. Fuzzy near-typo matches are excluded so counts
  stay trustworthy; optional `first_hit` returns the earliest occurrence.
- `get_context` accepts `at_time` (seconds or `1h09m46` / `69m46`) as an
  alternative to `chunk_index`, resolving to the chunk covering that moment.
- `get_episode` can return a lightweight `chunk_map` (positions and times, no
  text) and a raw `transcript` (`[MmSS] Speaker: text` lines), both from a
  single episode load.
- `list_episodes` gains a `broadcast_number` filter, a `fields` projection to
  skip heavy descriptions, and defaults to sorting by publication date.
- Every result carries a preformatted `start_hms` timestamp for citation.

### Broadcast numbers

Shows can set a `broadcast_number_pattern` regex to extract an airing number
from the episode title at index time, distinct from the per-season episode
number.

## [0.2.3] - 2026-06-22

### Verified versions

Mark any transcript or corrected version as the episode's verified (final)
source. The verified version becomes the canonical input for translate,
search indexing, and synthesis, ahead of the usual latest-version cascade.

- Set or clear it from the transcript/correct editor's version bar (star).
- Surfaced read-only everywhere else: a rose "verified" marker on the episode
  list status, the show progress strip count, the episode overview preview, and
  the "All transcript versions" table. The transcript preview and version rows
  now open the exact version you click.

### Per-task AI models

Each show remembers a separate AI model per mode, so correction and translation
can run on different models without re-picking each time.

### Fixes

- Speakers tab scrolls to the bottom when a show has many speakers.

## [0.2.2] - 2026-05-26

### Transcript editor polish

- Auto-follow the playing segment; scroll disables, **Now playing** re-engages.
- `Ctrl+Space` / `Shift+Space` toggle play/pause without leaving the textarea.
- Gap dividers between segments: subtle (10s+), prominent (60s+).
- Edited textareas tinted so unsaved changes stand out.
- Wider speaker dropdown so long names and the menu are not clipped.
- Split inherits the chip-renamed speaker; merge no longer drops it.

## [0.2.1] - 2026-05-20

### Settings rework

Every show now has its own pipeline settings in its Settings tab:
transcription model, speakers, AI provider/key/model, show context,
translation language, search-index model and chunker. Each field falls
back to the app-wide default, so existing shows keep working unchanged.

Episode panels show a small banner when a setting diverges from the
show's saved values, with **Save to show** and **Reset** buttons. Panel
edits stay per-run unless you push them.

### Fixes

- MCP / AI search now finds shows indexed under a non-default embedding
  model. Each show resolves to a single collection from its own setting.
- Discord bot passwords can be configured on registered shows before
  they are indexed.

First public release. PodCodex turns podcasts (and any audio source: YouTube
channels, local recordings) into a local, searchable knowledge base. Everything
runs on your machine; cloud LLMs are optional.

### Pipeline

- **Ingest:** subscribe to RSS feeds via Apple Podcasts search or direct URL,
  follow YouTube channels (audio + subtitles), or point at a folder of files.
- **Transcribe:** WhisperX with optional pyannote diarization, multiple model
  sizes per hardware tier. Import existing `.srt` / `.vtt` files or pull
  YouTube auto-subtitles to skip transcription.
- **Correct:** synced-audio inline editor for manual fixes, plus an optional
  LLM cleanup pass (local Ollama, hosted API keys, or copy-paste into your
  usual chat).
- **Index and search:** Chonkie chunker, LanceDB vector store, choice of
  BGE-M3 / E5 / Perplexity embedders. Semantic search and exact-phrase
  search across the full library, with timestamps.
- **Translate** (optional): same LLM backends as correct, into any target
  language.
- **Synthesize** (optional): Qwen3-TTS voice cloning to generate dubbed audio
  from a translated transcript while preserving the original speaker's voice.

### Integrations

- **Discord bot:** self-hostable, slash commands for semantic search, exact
  phrase, random samples, episode browsing, speaker stats. Per-server
  passwords when running multiple shows on one bot.
- **MCP server:** any MCP-compatible client (Claude Desktop, Claude Code,
  Cursor, Continue, Zed) can search the archive mid-conversation. One-click
  Claude Desktop setup from Settings.

### Platform

- Native desktop bundles for macOS (Apple Silicon) and Windows x64.
- CPU-only by default; NVIDIA users activate the in-app CUDA backend from
  Settings. Pascal GPU support via the `gpu-pascal` source extra.
- Per-step versioned outputs with a SQLite provenance database, so any run
  can be rolled back and any setting can be re-tried without losing prior
  results.
