# Changelog

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

### Speaker aliases

Per-show speaker alias table in `show.toml` (`[speaker_aliases]`), applied at
read time so it fixes an existing index with no reindex. Aliased labels fold
together in `speaker_stats` and are matched by speaker filters.

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
