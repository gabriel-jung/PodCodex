# Changelog

## [0.2.0] - 2026-05-13

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
