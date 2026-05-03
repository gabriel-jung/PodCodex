<p align="center">
  <img src="assets/icon.png" alt="PodCodex" width="160" />
</p>

# PodCodex

**Turn podcasts into a searchable knowledge base.**

With PodCodex you can produce high-quality transcriptions from any audio source, whether podcasts, YouTube channels, or your own recordings, and turn them into a local and searchable archive.

Plug it into a Discord bot or an MCP-compatible chat (Claude Desktop, Cursor, etc.) and the whole archive becomes a conversational knowledge base.

You can also translate the results into other languages, or synthesize dubbed audio through voice cloning.

> **Screenshots and a 30-second demo are landing with v0.2.0.** Tracked in [ROADMAP.md](ROADMAP.md).

---

## What it does

Point it at audio. Six steps, all on your machine:

1. **Ingest**
   - As in a regular podcast app: subscribe to any show available through Apple Podcasts search or any RSS URL.
   - Follow YouTube channels, with audio and subtitle downloads.
   - Or just point it at a folder of recordings you already have.

2. **Transcribe**
   - Turn audio into a transcript using [WhisperX](https://github.com/m-bain/whisperX), with optional speaker labels from [pyannote-audio](https://github.com/pyannote/pyannote-audio).
   - Different model sizes depending on your hardware (CPU or GPU) and speed-versus-quality needs.
   - Already have subtitles? Import them (.srt, .vtt) and skip transcription entirely.

3. **Correct**
   - Fix errors (text, speakers, timestamps) by hand in the transcript editor, with synced audio playback.
   - Or run an LLM cleanup pass: copy-paste into most LLM web chats (ChatGPT, Claude, etc.), point it at a local Ollama model, or use most LLM API keys.

4. **Index & search**
   - Search across your whole library by meaning or exact phrase.
   - Split each transcript into smaller pieces using [Chonkie](https://github.com/chonkie-inc/chonkie), and vectorize with [BGE-M3](https://huggingface.co/BAAI/bge-m3), [E5](https://huggingface.co/collections/intfloat/multilingual-e5-text-embeddings) or [Perplexity](https://huggingface.co/collections/perplexity-ai/pplx-embed) embedder models for semantic retrieval.
   - Local [LanceDB](https://github.com/lancedb/lancedb) index on your disk: stores the vectors and powers fast search.

5. **Translate** *(optional)*
   - Translate the transcript into any target language.
   - Uses the same backends as correction (web chat, local Ollama, or API key).

6. **Synthesize** *(optional)*
   - Generate dubbed audio in the target language with [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) voice cloning.
   - Keeps the original speaker's voice.

Each step is saved with its model and settings to a SQLite database. Roll back any time, or experiment with a different setup without losing earlier runs.

---

## Get it

### Pre-built release

Direct download (latest):

- **macOS (Apple Silicon)** — [PodCodex-macos-arm64.dmg](https://github.com/gabriel-jung/podcodex/releases/latest/download/PodCodex-macos-arm64.dmg)
- **Windows x64** — [PodCodex-windows-x64.msi](https://github.com/gabriel-jung/podcodex/releases/latest/download/PodCodex-windows-x64.msi)

All assets + checksums on the [Releases](https://github.com/gabriel-jung/podcodex/releases) page.

PodCodex shells out to a system [FFmpeg](https://ffmpeg.org/download.html) install for transcription, clip extraction, and synthesis — install it before first launch (the app's first-run check surfaces a dialog if missing).

**macOS quarantine on first launch.** The DMG is not yet signed/notarized, so Gatekeeper will say *"PodCodex.app is damaged and can't be opened"*. The app is fine. Drag it to `/Applications`, then once:

```bash
xattr -dr com.apple.quarantine /Applications/PodCodex.app
```

Subsequent launches don't need it. Signing + notarization is a v0.1.0 blocker.

### Build from source

You'll need:

- **[uv](https://docs.astral.sh/uv/)** — handles Python 3.12 install + dependencies.
- **[Node.js](https://nodejs.org/) LTS** — frontend (or via nvm, Homebrew, winget, etc.).
- **[FFmpeg](https://ffmpeg.org/download.html)** — system install on PATH (used for transcription, clip extraction, voice synthesis).
- **[Rust](https://www.rust-lang.org/)** *(optional)* — for the native Tauri window.

**Install:**

```bash
git clone https://github.com/gabriel-jung/podcodex && cd podcodex
make setup          # uv sync + npm install
```

**Run the app:**

Browser only (no Rust, faster to try):
```bash
make dev-no-tauri
```

Native Tauri window (needs Rust):
```bash
make dev
```

More on building and signing in [`deploy/BUILD.md`](deploy/BUILD.md).

### Hardware support

| Hardware                       | GPU support                                              |
|--------------------------------|----------------------------------------------------------|
| NVIDIA RTX 20xx or newer       | Bundled installer, or `--extra gpu` from source          |
| NVIDIA Pascal (GTX 10xx, P40)  | `--extra gpu-pascal` only, see [PASCAL.md](deploy/PASCAL.md) |
| Apple Silicon                  | CPU only, no GPU/MPS path yet                            |
| Other / no GPU                 | CPU only                                                 |

Force CPU at runtime via **Settings → GPU acceleration → Force CPU** (useful when the GPU build mismatches your card or for ad-hoc CPU testing).

---

## Integrations

### Discord bot

Share your archive with a small community. Drop the bot into any Discord server and listeners can run slash commands for semantic search (`/search`), exact phrase (`/exact`), random samples (`/random`), episode browsing (`/episodes`), speaker stats (`/speakers`), and library stats (`/stats`). Per-server passwords if you're running multiple shows on one bot.

```bash
uv sync --extra bot --extra rag
DISCORD_TOKEN=... uv run podcodex-bot
```

Full deploy guide (uv + Docker, systemd, password rotation, VPS rsync) in [`deploy/BOT.md`](deploy/BOT.md).

### MCP

PodCodex ships a Model Context Protocol server, so any MCP-compatible client (Claude Desktop, Claude Code, Cursor, Continue, Zed, and others) can search your archive mid-conversation. It exposes search, exact phrase, show listing, and context-fetch tools, plus editable slash prompts like `/brief`, `/speaker`, `/timeline`.

**Claude Desktop:** one-click setup at **Settings → Claude Desktop → Enable integration** writes the stdio config for you.

**Other clients:** point them at `http://127.0.0.1:18811/mcp` (HTTP transport, while the app is running), or run the server manually over stdio. See [`deploy/MCP.md`](deploy/MCP.md).

---

## Tech stack

| Layer          | Technology                                                                                                |
|----------------|-----------------------------------------------------------------------------------------------------------|
| Desktop shell  | [Tauri v2](https://tauri.app) (Rust)                                                                      |
| Frontend       | [React 19](https://react.dev), [Vite](https://vitejs.dev), [TypeScript](https://www.typescriptlang.org), [Tailwind](https://tailwindcss.com), [shadcn/ui](https://ui.shadcn.com) |
| Backend        | [FastAPI](https://fastapi.tiangolo.com) (REST + WebSocket, background tasks)                              |
| Ingest         | [yt-dlp](https://github.com/yt-dlp/yt-dlp), [feedparser](https://github.com/kurtmckee/feedparser), system [FFmpeg](https://ffmpeg.org/) |
| Transcription  | [WhisperX](https://github.com/m-bain/whisperX), [pyannote-audio](https://github.com/pyannote/pyannote-audio) |
| LLM            | [Ollama](https://ollama.com) (local), [OpenAI](https://openai.com), [Anthropic](https://anthropic.com), [Mistral](https://mistral.ai), [DeepSeek](https://deepseek.com), and others; or any OpenAI-compatible API |
| Voice cloning  | [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts)                                            |
| Search         | [LanceDB](https://github.com/lancedb/lancedb), [BGE-M3](https://huggingface.co/BAAI/bge-m3) / [E5](https://huggingface.co/collections/intfloat/multilingual-e5-text-embeddings) / [Perplexity](https://huggingface.co/collections/perplexity-ai/pplx-embed) embedders, [Chonkie](https://github.com/chonkie-inc/chonkie) chunker |
| State          | [SQLite](https://sqlite.org)                                                                              |

---

## Notes & caveats

- YouTube auto-generated subtitles need [deno](https://deno.com/) installed (yt-dlp hands the JS challenge off to it). Manual subtitles work fine without.
- Ollama correct/translate needs a model with reliable structured-output support; small models tend to break the JSON format.
- Qwen3-TTS needs CUDA for reasonable synthesis speed.

---

## Develop / contribute

Start with [CONTRIBUTING.md](CONTRIBUTING.md). System wiring in [ARCHITECTURE.md](ARCHITECTURE.md). AI assistant context in [CLAUDE.md](CLAUDE.md). Frontend design rules in [DESIGN.md](DESIGN.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for v0.1.0 blockers and longer-term plans.

## License

MIT. See [LICENSE](LICENSE).
