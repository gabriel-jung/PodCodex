# podcodex

AI tools for podcast production — transcription, polishing, translation, voice synthesis, and semantic search.

The **Discord bot** lets anyone search through your podcast transcripts. The **pipeline** handles the processing: transcribe, polish, translate, synthesize, and index.

## Discord bot

Search your podcast transcripts from Discord with slash commands.

**Commands:**

- `/search <question>` — find relevant passages by meaning (semantic) or keywords
- `/exact <query>` — literal text search, like Ctrl+F
- `/stats` — overview of indexed shows, episodes, and duration
- `/episodes [show]` — list episodes (auto-selects if only one show)
- `/setup` — configure server defaults (admin)
- `/sync` — manually sync the command tree (admin)

All search commands support optional `show`, `episode`, `speaker`, `source`, and `compact` filters.

### Setting up the Discord application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**
2. Under **Bot**, click **Reset Token** and copy it — this is your `DISCORD_TOKEN`
3. Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent** (needed for the bot to read messages if you add message-based features later)
4. Under **OAuth2 → URL Generator**:
   - **Scopes**: select `bot` and `applications.commands`
   - **Bot Permissions**: select `Send Messages`, `Embed Links`, `Use Slash Commands`
5. Copy the generated URL and open it in your browser to invite the bot to your server

### Running the bot locally

```bash
# Required env vars
DISCORD_TOKEN=your_bot_token
QDRANT_URL=http://localhost:6333   # Qdrant must be running for search

# Start
podcodex-bot --model bge-m3 --chunking semantic --top-k 5
podcodex-bot --dev-guild 123456789  # instant command sync for development
```

### Deploying to a server

The `deploy/` directory contains everything needed to run the bot + Qdrant on a VPS (tested on Ubuntu 4GB RAM). Only Docker is required on the server — no Python, no GPU.

#### 1. Clone and configure

```bash
ssh user@vps
git clone https://github.com/gabriel-jung/podcodex ~/podcodex
cd ~/podcodex
cp deploy/.env.example deploy/.env.production
# Edit deploy/.env.production and set DISCORD_TOKEN
```

#### 2. Copy your vectors database

```bash
# From your local machine
scp /path/to/vectors.db user@vps:~/podcodex/deploy/data/
```

#### 3. Build and seed

```bash
cd ~/podcodex

# Build the bot image (includes BGE-M3 model download — takes a few minutes)
docker compose -f deploy/docker-compose.yml build

# Start Qdrant first
docker compose -f deploy/docker-compose.yml up -d qdrant

# Seed Qdrant from your local SQLite database
docker compose -f deploy/docker-compose.yml run --rm --entrypoint podcodex bot \
    sync --db /app/data/vectors.db --show "My Podcast"
```

#### 4. Start the bot

```bash
docker compose -f deploy/docker-compose.yml up -d
```

The bot image installs only core + bot dependencies (no pipeline), keeping it under 3GB. Qdrant is limited to 512MB. Logs rotate automatically (50MB × 3 files). The bot auto-restarts on crash or reboot.

#### Updating

**New code:**

```bash
cd ~/podcodex
git pull
docker compose -f deploy/docker-compose.yml build bot
docker compose -f deploy/docker-compose.yml up -d
```

**New episodes indexed locally:**

```bash
# From your local machine
scp /path/to/vectors.db user@vps:~/podcodex/deploy/data/

# On the VPS
docker compose -f deploy/docker-compose.yml run --rm --entrypoint podcodex bot \
    sync --db /app/data/vectors.db --show "My Podcast"
```

No manual stop needed — `up -d` replaces the running container automatically.

**Checking logs:**

```bash
docker compose -f deploy/docker-compose.yml logs bot --tail 50
docker compose -f deploy/docker-compose.yml logs -f bot        # follow live
docker compose -f deploy/docker-compose.yml ps                 # container status
```

## Pipeline

The processing pipeline runs locally and requires more setup than the bot.

```text
Audio → transcribe → polish → translate → synthesize
                       ↓
                  vectorize → query
```

### System requirements

These are only needed if you run the pipeline locally (not needed for the bot deployment).

```bash
# macOS
brew install ffmpeg sox

# Ubuntu/Debian
sudo apt install ffmpeg sox
```

### Installation

```bash
git clone https://github.com/gabriel-jung/podcodex
cd podcodex

# Bot only (search + Discord)
uv pip install -e ".[bot]"

# Full install (pipeline + bot + app)
uv pip install -e ".[bot,pipeline,app]"
```

### Environment variables

Create a `.env` file at the root. Only set what you need:

```env
# Bot
DISCORD_TOKEN=your_bot_token      # required for the Discord bot

# Search
QDRANT_URL=http://localhost:6333  # optional, defaults to localhost:6333
PODCODEX_DB=/path/to/vectors.db   # optional, overrides default SQLite location

# Pipeline
HF_TOKEN=your_huggingface_token   # required for speaker diarization
API_KEY=your_api_key              # any OpenAI-compatible provider (Mistral, OpenAI, etc.)
```

`HF_TOKEN` requires accepting the pyannote model terms on Hugging Face:

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Transcription

```python
from podcodex.core import transcribe

transcribe.transcribe_file("episode.mp3", output_dir="ep01/")
transcribe.diarize_file("episode.mp3", output_dir="ep01/")
transcribe.assign_speakers("episode.mp3", output_dir="ep01/")

transcribe.save_speaker_map("episode.mp3", {
    "SPEAKER_00": "Alice",
    "SPEAKER_01": "Bob",
}, output_dir="ep01/")

transcript = transcribe.export_transcript("episode.mp3", output_dir="ep01/",
                                          show="My Podcast", episode="ep01")

# Check pipeline status
transcribe.processing_status("episode.mp3", output_dir="ep01/")
```

### Polishing

```python
from podcodex.core import polish

segments = transcribe.load_transcript("episode.mp3", output_dir="ep01/")

# Via API
polished = polish.polish_segments(
    segments, mode="api", source_lang="French",
    context="French podcast about film music, hosted by Alice and Bob",
    model="mistral-small-latest",
    api_base_url="https://api.mistral.ai/v1",
)

# Via Ollama
polished = polish.polish_segments(
    segments, mode="ollama", model="qwen3:14b", source_lang="French",
)

polish.save_polished_raw("episode.mp3", polished, output_dir="ep01/")
# After review:
polish.save_polished("episode.mp3", polished, output_dir="ep01/")
```

### Translation

```python
from podcodex.core import translate

segments = polish.load_polished("episode.mp3", output_dir="ep01/")

# Via API
translated = translate.translate_segments(
    segments, mode="api",
    source_lang="French", target_lang="English",
    context="French podcast about film music",
    model="mistral-small-latest",
    api_base_url="https://api.mistral.ai/v1",
)

# Via Ollama
translated = translate.translate_segments(
    segments, mode="ollama", model="qwen3:14b",
    source_lang="French", target_lang="English",
)

# Manual copy/paste workflow
batches = translate.build_manual_prompts_batched(
    segments, source_lang="French", target_lang="English",
)
# ... paste each batch into an LLM, collect results ...
translated = translate.translate_segments(json_from_llm, mode="manual")

translate.save_translation_raw("episode.mp3", translated, "english", output_dir="ep01/")
```

#### Translation modes

| Mode | Description | Best for |
|------|-------------|----------|
| `ollama` | Local LLM via Ollama | Privacy, offline — use 14B+ models |
| `api` | OpenAI-compatible API | Best quality (Mistral, OpenAI, etc.) |
| `manual` | Copy/paste via LLM UI | Full control, best quality |

### Synthesis

```python
from podcodex.core import synthesize

voice_samples = synthesize.extract_voice_samples(
    "episode.mp3", translated, output_dir="ep01/", min_duration=5.0, top_k=3,
)

model = synthesize.load_tts_model(model_size="1.7B")
clone_prompts = synthesize.build_clone_prompts(model, voice_samples, sample_index=0)

# Generate one segment at a time
for i, seg in enumerate(translated):
    output_path = Path(f"ep01/tts_segments/{i:04d}_{seg['speaker']}.wav")
    result = synthesize.generate_segment(
        model, seg, clone_prompts, output_path, language="English",
    )

# Assemble into final episode
episode_path = synthesize.assemble_episode(
    generated, "episode.mp3", output_dir="ep01/", strategy="original_timing",
)
```

### Indexing & search

Indexing saves everything locally in a SQLite file. To enable search, you also need Qdrant running — start it with `docker compose up -d` or set `QDRANT_URL`.

```bash
# Embed and store a transcript
podcodex vectorize ep01/ep01.transcript.json --show "My Podcast"
podcodex vectorize ep01/ep01.transcript.json --show "My Podcast" --model bge-m3 --chunking semantic

# Query
podcodex query "film music composer" --show "My Podcast"
podcodex query "film music composer" --show "My Podcast" --top-k 3 --alpha 0.7

# Sync SQLite → Qdrant (rebuild without re-embedding)
podcodex sync --show "My Podcast"
podcodex sync --db /path/to/vectors.db

# Manage collections
podcodex list
podcodex list --show "My Podcast"
podcodex delete my_podcast__bge-m3__semantic
```

#### Embedding models

| Model key | Embedder | Dim | Search |
|-----------|----------|-----|--------|
| `bge-m3` | BAAI/bge-m3 | 1024 | hybrid (dense + sparse via RRF) |
| `e5-small` | multilingual-e5-small | 384 | dense cosine |
| `e5-large` | multilingual-e5-large | 1024 | dense cosine |
| `pplx` | pplx-embed-context | 1024 | dense cosine |

#### Chunking strategies

| Strategy | Description |
|----------|-------------|
| `semantic` | Semantic similarity splitting (Chonkie) — recommended default |
| `speaker` | One chunk per speaker turn — fast, no extra deps |

Collection names follow the format `{show}__{model}__{chunker}`, e.g. `my_podcast__bge-m3__semantic`.

#### In Python

```python
from podcodex.rag.store import QdrantStore, collection_name
from podcodex.rag.retriever import Retriever

store = QdrantStore()                    # connects to QDRANT_URL or localhost:6333

col = collection_name("My Podcast", model="bge-m3", chunker="semantic")
retriever = Retriever(model="bge-m3")
results = retriever.retrieve("film music", col, top_k=5, alpha=0.5)

# Filter by episode, speaker, or source
results = retriever.retrieve(
    "film music", col, top_k=5, episode="ep01", speaker="Alice",
)
```

### Streamlit app

```bash
streamlit run streamlit/app.py
```

The app has two modes:

- **Simple mode** (default) — drop an audio file and go through the pipeline. Tabs: **Transcribe** → **Polish** → **Translate** → **Synthesize**.
- **Podcast mode** (toggle in sidebar) — manage a show folder with RSS feed integration, episode browsing, indexing, and search. Tabs: **Transcribe** → **Polish** → **Index** → **Translate** → **Synthesize** → **Search**.

#### Podcast mode

1. Set a **local folder** for the show in the sidebar
2. Search for a podcast by name (uses iTunes/Apple Podcasts directory) or paste an RSS feed URL
3. The show overview lists all episodes from the feed, with download and open buttons
4. Episodes can be opened without downloading audio — useful for importing transcripts from external sources
5. Per-episode pipeline status is shown in the sidebar and overview (transcribed, polished, indexed, etc.)
6. Show settings (name, RSS URL, language, speakers) are stored in `show.toml`

#### RSS episode metadata

When downloading from an RSS feed, episode metadata (title, pub date, description, episode/season number) is saved as `.episode_meta.json` in each episode's output directory. This title is displayed throughout the app instead of the filesystem slug.

### CLI — RSS and import

```bash
# Fetch RSS feed and list episodes (uses show.toml for the URL, or --rss to override)
podcodex rss <show_folder>
podcodex rss <show_folder> --rss https://example.com/feed.xml
podcodex rss <show_folder> --download          # also download audio files

# Import an external transcript
podcodex import transcript.json <show_folder>
podcodex import transcript.json <show_folder> --episode ep01 --show "My Podcast"
```

## Output files

Outputs are organised per episode under the show folder. Each step produces a `.raw.json` (pipeline output) and a `.json` (user-validated) version:

```text
/shows/my_podcast/
├── show.toml                          ← show settings (name, RSS, language, speakers)
├── .feed_cache.json                   ← cached RSS feed data
├── ep01.mp3
├── ep01/
│   ├── .episode_meta.json             ← RSS episode metadata (title, pub date, etc.)
│   ├── ep01.segments.parquet          ← WhisperX raw segments
│   ├── ep01.segments.meta.json
│   ├── ep01.diarization.parquet       ← pyannote diarization
│   ├── ep01.diarization.meta.json
│   ├── ep01.diarized_segments.parquet
│   ├── ep01.speaker_map.json
│   ├── ep01.transcript.raw.json       ← exported transcript (raw)
│   ├── ep01.transcript.json           ← validated transcript
│   ├── ep01.polished.raw.json         ← LLM-corrected (raw)
│   ├── ep01.polished.json             ← validated polished
│   ├── ep01.english.raw.json          ← translation (raw)
│   ├── ep01.english.json              ← validated translation
│   ├── ep01.synthesized.wav           ← assembled episode
│   ├── vectors.db                     ← SQLite (embedded chunks)
│   ├── voice_samples/
│   └── tts_segments/
└── ep02/
    └── ...
```

## Modules

| Module | Description |
|--------|-------------|
| `podcodex.bot` | Discord bot with `/search`, `/exact`, `/stats`, `/episodes` commands |
| `podcodex.rag` | Chunking, embedding, vector storage (Qdrant + SQLite), and hybrid retrieval |
| `podcodex.cli` | CLI: `podcodex vectorize / sync / query / list / delete` |
| `podcodex.core.transcribe` | WhisperX transcription + phonetic alignment + speaker diarization |
| `podcodex.core.polish` | LLM-based source correction (proper nouns, spelling, punctuation) |
| `podcodex.core.translate` | LLM-based transcript translation (Ollama, OpenAI-compatible API, or manual) |
| `podcodex.core.synthesize` | Qwen3-TTS voice cloning + episode assembly |
| `podcodex.ingest.folder` | Scan a show folder and report per-episode processing status |
| `podcodex.ingest.rss` | RSS feed parsing, episode metadata, audio download |
| `podcodex.ingest.importer` | Import external transcripts into the show folder structure |
| `podcodex.ingest.show` | Show-level metadata (`show.toml`) |

## Notes

- Transcription runs on CPU on Apple Silicon (WhisperX does not support MPS yet)
- Diarization requires a valid `HF_TOKEN` and model access on Hugging Face
- Synthesis (Qwen3-TTS) is GPU-heavy — recommended on CUDA for production use
- Ollama translation works best with models >= 14B for reliable JSON output
- All embeddings are saved locally in `vectors.db` (SQLite) — `podcodex sync` pushes them to Qdrant for search
