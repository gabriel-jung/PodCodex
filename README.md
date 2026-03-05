# podcodex

AI tools for podcast production — transcription, translation, voice synthesis, and semantic search.

## Modules

| Module | Description |
|--------|-------------|
| `podcodex.core.transcribe` | WhisperX transcription + phonetic alignment + speaker diarization |
| `podcodex.core.translate` | LLM-based transcript translation (Ollama, OpenAI-compatible API, or manual) |
| `podcodex.core.synthesize` | Qwen3-TTS voice cloning + episode assembly |
| `podcodex.rag` | Chunking, embedding, vector storage, and hybrid retrieval |
| `podcodex.ingest` | Scan a show folder and report per-episode processing status |

## Pipeline

```
Audio → transcribe → translate → synthesize
                ↓
           vectorize → query
```

## System Requirements

```bash
# macOS
brew install ffmpeg sox

# Ubuntu/Debian
sudo apt install ffmpeg sox
```

## Installation

```bash
git clone https://github.com/gabriel-jung/podcodex
cd podcodex
uv sync
```

## Environment Variables

Create a `.env` file at the root:

```env
HF_TOKEN=your_huggingface_token   # required for pyannote diarization
API_KEY=your_api_key              # any OpenAI-compatible provider (Mistral, OpenAI, etc.)
QDRANT_URL=http://localhost:6333  # optional, defaults to localhost:6333
```

`HF_TOKEN` requires accepting the pyannote model terms on Hugging Face:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Usage

### Transcription

```python
from podcodex.core import transcribe

transcribe.transcribe_file("episode.mp3", output_dir="ep01/")
transcribe.diarize_file("episode.mp3", output_dir="ep01/")
transcribe.assign_speakers_to_file("episode.mp3", output_dir="ep01/")

transcribe.save_speaker_map("episode.mp3", {
    "SPEAKER_00": "Alice",
    "SPEAKER_01": "Bob",
}, output_dir="ep01/")

transcript = transcribe.export_transcript("episode.mp3", output_dir="ep01/",
                                          show="Total Trax", episode="ep01")
print(transcribe.transcript_to_text(transcript[:5]))

# Check pipeline status
transcribe.processing_status("episode.mp3", output_dir="ep01/")
```

### Translation

```python
from podcodex.core import translate

segments = transcribe.load_transcript("episode.mp3", output_dir="ep01/")

# Via API (Mistral, OpenAI, or any compatible provider)
translated = translate.translate_segments(
    segments,
    mode="api",
    context="French podcast about film music",
    source_lang="French",
    target_lang="English",
    model="mistral-small-latest",
    api_base_url="https://api.mistral.ai/v1",
)

# Via Ollama (local — works best with 14B+ models)
translated = translate.translate_segments(
    segments,
    mode="ollama",
    model="qwen3:14b",
    context="French podcast about film music",
)

# Manual copy/paste workflow
print(translate.build_manual_prompt(segments, context="French podcast about film music"))
translated = translate.translate_segments(json_from_llm, mode="manual")

translate.save_translation("episode.mp3", translated, output_dir="ep01/")
print(translate.translation_to_text(translated[:5], lang="both"))
```

### Synthesis

```python
from podcodex.core import synthesize

voice_samples = synthesize.extract_voice_samples(
    "episode.mp3", translated, output_dir="ep01/", min_duration=5.0, top_k=3,
)

generated = synthesize.generate_segments(
    "episode.mp3", translated, voice_samples, output_dir="ep01/",
    model_size="1.7B", sample_index={"Alice": 0, "Bob": 1},
)

episode = synthesize.assemble_episode(
    generated, "episode.mp3", output_dir="ep01/", strategy="original_timing",
)
```

### RAG — vectorize & query

Requires Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`) or set `QDRANT_URL`.

```bash
# Embed and store a transcript
podcodex vectorize ep01/ep01.transcript.json --show "Total Trax" --strategy bge_speaker

# Semantic search
podcodex query "film music composer" --show "Total Trax" --strategy bge_speaker
podcodex query "film music composer" --show "Total Trax" --strategy bge_speaker --top-k 3

# Manage collections
podcodex list --show "Total Trax"
podcodex delete total_trax__bge_speaker
```

#### Embedding strategies

| Strategy | Chunker | Embedder | Search |
|----------|---------|----------|--------|
| `bge_speaker` | speaker turns | BAAI/bge-m3 (1024-dim) | hybrid RRF |
| `bge_semantic` | semantic | BAAI/bge-m3 (1024-dim) | hybrid RRF |
| `e5_semantic` | semantic | multilingual-e5-small (384-dim) | dense cosine |
| `pplx_context` | speaker turns | pplx-embed-context (1024-dim) | dense cosine |

#### In Python

```python
from podcodex.rag.store import QdrantStore, collection_name
from podcodex.rag.retriever import Retriever

store = QdrantStore()                          # connects to QDRANT_URL or localhost:6333
store = QdrantStore(in_memory=True)            # no server needed, for notebooks

col = collection_name("Total Trax", "bge_speaker")   # → "total_trax__bge_speaker"
store.create_collection(col, "bge_speaker")

retriever = Retriever("bge_speaker", store=store)
results = retriever.retrieve("film music", col, top_k=5)
```

### Streamlit app

```bash
# Single-episode view
streamlit run streamlit/app.py

# Show-level view — folder of episodes with processing status
streamlit run streamlit/show_app.py
```

The show-level view scans a folder of audio files and displays per-episode status
(`🔴 Pending` / `🟡 Transcribed` / `🟢 Indexed`). Opening an episode loads it into
the full pipeline tabs (Transcribe → Translate → Synthesize).

## Output Files

Outputs are organised per episode under the show folder:

```
/shows/total_trax/
├── ep01.mp3
├── ep02.mp3
├── ep01/
│   ├── ep01.segments.parquet
│   ├── ep01.segments.meta.json
│   ├── ep01.diarization.parquet
│   ├── ep01.diarization.meta.json
│   ├── ep01.diarized_segments.parquet
│   ├── ep01.speaker_map.json
│   ├── ep01.transcript.json
│   ├── ep01.translated.json
│   ├── ep01.synthesized.wav
│   ├── .rag_indexed              ← marker set by `podcodex vectorize`
│   ├── voice_samples/
│   └── tts_segments/
└── ep02/
    └── ...
```

## Translation Modes

| Mode | Description | Best for |
|------|-------------|----------|
| `ollama` | Local LLM via Ollama | Privacy, offline — use 14B+ models |
| `api` | OpenAI-compatible API | Best quality (Mistral, OpenAI, etc.) |
| `manual` | Copy/paste via LLM UI | Full control, best quality |

## Notes

- Transcription runs on CPU on Apple Silicon (WhisperX does not support MPS yet)
- Diarization requires a valid `HF_TOKEN` and model access on Hugging Face
- Synthesis (Qwen3-TTS) is GPU-heavy — recommended on CUDA for production use
- Ollama translation works best with models ≥ 14B for reliable JSON output
