# podcodex

A podcast transcription and translation pipeline. Transcribes audio using WhisperX + pyannote diarization, translates to any language via LLM, and optionally re-voices episodes using Qwen3-TTS voice cloning.

## Pipeline

```
Audio → transcribe.py → translate.py → synthesize.py
```

1. **Transcribe** — WhisperX transcription + phonetic alignment + speaker diarization
2. **Translate** — LLM-based correction + translation (Ollama, OpenAI-compatible API, or manual)
3. **Synthesize** — Qwen3-TTS voice cloning per speaker + episode assembly

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
```

`HF_TOKEN` requires accepting the pyannote model terms on Hugging Face:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Usage

### Transcription

```python
from podcodex.core import transcribe

# Full pipeline
transcribe.transcribe_file("episode.mp3", output_dir="Transcriptions")
transcribe.diarize_file("episode.mp3", output_dir="Transcriptions")
transcribe.assign_speakers_to_file("episode.mp3", output_dir="Transcriptions")

# Map speaker labels to names
transcribe.save_speaker_map("episode.mp3", {
    "SPEAKER_00": "Alice",
    "SPEAKER_01": "Bob",
}, output_dir="Transcriptions")

# Export final transcript
transcript = transcribe.export_transcript("episode.mp3", output_dir="Transcriptions")
print(transcribe.transcript_to_text(transcript[:5]))

# Check pipeline status
transcribe.processing_status("episode.mp3", output_dir="Transcriptions")
```

### Translation

```python
from podcodex.core import translate

# Load transcript
segments = transcribe.load_transcript("episode.mp3", output_dir="Transcriptions")
simplified = transcribe.simplify_transcript(segments)

# Translate via API (Mistral, OpenAI, or any compatible provider)
translated = translate.translate_segments(
    simplified,
    mode="api",
    context="French podcast about film music",
    source_lang="French",
    target_lang="English",
    model="mistral-small-latest",
    api_base_url="https://api.mistral.ai/v1",
)

# Or via Ollama (local — works best with 14B+ models)
translated = translate.translate_segments(
    simplified,
    mode="ollama",
    model="qwen3:14b",
    context="French podcast about film music",
)

# Or manually via a LLM UI (copy/paste workflow)
print(translate.build_manual_prompt(simplified, context="French podcast about film music"))
# → paste output into your LLM UI, paste result back as JSON
translated = translate.translate_segments(json_from_llm, mode="manual")

# Save
translate.save_translation("episode.mp3", translated, output_dir="Transcriptions")

# Review
print(translate.translation_to_text(translated[:5], lang="both"))
```

### Synthesis

```python
from podcodex.core import synthesize

# Extract voice samples for cloning (one per speaker)
voice_samples = synthesize.extract_voice_samples(
    "episode.mp3",
    translated,
    output_dir="Transcriptions",
    min_duration=5.0,
    top_k=3,
)

# Generate TTS audio for each segment
generated = synthesize.generate_segments(
    "episode.mp3",
    translated,
    voice_samples,
    output_dir="Transcriptions",
    model_size="1.7B",
    sample_index={"Alice": 0, "Bob": 1},  # pick best voice sample per speaker
)

# Assemble final episode
episode = synthesize.assemble_episode(
    generated,
    "episode.mp3",
    output_dir="Transcriptions",
    strategy="original_timing",  # or "silence"
)
```

## Output Files

All outputs are saved relative to the audio file, in `output_dir` (default: same folder as audio).

```
Transcriptions/
├── episode.segments.parquet
├── episode.segments.meta.json
├── episode.diarization.parquet
├── episode.diarization.meta.json
├── episode.diarized_segments.parquet
├── episode.speaker_map.json
├── episode.transcript.json
├── episode.translated.json
├── episode.synthesized.wav
├── voice_samples/
│   ├── Alice_00.wav
│   └── Bob_00.wav
└── tts_segments/
    ├── 0000_Alice.wav
    ├── 0001_Bob.wav
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
