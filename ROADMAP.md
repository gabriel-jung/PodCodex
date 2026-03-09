# Roadmap

## Semi-automatic speaker mapping

**Goal**: auto-generate `speaker_map.json` for new episodes without manual UI intervention.

### What's already in place
- `extract_voice_samples()` (`core/synthesize.py`) — extracts per-speaker audio clips from a diarized episode, ready to feed a reference database
- `save_speaker_map()` / `load_speaker_map()` (`core/transcribe.py`) — the map is already wired into the pipeline
- `assign_speakers_to_file()` — `SPEAKER_XX` labels are ready to match against

### What needs to be built (`core/identify.py`)
- **Embedding computation** — Resemblyzer or pyannote `SpeakerEmbedding`
- **Reference database** — `{speaker_name: embedding}` built from manually-labeled episodes
- **Matching logic** — cosine similarity + confidence threshold → `{SPEAKER_00: "Name", ...}`
- **Entry point** — takes a diarized episode + reference DB, writes `speaker_map.json`

### Bootstrapping
A few manually-labeled episodes are needed first to build the reference database.
For podcasts with a small fixed cast this is a one-time cost.
