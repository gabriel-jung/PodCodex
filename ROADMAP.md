# Roadmap

Forward-looking only. Shipped features live in the README. Per-change history will live in `CHANGELOG.md` once 0.2.0 is cut.

## v0.1.0 release blockers

Tag when all items are done.

- [ ] **Next up:** license updates for clean MIT redistribution (some bundled deps are GPL-licensed)

## Next

### Speaker auto-mapping

Auto-generate `speaker_map.json` for new episodes via voice embeddings. Show-level registry with cross-episode identity, cosine similarity over Resemblyzer / pyannote embeddings, bootstrapped from a handful of labeled episodes.

### Onboarding follow-ups

The wizard + home empty-state ship. Remaining:
- Hardware auto-detection (GPU / CPU / VRAM) to pick the default Whisper preset
- ShowPage empty-state CTA into the pipeline preset flow
- Replay wizard from Settings for users who dismissed it

### Developer experience

- Transcription engine abstraction: decouple `core/transcribe.py` from WhisperX (Voxtral, Whisper.cpp, cloud ASR)

## Parallel tracks

- Discord bot conversation context (thread-based or per-user stateful)

## Future

### Timeline editor

Multi-track assembly with jingle / music reinsertion: drag-to-reorder segments, insert intros / outros / music beds, per-segment volume + fade, single-file audio export.
