# Roadmap

Forward-looking only. Shipped features live in the README. Per-change history will live in `CHANGELOG.md` once 0.1.0 is cut.

## v0.1.0 release blockers

Tag when all items are done.

- [ ] Windows MSI + CUDA backend smoke pass (Phase M shipped to `phase-m-cuda-backend`; macOS DMG verified, Windows pending)
- [ ] LGPL ffmpeg for redistribution (current `imageio-ffmpeg` payload is GPL — fine for personal use, blocks MIT redistribution)
- [ ] Sign + notarize moved into the CI release workflow (currently local via `scripts/sign_and_notarize.sh`)
- [ ] Clean TypeScript + ESLint baseline (~190 strict-TS, ~40 lint errors) → flip CI to blocking
- [ ] README screenshots + 30 s demo GIF

## Next

### Speaker auto-mapping

Auto-generate `speaker_map.json` for new episodes via voice embeddings. Show-level registry with cross-episode identity, cosine similarity over Resemblyzer / pyannote embeddings, bootstrapped from a handful of labeled episodes.

### Onboarding follow-ups

The wizard + home empty-state ship. Remaining:
- Hardware auto-detection (GPU / CPU / VRAM) to pick the default Whisper preset
- ShowPage empty-state CTA into the pipeline preset flow
- Replay wizard from Settings for users who dismissed it

### Developer experience

- Auto-generate TS types from Pydantic to prevent API drift
- Service registry for LLM providers (typed objects, no growing conditionals)
- Transcription engine abstraction — decouple `core/transcribe.py` from WhisperX (Voxtral, Whisper.cpp, cloud ASR)

## Parallel tracks

- Discord bot — conversation context (thread-based or per-user stateful)
- PoToken for YouTube — Rust sidecar to bypass bot detection without cookies
- Export / import — `.podcodex` bundle (SQLite + versions + LanceDB tables); import merges or re-embeds on model mismatch

## Future

### Timeline editor

Multi-track assembly with jingle / music reinsertion: drag-to-reorder segments, insert intros / outros / music beds, per-segment volume + fade, single-file audio export.
