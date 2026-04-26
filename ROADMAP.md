# Roadmap

Forward-looking plan only. For the list of shipped features, see the README's *What it does* and *Features* sections; for per-change history, see [CHANGELOG.md](CHANGELOG.md).

## v0.1.0 release blockers

Everything outside this list is post-1.0. Tag when all items are ✅.

- [ ] Clean TypeScript + ESLint baseline (~190 strict-TS, ~40 lint errors) → flip CI to blocking
- [ ] README screenshots + 30 s demo GIF
- [ ] Manual smoke pass on Linux native and Windows (see [deploy/SMOKE.md](deploy/SMOKE.md))
- [ ] Distribution bundles — PyInstaller sidecar + `make build` producing `.app` / `.deb` / `.exe`. WhisperX + Torch bundling is the hard part.
  - [x] macOS `.app` + `.dmg` (arm64) — `make bundle` ships ~497 MB .app / ~459 MB DMG. End-to-end transcription verified on macOS 26.4.1.
  - [ ] Linux `.deb` + AppImage
  - [ ] Windows `.exe` (NSIS / MSI)
  - [ ] Sign + notarize CI step (mac done locally via `scripts/sign_and_notarize.sh`; needs to move to GitHub Actions with secrets)
  - [ ] LGPL ffmpeg build (current bundled ffmpeg is GPL — fine for personal use, blocks redistribution under MIT)

## Next (post-1.0)

### Phase M — Speaker auto-mapping

Auto-generate `speaker_map.json` for new episodes via voice embeddings.

- Show-level speaker registry (name, language, avatar, voice samples, synthesis config)
- Cross-episode identity: same speaker recognized across a show
- Voice embeddings via Resemblyzer or pyannote SpeakerEmbedding
- Reference DB built from manually-labeled episodes; cosine similarity + confidence threshold
- Bootstrapping: a handful of labeled episodes seed the registry for fixed-cast podcasts

### Phase N — Onboarding polish

The wizard + home empty-state already ship. Remaining polish:

- Hardware auto-detection (GPU / CPU / VRAM) to pick the default Whisper preset
- ShowPage empty-state CTA that guides into the pipeline preset flow
- Replay wizard from Settings for users who dismissed it

### Phase O — Developer experience

- Auto-generate TS types from Pydantic to prevent API drift
- Service registry for LLM providers (typed objects, no growing conditionals)
- Transcription engine abstraction — decouple `core/transcribe.py` from WhisperX to allow Voxtral, Whisper.cpp, or cloud ASR

## Parallel tracks

Can happen alongside any phase, usually driven by specific requests.

- Discord bot — conversation context (thread-based or per-user stateful)
- PoToken for YouTube — Rust sidecar to bypass bot detection without cookies
- Export / import — `podcodex export {show} → {show}.podcodex` bundling SQLite + versions + LanceDB tables; import merges or re-embeds on model mismatch

## Future

### Timeline editor

Multi-track assembly with jingle / music reinsertion for final episode production: drag-to-reorder segments, insert intros / outros / music beds, per-segment volume + fade, export a single audio file.
