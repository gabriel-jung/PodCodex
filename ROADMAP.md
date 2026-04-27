# Roadmap

Forward-looking plan only. For the list of shipped features, see the README's *What it does* and *Features* sections; for per-change history, see [CHANGELOG.md](CHANGELOG.md).

## v0.1.0 release blockers

Everything outside this list is post-1.0. Tag when all items are ✅.

- [ ] Clean TypeScript + ESLint baseline (~190 strict-TS, ~40 lint errors) → flip CI to blocking
- [ ] README screenshots + 30 s demo GIF
- [ ] Manual smoke pass on Windows (see [deploy/SMOKE.md](deploy/SMOKE.md))
- [ ] Distribution bundles — see [Phase M](#phase-m--standalone-distribution-v010). macOS `.dmg` done; Windows `.msi` + optional CUDA backend pending.
- [ ] Sign + notarize CI step (mac done locally via `scripts/sign_and_notarize.sh`; needs to move to GitHub Actions with secrets)
- [ ] LGPL ffmpeg build (current bundled ffmpeg is GPL — fine for personal use, blocks redistribution under MIT)

## Phases

### Phase L — Speaker auto-mapping

Auto-generate `speaker_map.json` for new episodes via voice embeddings.

- Show-level speaker registry (name, language, avatar, voice samples, synthesis config)
- Cross-episode identity: same speaker recognized across a show
- Voice embeddings via Resemblyzer or pyannote SpeakerEmbedding
- Reference DB built from manually-labeled episodes; cosine similarity + confidence threshold
- Bootstrapping: a handful of labeled episodes seed the registry for fixed-cast podcasts

### Phase M — Standalone distribution (v0.1.0)

VoiceBox-style: small CPU sidecar in the installer (`.dmg` for macOS, `.msi` for Windows) plus an optional CUDA backend downloaded at runtime (~2.4 GB total, ~300 MB server-core + ~2 GB cuda-libs) when an NVIDIA GPU is detected. Linux is build-from-source via `make dev`, no shipped artifact — the matrix of GPU vendors, distros, and glibc versions makes a single redistributable `.deb` impractical.

- M.1 — Drop Linux/AppImage/rpm bundle targets; commit OpenSSL spec fix (done in working tree)
- M.2 — Two-binary build in `packaging/build_server.py`: CPU `--onefile` (default) + GPU `--onedir` (`--gpu` flag). Auto-swap to `torch+cpu` wheel for the CPU bundle on hosts where the venv has GPU torch, restore after
- M.3 — `packaging/package_gpu.py`: split GPU `--onedir` into server-core (~300 MB, app-versioned) + cuda-libs (~2 GB, torch-major-pinned) + `cuda-libs.json` manifest with sha256
- M.4 — Backend service: NVIDIA detection, `GET /api/gpu/status`, `POST /api/gpu/download`, `POST /api/gpu/activate`; integrate with existing TaskInfo system
- M.5 — Tauri Rust: spawn GPU sidecar from app data dir when activated, else bundled CPU sidecar; Tauri updater config + GitHub releases endpoint
- M.6 — Frontend Settings page (CPU-only / CUDA backend status, download UX, progress)
- M.7 — CI: macos-latest builds DMG; windows-latest builds CPU MSI + GPU archives, uploads to release

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
