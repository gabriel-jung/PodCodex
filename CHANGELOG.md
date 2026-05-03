# Changelog

All notable changes to PodCodex are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versioning
follows [Semantic Versioning](https://semver.org/). Dates are ISO 8601.

## [Unreleased]

### Removed
- Discord bot `/ask` and `/ask-advanced` commands, plus the `synthesis` module and all `ask_*` config fields (CLI, `BotConfig`, `ServerSettings`, `/setup` params). LLM-synthesized answers are not yet ready for release — surface to be revisited after prompt-isolation and grounding guarantees are in place. `openai` dependency dropped from the `bot` extra. Pre-existing `ask_*` entries in `server_config.json` are ignored at load time (no migration needed).

### Added
- `src/podcodex/core/device.py` — single facility for device + dtype + compute-type resolution. Replaces seven scattered `torch.cuda.is_available()` callsites and the inline `compute_type="float16"` / `dtype=torch.bfloat16` hardcodes. Picks per compute capability: `float16` + `bfloat16` for Ampere+, `float16` + `float16` for Volta/Turing, `int8_float32` + `float32` for Pascal (GTX 10xx, P40, P100), `int8` + `float32` for CPU.
- `PODCODEX_DEVICE=auto|cpu|cuda` env var. `cpu` skips GPU init even when CUDA is available — useful for `make dev-no-tauri-cpu`, ad-hoc CPU testing, or as an escape hatch when the installed torch wheel doesn't match the GPU. `cuda` raises if no CUDA device is present.
- `make dev-no-tauri-cpu` target — `dev-no-tauri` with `PODCODEX_DEVICE=cpu` exported.
- Bootstrap-time CUDA kernel guard (`bootstrap._check_cuda_kernels_or_degrade`). On a wheel/GPU mismatch (e.g. cu128 wheel on a Pascal box), sets `PODCODEX_DEVICE=cpu` and logs a clear warning instead of letting `CUDA error: no kernel image is available` surface from the first transcription call.
- `GET /api/system/device` endpoint — returns resolved device, compute_type, dtype, GPU name, compute capability, arch list, and active env override. Diagnostic surface for the frontend GPU panel.
- `gpu` and `gpu-pascal` uv extras with explicit `[tool.uv.sources]` routing — `--extra gpu` pulls torch + torchaudio from PyTorch's cu128 index (Turing+), `--extra gpu-pascal` pulls cu126 (GTX 10xx, Titan Xp, P40, P100). Mutually exclusive.
- `deploy/PASCAL.md` — install path, verification, and CPU fallback for Pascal users. Linked from README and from the bootstrap kernel-guard warning.
- README "Hardware support" table — explicit GPU coverage matrix with install paths for Ampere+, Turing, Pascal, Apple Silicon, and CPU.
- macOS standalone `.app` / `.dmg` build via PyInstaller-frozen FastAPI sidecar (`make bundle`). Single-file `podcodex-server` onefile binary (~420 MB), bundled `ffmpeg` + `yt-dlp` static binaries, ML caches relocated under `~/Library/Application Support/com.podcodex.desktop/models/`. Cold start 10-30 s while PyInstaller extracts; warm <1 s. Phased boot UI in `RootLayout` keeps the user informed during first launch.
- Sign + notarize wrapper at `scripts/sign_and_notarize.sh` (Developer ID + notarytool keychain profile). Resigns nested PyInstaller `.so` / `.dylib` defensively before stapling.
- Tauri shell (`src-tauri/src/lib.rs`) auto-spawns the bundled sidecar, polls `/api/health` before revealing the window, kills the child on `RunEvent::Exit`. `PODCODEX_SKIP_BACKEND_SPAWN=1` keeps `make dev` working with an external uvicorn.
- Two-tier stale UX in `ProgressBar` (`SLOW=30s` indeterminate sweep, `STUCK=600s` warning + Retry) so opaque ML steps and hung batch items get distinct treatment.
- User-scoped secrets store at `~/.config/podcodex/secrets.env` (mode 0600). Managed from **Settings → Credentials** for `HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `DISCORD_TOKEN`. Backend loads it after repo `.env` (overrides dev values) and reloads live on save.
- `GET` / `PUT /api/config/secrets` routes with atomic writes and per-value source reporting (`file` / `env` / `none`).
- Settings page routing: `?tab=<name>` picks the initial panel; `#<anchor>` scrolls to it.
- `core/app_paths.py` — canonical resolver for the user config directory.
- `podcodex-reindex` CLI — drops and rebuilds a show's LanceDB collections from its transcripts. Supports `--model`, `--chunker`, `--all-models`, `--list`, and `--dry-run`. LanceDB is treated as derived state; reindex is the recovery path after corruption or model change.
- Startup recovery reaper (`core/recovery.py`) — removes orphaned `.tmp_*` / `*.tmp` files older than 30 minutes from show folders, config dir, and the LanceDB index dir. Runs automatically on backend startup. Atomic writes normally clean their own tempfiles; the reaper handles the SIGKILL / OOM / power-loss case.
- CI workflow (`.github/workflows/ci.yml`) — pytest (backend) and `cargo check` (Tauri) block on failure; frontend `tsc -b` and `npm run lint` run non-blocking pending a baseline cleanup.

### Changed
- **Speaker identification** in the batch Transcribe dialog now renders as a proper toggle with description, sibling to "Clean transcript" under a shared "Options" section header. When enabled without a token, a one-line callout links to **Settings → Credentials**.
- Default `diarize: false` (new installs). Existing users' stored preference is preserved.
- Pipeline defaults panel no longer embeds an inline HF token input; Credentials tab is the single source of truth.

### Documented
- `deploy/SMOKE.md` — cross-platform smoke checklist (macOS / Ubuntu / Windows / WSL2), covering startup, onboarding, pipeline, editor, search, integrations, reindex CLI, and recovery paths.

### Fixed
- `transcribe.py` no longer hardcodes `compute_type="float16"`. CTranslate2 rejects FP16 on Pascal GPUs (`Requested float16 compute type, but the target device or backend does not support efficient float16 computation`); GTX 10xx users now get `int8_float32` automatically.
- `synthesize.py` no longer hardcodes `dtype=torch.bfloat16` for Qwen3-TTS. bfloat16 requires sm_80 (Ampere); Pascal users would have hit a kernel error. Dtype is now picked per compute capability via `device.torch_dtype()`.
- Three flaky unit tests realigned with current behavior:
  - `test_translate.py::test_build_manual_prompt_asks_for_positional_text_output` now checks for position-based output contract rather than the removed `"index"` key.
  - `test_utils.py::test_call_and_parse_count_mismatch_rejects_whole_batch` validates that a short LLM reply triggers batch rejection (keeping all originals) rather than partial application.
  - `test_utils.py::test_call_and_parse_skips_break_segments` asserts the current user-message wording (`"all 2 segments"`).

## [0.1.0] — unreleased

First public preview. Scope gate: distribution bundles (macOS / Linux / Windows),
onboarding wizard, and cross-platform smoke tests precede tagging. Up-to-date
feature scope lives in [ROADMAP.md](ROADMAP.md) under *Completed*.
