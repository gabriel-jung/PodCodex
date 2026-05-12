# CLAUDE.md

Stack pointers in `README.md`. System wiring in `ARCHITECTURE.md`. Human contributor workflow in `CONTRIBUTING.md`. Build/deploy in `deploy/*.md`. Frontend design rules in `DESIGN.md` (read before writing UI). ML stack runtime quirks (torch/transformers/qwen-tts version pins, HF cache layout, mask path workarounds) in `ML_RUNTIME.md`. Makefile is dev entry, run `make help`.

## Terminology

LLM-correction pipeline step is **correct** (or "AI correct"). Never "polish"; that name was renamed project-wide. Exceptions: historical migration code in `pipeline_db.py`, language code `pl: Polish`, generic English ("onboarding polish").

## Python env

- Pinned 3.12. Install: `uv sync --extra desktop --extra pipeline --extra rag --extra youtube --extra mcp`
- Don't use `.venv/bin/pip`. Use `uv pip install -e . --python .venv/bin/python` if needed.
- Tests: `.venv/bin/python -m pytest`. No root `conftest.py`; fixtures explicit-import from `tests/fixtures/`.
- GPU extras: `--extra gpu` (cu128, Turing+) or `--extra gpu-pascal` (cu126, GTX 10xx/P40/P100). Mutually exclusive; never enable both. Wheel routing in `[tool.uv.sources]` of `pyproject.toml`.

## GPU extras lifecycle

`gpu-pascal` is a transient escape hatch for Pascal users (sm_60-62) and is pinned to `torch>=2.8,<2.10`. PyTorch 2.10 is expected to drop sm_61 entirely. When the project's torch baseline crosses 2.10:

1. Delete the `gpu-pascal` entry from `[project.optional-dependencies]` in `pyproject.toml`.
2. Delete both `gpu-pascal` lines from `[tool.uv.sources]` torch + torchaudio routing.
3. Delete the `pytorch-cu126` `[[tool.uv.index]]` entry.
4. Delete `deploy/PASCAL.md`.
5. Drop the Pascal row from the README "Hardware support" table; remove the Pascal note below it.
6. Drop the Pascal-specific fallback path from the bootstrap kernel guard's warning text in `src/podcodex/bootstrap.py:_check_cuda_kernels_or_degrade` (currently points to `deploy/PASCAL.md`).

## Versioning (bump on every Windows MSI release)

WiX skips file replace on same version → silent broken upgrade. Keep three files in sync:

1. `pyproject.toml`: `version = "X.Y.Z"`
2. `src-tauri/Cargo.toml`: `version = "X.Y.Z"`
3. Refresh `src-tauri/Cargo.lock` (`podcodex-app` entry).

Don't add `version` back to `tauri.conf.json` or hardcode `__version__` in `src/podcodex/__init__.py`; both derive from above. `importlib.metadata.version("podcodex")` works in the PyInstaller bundle only because `"podcodex"` is in `COPY_METADATA` in `packaging/build_server.py`. Don't remove.

## Release tags (controls README download links)

`release.yml` triggers on `push: tags: v*`. Tag name decides flow:

| Tag | Result |
|-----|--------|
| `vX.Y.Z` | Stable. Becomes "latest". README's `/releases/latest/download/PodCodex-{macos-arm64.dmg,windows-x64.msi}` links resolve here. |
| `vX.Y.Z-beta.N`, `vX.Y.Z-rc.N` (any hyphen suffix) | Prerelease. Skipped by "latest". README links untouched. Direct asset URL still works. |
| Actions UI → "Run workflow" | Draft. Hidden until manually published. |

Workflow detects prerelease via `contains(github.ref_name, '-')`. Tag from main after merge for stable; tag from branch with `-beta`/`-rc` suffix for safe branch builds. Never tag stable from branch; main's README link will start serving unmerged code.

README download links use stable aliases (`PodCodex-macos-arm64.dmg`, `PodCodex-windows-x64.msi`) uploaded by post-`tauri-action` `gh release upload ... #<alias>` steps. Don't rename, README breaks.

## Footguns

- **Bootstrap order:** `PODCODEX_DATA_DIR`, `HF_HOME`, `TORCH_HOME` must be set before `bootstrap_for_*()`. Touching `torch.*` before bootstrap triggers a `function 'abs' already has a docstring` race.
- **`HF_TOKEN` required** for `pyannote/speaker-diarization-community-1`. Missing token: transcription hangs silently at the diarization step.
- **PyInstaller config single source:** `packaging/build_server.py`. ~100 hidden imports + COPY_METADATA hardcoded. CPU builds swap torch to CPU wheel (-1.5 GB); GPU builds install `cu128` JIT and skip the swap.
- **Frontend TS/eslint non-blocking in CI**. Strict TS is ON and clean (0 errors); ESLint has ~38 warnings, mostly `react-refresh/only-export-components`.
- **Type sync:** after editing any Pydantic model in `src/podcodex/api/`, run `make types` to regen `frontend/src/api/types.ts`. The file is checked in; never hand-edit.
- **Icon source of truth:** `assets/icon.png` (1024x1024 RGBA). `frontend/public/icon.png` and `src-tauri/icons/*` (desktop sizes + `.icns`/`.ico` only) are derived; `make icons` regenerates them and strips the iOS/Android trees the Tauri CLI emits unconditionally. Don't hand-edit the derived copies; they get blown away on next regen.
- **Episode metadata flow:** `.feed_cache.json` (per show, all known episodes) → `.episode_meta.json` (per episode, indexer reads this) → chunk meta + scalar `pub_date` column in LanceDB. YouTube flat-extraction writes sparse meta files; per-video subtitle import enriches. Use `fill_empty_fields()` (`ingest/rss.py`) for any merge; three sites previously each rolled their own and drifted on which keys count. Sparse `.episode_meta.json` silently breaks RAG date filters.
- **Device / dtype facility:** route all GPU detection through `src/podcodex/core/device.py`. Never call `torch.cuda.is_available()` directly, never hardcode `compute_type="float16"` or `dtype=torch.bfloat16`. `device.resolve_device()`, `device.cuda_available()`, `device.torch_dtype()`, `device.device_str()` are the canonical entry points. Pascal GPUs (sm_60-62) need `int8_float32` + `float32`; bfloat16 needs sm_80+. The bootstrap kernel guard sets `PODCODEX_DEVICE=cpu` if the wheel doesn't ship kernels for the local GPU; downstream code only needs to honor that env.
- **`PODCODEX_DEVICE=auto|cpu|cuda`** env var. `cpu` skips GPU init; `cuda` raises if no CUDA. `make dev-no-tauri-cpu` exports `cpu` for ad-hoc CPU testing on a GPU box.
- **Frontend single facilities** (read `DESIGN.md` first): `frontend/src/lib/stageClasses.ts` for the stage palette; `frontend/src/lib/stepStatus.ts` for review-state derivation (`reviewStatus` / `plainStatus` / `translationsStatus`, plus the `PanelStatus` type used by `PipelinePanel`); `frontend/src/lib/showCounts.ts` for episode count labels. Don't reroll inline; extend the facility.
- **Speaker labels can be empty string.** YouTube subtitle imports without `<v Speaker>` tags produce segments with `speaker=""`. Subtitle parsers in `core/_utils.py` (`srt_to_segments`, `vtt_to_segments`) default empty to `NARRATOR_SPEAKER` ("Narrator"), but legacy transcripts on disk still carry `""`. State guards keyed by speaker name must compare against `null`/`undefined` explicitly (e.g. `editingFor === null`), never `if (!editingFor)`; empty string is falsy and silently no-ops the rename/edit flow. `SpeakerStrip` treats `""` and `SPEAKER_\d+` as "unnamed" chips so users can rename them.
- **Per-step symmetry (versioning).** Every pipeline step (`transcript`, `corrected`, language codes, `synthesize`, `speaker_map`, parquet substeps) writes through `save_version` (or `save_synthesize_version` for `.wav`) at `version_path(base, step, id)` and is removed through the generic `delete_version`. `_refresh_status_after_delete` is the sole cleanup hook: it demotes the `pipeline_db` flag (`transcribed`/`corrected`/`synthesized`) and prunes the `translations` list when the last version of a step is deleted. Flags must demote, not only promote; `shows.py` `unified_episodes` also runs a reconcile pass so a missing-versions episode never shows `synthesized=True`. When adding a new step, mirror all four touchpoints (path, save, delete, status refresh). Don't fork.
- **Synthesize step quirks.** Step output is `.wav`, not seglist JSON. `save_synthesize_version` is a thin variant of `save_version` that stat-hashes the audio (`content_hash="size:<bytes>"`) instead of hashing segments, but storage path (`{ep_dir}/synthesize/{id}.wav`) and delete path (`delete_version`) match all other steps. No `params.filename`; the path is fully determined by `version_path`.
- **Episode panel state.** Mount panels with `<Panel key={"${step}|${episode.id}"}>` (see `frontend/src/pages/EpisodePage.tsx`) so per-episode local state resets on episode switch. State that must survive a panel unmount during an active background job (e.g. source-segment picker selection while assemble runs) belongs in a parent-owned `useRef`, not in the panel itself.
- **Source-aware synth queries.** Frontend synth queries (`voice-samples`, `generated-segments`, `versions`, `status`) key on the value returned by `getEpisodeSourceRef` (`frontend/src/lib/episodeRef.ts`): `audio_path` when present, else `output_dir`. Never key on `audio_path` alone; YouTube subtitle-only episodes have no source audio and their queries silently miss the cache invalidation pass.
- **HF cache vars + transformers mask patches.** Three HF env vars must all align so the snapshot store and transformers loader use the same directory. `HF_HOME=<data_dir>/models/huggingface/`, `HF_HUB_CACHE=<HF_HOME>/hub/`, `TRANSFORMERS_CACHE=<HF_HOME>/hub/`. Setters live in three places (Tauri shell, `_wire_ml_caches`, `get_hf_cache_dir`); if any of them diverges from this layout, the loaders split-brain. Two transformers vmap bugs are worked around in `bootstrap.py:_install_transformers_torch_check_patch` (Pplx path) and `core/synthesize.py:_patch_sdpa_mask_for_mimi_vmap_bug` (MiMi codec in qwen-tts). Full details, symptoms, and upgrade-path notes in `ML_RUNTIME.md`. Read it before touching cache setters or model loaders.
