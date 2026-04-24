# Changelog

All notable changes to PodCodex are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versioning
follows [Semantic Versioning](https://semver.org/). Dates are ISO 8601.

## [Unreleased]

### Removed
- Discord bot `/ask` and `/ask-advanced` commands, plus the `synthesis` module and all `ask_*` config fields (CLI, `BotConfig`, `ServerSettings`, `/setup` params). LLM-synthesized answers are not yet ready for release — surface to be revisited after prompt-isolation and grounding guarantees are in place. `openai` dependency dropped from the `bot` extra. Pre-existing `ask_*` entries in `server_config.json` are ignored at load time (no migration needed).

### Added
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
- Three flaky unit tests realigned with current behavior:
  - `test_translate.py::test_build_manual_prompt_asks_for_positional_text_output` now checks for position-based output contract rather than the removed `"index"` key.
  - `test_utils.py::test_call_and_parse_count_mismatch_rejects_whole_batch` validates that a short LLM reply triggers batch rejection (keeping all originals) rather than partial application.
  - `test_utils.py::test_call_and_parse_skips_break_segments` asserts the current user-message wording (`"all 2 segments"`).

## [0.1.0] — unreleased

First public preview. Scope gate: distribution bundles (macOS / Linux / Windows),
onboarding wizard, and cross-platform smoke tests precede tagging. Up-to-date
feature scope lives in [ROADMAP.md](ROADMAP.md) under *Completed*.
