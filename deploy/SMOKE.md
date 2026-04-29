# Cross-platform smoke checklist

Pre-release manual pass. Run once on each target before cutting a build.
Record outcomes in the PR description, linking to logs where relevant.

## Matrix

| OS            | Shell | Tauri | Native window | MCP toggle |
|---------------|-------|-------|---------------|------------|
| macOS 14+     | zsh   | ✓     | ✓             | stdio path |
| Ubuntu 22/24  | bash  | ✓     | ✓             | stdio path |
| Windows 11    | PS    | ✓     | ✓             | stdio path |
| WSL2 (Ubuntu) | bash  | ✓     | n/a (browser) | `wsl.exe`  |

## Before you start

- Fresh user account or wiped `~/.config/podcodex/`, `~/.local/share/podcodex/` (Linux), `~/Library/Application Support/podcodex/` (macOS), and `%APPDATA%\podcodex\` (Windows) — onboarding has to behave like a true first launch.
- Ensure `ffmpeg` on PATH. Ensure Rust toolchain for Tauri native window.
- Record: OS version, Node version, Python version, GPU (if any), `make dev` vs `make build`.

## Smoke run

Start with `make dev` on the host platform (or `make dev-no-tauri` for WSL / browser-only).

### 1. Startup

- [ ] Backend boots without import errors; `/api/health` returns `ok`.
- [ ] `~/.config/podcodex/` is created on first launch (platform-native equivalent on Windows/macOS is still this path for now).
- [ ] Startup log shows "Startup recovery: reaped N stale temp file(s)" only if orphans existed (absent on clean state is expected).

### 2. Onboarding

- [ ] Empty home page shows the 3-step wizard.
- [ ] "Skip" dismisses; `seen=true` persists.
- [ ] EmptyState CTA remains after skip.
- [ ] Wizard does not reappear after a restart.

### 3. Add show

- [ ] Apple Podcasts search returns results for a generic query ("this american life").
- [ ] RSS URL add → folder created → first episode row appears.
- [ ] YouTube channel URL add → folder created → episodes populated (sub-only import works without deno if subs are manual).
- [ ] Local folder import lists existing audio.

### 4. Pipeline

- [ ] Transcribe one short episode on CPU preset with diarize off — no HF token needed, no warning shown.
- [ ] Toggle Speaker identification → banner "HuggingFace token needed — set it up in Credentials" appears.
- [ ] Click the link → Settings opens on Credentials tab, HF_TOKEN card scrolled into view.
- [ ] Save a valid token → badge flips to "Stored · hf_…" and Transcribe dialog banner disappears.
- [ ] Batch transcribe → task bar shows per-episode progress, cancellation works.
- [ ] Correct / Translate flows open and can enqueue work (even if the API key is absent, manual mode should still render).

### 5. Editor

- [ ] Virtualized transcript scrolls smoothly at 10k+ segments.
- [ ] "Now playing" button re-centers on the active segment; auto-follow does NOT hijack manual scroll.
- [ ] Save a segment → version appears under History dropdown with provenance populated.

### 6. Search

- [ ] Index one episode → Semantic, Exact, Random all return results on the indexed show.
- [ ] Cmd/Ctrl+K opens the command palette; transcript search returns episode matches.
- [ ] Empty index message is shown for shows that haven't been indexed yet (no blank states).

### 7. Integrations

- [ ] Settings → Integrations → Claude Desktop toggle: writes `claude_desktop_config.json`, with the correct absolute `podcodex-mcp` path for the current venv.
- [ ] On WSL hosts, the `wsl.exe` wrapper form is emitted.

### 8. Reindex CLI

- [ ] `podcodex-reindex <show> --list` prints current collections.
- [ ] `podcodex-reindex <show> --dry-run` reports what would happen without touching LanceDB.
- [ ] Real reindex completes and downstream search returns results.

### 9. Recovery

- [ ] Drop a `.tmp_test` file (older than 30 min) into a show folder → backend restart reports "reaped 1 stale temp file".
- [ ] Drop a fresh `.tmp_test` file → it survives (race safety).

### 10. Packaging smoke (once Phase P begins)

- [ ] `make build` produces a native bundle for the current OS.
- [ ] Installed bundle launches without a dev checkout present.
- [ ] Config and secrets still resolve to `~/.config/podcodex/`.
- [ ] Claude Desktop integration picks the installed binary, not the dev venv.
