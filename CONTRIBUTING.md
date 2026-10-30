# Contributing

Quick orientation for human contributors. AI assistant context lives in [CLAUDE.md](CLAUDE.md); design rules in [DESIGN.md](DESIGN.md); system wiring in [ARCHITECTURE.md](ARCHITECTURE.md); build / packaging in [deploy/BUILD.md](deploy/BUILD.md).

## Setup

You'll need:

- **[uv](https://docs.astral.sh/uv/)**: installs Python 3.12 and the dependencies.
- **[Node.js](https://nodejs.org/) LTS**: frontend.
- **[Rust](https://www.rust-lang.org/)** *(optional)*: only for the native Tauri window.
- **[FFmpeg](https://ffmpeg.org/download.html)** on PATH, as for the installed app.
- On Linux, the Tauri system libraries listed in the `Makefile` header (apt or dnf).

```bash
git clone https://github.com/gabriel-jung/PodCodex && cd PodCodex
make setup                # uv sync + npm install
make dev                  # FastAPI + Vite + Tauri, hot-reload (needs Rust)
make dev-no-tauri         # same, in the browser, no Rust needed
```

`make help` lists all targets.

## Workflows

- **Run a single test:** `.venv/bin/python -m pytest tests/test_foo.py::test_bar -xvs`
- **Run all Python tests:** `make test`
- **Regenerate frontend types / icons:** see [deploy/BUILD.md § Regenerated artifacts](deploy/BUILD.md#regenerated-artifacts). `make types` after editing Pydantic models in `src/podcodex/api/`; `make icons` after replacing `assets/icon.png` (1024x1024 canonical).
- **Run frontend tests:** `cd frontend && npm test` (vitest; `npm run test:watch` while working). Config in `frontend/vitest.config.ts`, kept separate from `vite.config.ts` because Vitest bundles its own Vite and merging the two breaks `tsc -b`.
- **Lint frontend:** `cd frontend && npm run lint` (runs with `--max-warnings 0`; fails CI on any warning)
- **Type-check frontend:** `cd frontend && npx tsc -b` (strict mode is on and clean; fails CI on any error)
- **Dead-code check frontend:** `cd frontend && npm run knip` (config in `frontend/knip.json`; fails CI on unused files, exports or dependencies)

## Conventions

- **Commits:** conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`); imperative mood; concise; no trial-and-error context.
- **Versioning:** run `make bump VERSION=X.Y.Z` on every Windows MSI release (WiX skips same-version replace); it keeps both manifests and both lockfiles in sync. See [CLAUDE.md § Versioning](CLAUDE.md#versioning-bump-on-every-windows-msi-release).
- **Terminology:** the LLM-correction step is **correct** (or "AI correct"), never "polish". See [CLAUDE.md § Terminology](CLAUDE.md#terminology).

## Pull requests

- Run `make test` (Python) and, for frontend changes, `npx tsc -b`, `npm run lint`, `npm test` and `npm run knip` in `frontend/` (all four block CI) before opening
- Reference the roadmap item (README) or issue in the PR description when applicable
- Frontend changes: include a screenshot or short clip when behavior is visual
