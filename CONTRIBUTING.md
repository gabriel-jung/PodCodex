# Contributing

Quick orientation for human contributors. AI assistant context lives in [CLAUDE.md](CLAUDE.md); design rules in [DESIGN.md](DESIGN.md); system wiring in [ARCHITECTURE.md](ARCHITECTURE.md); ML stack pins and runtime patches in [ML_RUNTIME.md](ML_RUNTIME.md); build / packaging in [deploy/BUILD.md](deploy/BUILD.md).

## Setup

```bash
git clone https://github.com/gabriel-jung/PodCodex && cd PodCodex
make setup                # uv sync + npm install
make dev                  # FastAPI + Vite + Tauri, hot-reload
```

`make help` lists all targets. `make dev-no-tauri` runs without the native window.

## Workflows

- **Run a single test:** `.venv/bin/python -m pytest tests/test_foo.py::test_bar -xvs`
- **Run all Python tests:** `make test`
- **Regenerate frontend types / icons:** see [deploy/BUILD.md § Regenerated artifacts](deploy/BUILD.md#regenerated-artifacts). `make types` after editing Pydantic models in `src/podcodex/api/`; `make icons` after replacing `assets/icon.png` (1024x1024 canonical).
- **Lint frontend:** `cd frontend && npm run lint` (currently non-blocking)
- **Type-check frontend:** `cd frontend && npx tsc -b` (strict mode is on and clean; non-blocking in CI)

## Conventions

- **Commits:** conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`); imperative mood; concise; no trial-and-error context.
- **Versioning:** bump `pyproject.toml` + `src-tauri/Cargo.toml` together on every Windows MSI release (WiX skips same-version replace). See [CLAUDE.md § Versioning](CLAUDE.md#versioning-bump-on-every-windows-msi-release).
- **Terminology:** the LLM-correction step is **correct** (or "AI correct"), never "polish". See [CLAUDE.md § Terminology](CLAUDE.md#terminology).

## Pull requests

- Run `make test` and the relevant smoke items from [`deploy/SMOKE.md`](deploy/SMOKE.md) before opening
- Reference the ROADMAP item or issue in the PR description when applicable
- Frontend changes: include a screenshot or short clip when behavior is visual
