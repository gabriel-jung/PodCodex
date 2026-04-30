# CLAUDE.md

Stack pointers in `README.md`. System wiring in `ARCHITECTURE.md`. Human contributor workflow in `CONTRIBUTING.md`. Build/deploy in `deploy/*.md`. Frontend design rules in `DESIGN.md` (read before writing UI). Makefile is dev entry — read `make help`.

## Terminology

LLM-correction pipeline step is **correct** (or "AI correct"). Never "polish" — that name was renamed project-wide. Exceptions: historical migration code in `pipeline_db.py`, language code `pl: Polish`, generic English ("onboarding polish").

## Python env

- Pinned 3.12. Install: `uv sync --extra desktop --extra pipeline --extra rag --extra youtube --extra mcp`
- Don't use `.venv/bin/pip`. Use `uv pip install -e . --python .venv/bin/python` if needed.
- Tests: `.venv/bin/python -m pytest`. No root `conftest.py`; fixtures explicit-import from `tests/fixtures/`.

## Versioning (bump on every Windows MSI release)

WiX skips file replace on same version → silent broken upgrade. Keep two files in sync:

1. `pyproject.toml` — `version = "X.Y.Z"`
2. `src-tauri/Cargo.toml` — `version = "X.Y.Z"`
3. Refresh `src-tauri/Cargo.lock` (`podcodex-app` entry).

Don't add `version` back to `tauri.conf.json` or hardcode `__version__` in `src/podcodex/__init__.py` — both derive from above. `importlib.metadata.version("podcodex")` works in the PyInstaller bundle only because `"podcodex"` is in `COPY_METADATA` in `packaging/build_server.py`. Don't remove.

## Footguns

- **Bootstrap order:** `PODCODEX_DATA_DIR`, `HF_HOME`, `TORCH_HOME` must be set before `bootstrap_for_*()`. Touching `torch.*` before bootstrap → `function 'abs' already has a docstring` race.
- **`HF_TOKEN` required** for `pyannote/speaker-diarization-community-1`. Missing → transcription hangs silently at the diarization step.
- **PyInstaller config single source:** `packaging/build_server.py`. ~100 hidden imports + COPY_METADATA hardcoded. CPU builds swap torch to CPU wheel (-1.5 GB); GPU builds install `cu128` JIT and skip the swap.
- **Frontend TS/eslint non-blocking in CI** (~190 TS + ~40 eslint baseline). Don't enable strict without a planned cleanup PR.
- **Type sync:** after editing any Pydantic model in `src/podcodex/api/`, run `make types` to regen `frontend/src/api/types.ts`. The file is checked in; never hand-edit.
