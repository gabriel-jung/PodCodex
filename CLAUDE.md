# CLAUDE.md

## Bumping the app version

Version is sourced from **two** files (Tauri reads Cargo.toml's `package.version` when the field is omitted from `tauri.conf.json`; Python reads `pyproject.toml` via `importlib.metadata`):

1. `pyproject.toml` — `version = "X.Y.Z"`
2. `src-tauri/Cargo.toml` — `version = "X.Y.Z"`

Then refresh `src-tauri/Cargo.lock` (the `podcodex-app` entry's `version`) so CI doesn't trip.

Do not add a `version` field back to `src-tauri/tauri.conf.json` or a literal `__version__` in `src/podcodex/__init__.py` — both derive from the two files above.

Bump on every Windows MSI release: WiX skips file replacement when the product version doesn't change, which silently breaks upgrades.

`importlib.metadata.version("podcodex")` only works in the PyInstaller bundle because `"podcodex"` is in `COPY_METADATA` in `packaging/build_server.py`. Don't remove it.
