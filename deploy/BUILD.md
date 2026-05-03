# Building standalone bundles

How to produce a distributable PodCodex `.dmg` (macOS) or `.msi`
(Windows) from a clean checkout. Linux is not shipped; see [Linux](#linux-build-from-source)
below.

> **Status:** macOS arm64 verified end-to-end (~500 MB bundle).
> Windows path is documented but not yet smoke-tested. Expect minor gaps;
> report what breaks.

The flow is the same on every shipped target:

1. **Freeze the Python backend** with PyInstaller → single
   `podcodex-server` binary (CPU-only).
2. **Fetch the `yt-dlp` static binary** for the host triple. ffmpeg is **not** bundled — the app shells out to the user's system install (see [LICENSE_AUDIT.md](../LICENSE_AUDIT.md)).
3. **Build the frontend** (`npm run build`).
4. **Bundle with Tauri** (`cargo tauri build`) → native installer.

`make bundle` chains all four. Use the individual `make bundle-*` targets
when iterating on a single layer.

> **GPU acceleration:** the bundled installer is CPU-only. NVIDIA users
> activate an in-app CUDA backend (~2.4 GB download) from the GPU panel
> in settings. Pascal cards (sm_60–62) need the source-build path
> (`--extra gpu-pascal`); see [`PASCAL.md`](PASCAL.md).

---

## macOS (arm64 / x86_64)

```bash
brew install node rust uv ffmpeg
cargo install tauri-cli --version "^2"

git clone https://github.com/gabriel-jung/podcodex && cd podcodex
make setup                # uv sync + npm install
make setup-pyinstaller    # uv pip install pyinstaller into .venv
make bundle               # ~5 min total, produces .app + .dmg
```

### Sign + notarize (optional, distribution only)

For maintainers distributing their own bundle. All values below are
placeholders; real credentials never live in this file.

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
xcrun notarytool store-credentials podcodex-notary \
    --apple-id you@example.com --team-id TEAMID --password app-specific-pwd
make bundle-sign          # signs nested .so/.dylib + notarizes + staples
make bundle-sign-only     # sign without notarizing (fast iteration)
```

Without signing, Gatekeeper blocks first launch. Drag the app to
`/Applications`, then run once:

```bash
xattr -dr com.apple.quarantine /Applications/PodCodex.app
```

### Auto-updater signing

The Tauri updater needs an ed25519 keypair. Three steps:

1. Generate the keypair: `cargo tauri signer generate -w ~/.tauri/podcodex.key`.
2. In `src-tauri/tauri.conf.json`: paste the public key into `plugins.updater.pubkey`,
   set `plugins.updater.active = true`, set `bundle.createUpdaterArtifacts = "v1Compatible"`.
3. Add `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` to GitHub repo secrets.

Without these, releases ship without `latest.json` (auto-update disabled, manual install still works).

### macOS gotchas

- WhisperX runs CPU-only on Apple Silicon (no MPS support upstream).
- First launch each session pays a 10-30 s cold start while PyInstaller
  extracts to `/tmp/_MEIxxx/`. Warm launches are <1 s.

---

## Linux (build from source)

PodCodex doesn't ship a Linux installer. Linux users run from source:

```bash
# Install Linux dev prereqs from the Makefile header (apt or dnf).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli --version "^2"
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/gabriel-jung/podcodex && cd podcodex
make setup
make dev                  # FastAPI + Vite + Tauri, hot-reload
```

WSL2 is fine for development (uses WSLg on Windows 11). It cannot
produce the Windows `.msi`; build that on a Windows host directly.

---

## Windows 11 native (.msi)

```powershell
# Install prereqs (winget)
winget install --id=Microsoft.VisualStudio.2022.BuildTools --silent
winget install --id=OpenJS.NodeJS.LTS
winget install --id=Rustlang.Rustup
winget install --id=astral-sh.uv
rustup target add x86_64-pc-windows-msvc
cargo install tauri-cli --version "^2"

git clone https://github.com/gabriel-jung/podcodex
cd podcodex
make setup
make setup-pyinstaller
make bundle               # produces .msi
```

### Windows gotchas

- `make` requires GNU make. Install via `winget install GnuWin32.Make`,
  or run the equivalent commands from `Makefile` directly in PowerShell.
- The MSVC build toolchain is mandatory; MinGW does not work for Tauri.
- Code signing requires an Authenticode certificate (sectigo / digicert).
  Without it, SmartScreen warns on first launch.

---

## GPU extras (source builds)

Bundled installers handle GPU through the in-app activation flow. Source
builds need an explicit extra:

```bash
uv sync --extra gpu          # cu128 wheels: Turing, Ampere, Ada, Blackwell (sm_75+)
uv sync --extra gpu-pascal   # cu126 wheels: Pascal only (sm_60–62), see PASCAL.md
```

Mutually exclusive; never enable both.

`PODCODEX_DEVICE=auto|cpu|cuda` overrides device selection at runtime.
`cpu` skips GPU init even when CUDA is available; `cuda` raises if no
CUDA. Default is `auto`.

---

## Regenerated artifacts

A few files are checked in but auto-generated. Re-run after editing the source:

```bash
make types     # frontend/src/api/types.ts from Pydantic models
make icons     # src-tauri/icons/ + frontend/public/icon.png from assets/icon.png
```

---

## Cleaning + rebuilding

```bash
make clean                # removes frontend/dist, src-tauri/target,
                          # src-tauri/binaries, packaging/dist,
                          # packaging/build_work
make bundle               # full rebuild
```

Iterating on a single layer is faster:

| Changed             | Re-run                                                             |
|---------------------|--------------------------------------------------------------------|
| Frontend (TS / CSS) | `cd frontend && npm run build && cd src-tauri && cargo tauri build` |
| Rust shell only     | `cd src-tauri && cargo tauri build`                                |
| Python backend      | `make bundle-server && cd src-tauri && cargo tauri build`          |
| yt-dlp              | `make bundle-natives`                                              |

---

## Smoke + report

After `make bundle`, smoke-test:

1. Move the dev checkout out of the way (or run on a different account).
2. Launch the bundle from its installed location.
3. Confirm splash phase messages on cold start (10-30 s first launch each session).
4. Step through [`SMOKE.md`](SMOKE.md).

If `/api/health` never returns and the splash stays past ~60 s, run
from a terminal to surface stderr:

- macOS: `PodCodex.app/Contents/MacOS/podcodex-app 2>&1 | head -50`
- Windows: launch from PowerShell with `& "C:\Program Files\PodCodex\PodCodex.exe"`

The bundled sidecar logs to `<data_dir>/logs/server.log`
(`~/Library/Application Support/podcodex/` on macOS,
`%APPDATA%\podcodex\` on Windows,
`~/.local/share/podcodex/` on Linux).

Report build issues at <https://github.com/gabriel-jung/podcodex/issues>. Include:

- OS + version (`uname -a`, `sw_vers`, or `winver`).
- Tool versions: `python --version`, `node --version`, `rustc --version`, `cargo tauri --version`.
- Full `make bundle` output (use `tee build.log`).
- For PyInstaller failures, also attach `packaging/build_work/podcodex-server/warn-podcodex-server.txt`.
