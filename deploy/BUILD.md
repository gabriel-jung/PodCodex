# Building standalone bundles

How to produce a distributable PodCodex `.dmg` (macOS) or `.msi`
(Windows) from a clean checkout. Linux is not shipped — see [Linux](#linux-build-from-source)
below.

> **Status:** macOS arm64 verified end-to-end (~497 MB `.app`, ~459 MB DMG).
> Windows path is documented but not yet smoke-tested. Expect minor gaps;
> report what breaks.

The flow is the same on every shipped target:

1. **Freeze the Python backend** with PyInstaller → single
   `podcodex-server` binary (~420 MB, CPU-only).
2. **Fetch native sidecars** (`ffmpeg`, `yt-dlp`) for the host triple.
3. **Build the frontend** (`npm run build`).
4. **Bundle with Tauri** (`cargo tauri build`) → native installer.

`make bundle` chains all four. Use the individual `make bundle-*` targets
when iterating on a single layer.

> **GPU acceleration:** the shipped installer is CPU-only. On NVIDIA
> hosts the app offers an in-app CUDA backend download (~2.4 GB) —
> see [Phase M](../ROADMAP.md#phase-m--standalone-distribution-v010).
> `make bundle` produces only the CPU sidecar.

---

## macOS (arm64 / x86_64)

```bash
brew install node rust uv ffmpeg
cargo install tauri-cli --version "^2"

git clone https://github.com/gabriel-jung/podcodex && cd podcodex
make setup                # uv sync + npm install
make setup-pyinstaller    # uv pip install pyinstaller into .venv
make bundle               # ~5 min total — produces .app + .dmg
```

Outputs:

- `src-tauri/target/release/bundle/macos/PodCodex.app`
- `src-tauri/target/release/bundle/dmg/PodCodex_<version>_<arch>.dmg`

### Sign + notarize (optional, distribution only)

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
xcrun notarytool store-credentials podcodex-notary \
    --apple-id you@example.com --team-id TEAMID --password app-specific-pwd
make bundle-sign          # signs nested .so/.dylib + notarizes + staples
make bundle-sign-only     # sign without notarizing (fast iteration)
```

Without signing, Gatekeeper blocks first launch — users must
right-click → Open to bypass.

### Auto-update keypair (release infrastructure)

The Tauri updater (configured in `src-tauri/tauri.conf.json` under
`plugins.updater`) needs an ed25519 keypair to sign update artifacts.
The CI release workflow uses `tauri-apps/tauri-action` which automatically
emits `latest.json` (the updater manifest) when the signing secrets are set.

One-time setup:

```bash
# Generate the keypair (writes ~/.tauri/podcodex.key + ~/.tauri/podcodex.key.pub)
cargo tauri signer generate -w ~/.tauri/podcodex.key

# In src-tauri/tauri.conf.json:
#   bundle.createUpdaterArtifacts = "v1Compatible"      (currently absent)
#   plugins.updater.active        = true                 (currently false)
#   plugins.updater.pubkey        = "<paste public key>" (currently REPLACE_ME)

# Add the private key + password to GitHub repo secrets:
#   TAURI_SIGNING_PRIVATE_KEY          (contents of ~/.tauri/podcodex.key)
#   TAURI_SIGNING_PRIVATE_KEY_PASSWORD (password you set during generate)
```

Until the secrets are configured, releases ship without `latest.json` —
the desktop app still installs and works, it just can't auto-update.

### macOS gotchas

- WhisperX runs CPU-only on Apple Silicon (no MPS support upstream).
- ffmpeg bundled is GPL (`libx264` / `libx265`). For redistribution under
  MIT, swap to an LGPL audio-only build by editing
  `scripts/fetch_native_binaries.py` `FFMPEG_SOURCES`.
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
produce the Windows `.msi` — build that on a Windows host directly.

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

Outputs:

- `src-tauri\target\release\bundle\msi\PodCodex_<version>_x64_en-US.msi`

### Windows gotchas

- `make` requires GNU make — install via `winget install GnuWin32.Make`
  or run the equivalent commands from `Makefile` directly in PowerShell.
- The MSVC build toolchain is mandatory; MinGW does not work for Tauri.
- Code signing requires an Authenticode certificate (sectigo / digicert).
  Without it, SmartScreen warns on first launch.

---

## Cleaning + rebuilding

```bash
make clean                # removes frontend/dist, src-tauri/target,
                          # src-tauri/binaries, packaging/dist,
                          # packaging/build_work
make bundle               # full rebuild
```

Iterating on a single layer is faster:

| Changed             | Re-run                                       |
|---------------------|----------------------------------------------|
| Frontend (TS / CSS) | `cd frontend && npm run build && cd src-tauri && cargo tauri build` |
| Rust shell only     | `cd src-tauri && cargo tauri build`           |
| Python backend      | `make bundle-server && cd src-tauri && cargo tauri build` |
| ffmpeg / yt-dlp     | `make bundle-natives`                         |

---

## Verifying a fresh build

After `make bundle`, smoke-test the produced bundle:

1. Move the dev checkout out of the way (or run on a different account).
2. Launch the bundle from its installed location (open the `.app` or
   install the `.msi`).
3. Confirm the splash phase messages appear during cold start (10-30 s
   on first launch each session).
4. Step through the [`deploy/SMOKE.md`](SMOKE.md) checklist.

If `/api/health` never returns and the splash stays on "Almost ready,
hang tight…" past ~60 s, run from a terminal to surface stderr:

- macOS: `PodCodex.app/Contents/MacOS/podcodex-app 2>&1 | head -50`
- Windows: launch from PowerShell — `& "C:\Program Files\PodCodex\PodCodex.exe"`

The bundled sidecar logs to `<data_dir>/logs/server.log`:

| OS      | `<data_dir>` path                                        |
|---------|----------------------------------------------------------|
| macOS   | `~/Library/Application Support/podcodex/`                |
| Windows | `%APPDATA%\podcodex\`                                    |
| Linux   | `~/.local/share/podcodex/`                               |

---

## Reporting build issues

Open an issue at <https://github.com/gabriel-jung/podcodex/issues> with:

- OS + version (`uname -a`, `sw_vers`, `winver`)
- `python --version`, `node --version`, `rustc --version`,
  `cargo tauri --version`
- The full `make bundle` output (use `tee build.log`)
- For PyInstaller failures: `packaging/build_work/podcodex-server/warn-podcodex-server.txt`
