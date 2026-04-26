# Building standalone bundles

How to produce a distributable PodCodex `.app` / `.dmg` / `.deb` /
`.AppImage` / `.exe` from a clean checkout. Pick the section that matches
your host OS — each path is self-contained.

> **Status:** macOS arm64 verified end-to-end (~497 MB `.app`, ~459 MB DMG).
> Linux + Windows paths are documented but not yet smoke-tested. Expect
> minor gaps; report what breaks.

The flow is the same on every host:

1. **Freeze the Python backend** with PyInstaller → single
   `podcodex-server` binary (~420 MB).
2. **Fetch native sidecars** (`ffmpeg`, `yt-dlp`) for the host triple.
3. **Build the frontend** (`npm run build`).
4. **Bundle with Tauri** (`cargo tauri build`) → native installer.

`make bundle` chains all four. Use the individual `make bundle-*` targets
when iterating on a single layer.

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

### macOS gotchas

- WhisperX runs CPU-only on Apple Silicon (no MPS support upstream).
- ffmpeg bundled is GPL (`libx264` / `libx265`). For redistribution under
  MIT, swap to an LGPL audio-only build by editing
  `scripts/fetch_native_binaries.py` `FFMPEG_SOURCES`.
- First launch each session pays a 10-30 s cold start while PyInstaller
  extracts to `/tmp/_MEIxxx/`. Warm launches are <1 s.

---

## Linux (Ubuntu 22.04 / 24.04 native)

```bash
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev \
                 libayatana-appindicator3-dev librsvg2-dev libssl-dev \
                 pkg-config libsndfile1 libfuse2
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli --version "^2"
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/gabriel-jung/podcodex && cd podcodex
make setup
make setup-pyinstaller
make bundle               # produces .deb + .AppImage
```

Outputs:

- `src-tauri/target/release/bundle/deb/podcodex_<version>_amd64.deb`
- `src-tauri/target/release/bundle/appimage/podcodex_<version>_amd64.AppImage`

### Linux gotchas

- `libfuse2` is only needed if you want the `.AppImage`. Skip if you only
  ship the `.deb`.
- Older glibc (Ubuntu <22.04, Debian <12) breaks PyInstaller's
  manylinux2014 wheels for torch / ctranslate2.
- The `.deb` declares dependencies on
  `libwebkit2gtk-4.1-0`, `libgtk-3-0`, `libayatana-appindicator3-1`,
  `librsvg2-2`, `libsndfile1`. End users get them auto-installed by
  `apt install ./podcodex_*.deb`.

---

## WSL2 (Ubuntu inside Windows)

WSL2 builds **Linux artifacts** (`.deb` / `.AppImage`), not Windows
`.exe`. To produce a Windows `.exe`, build on a real Windows host (see
below) — WSL cannot cross-compile Tauri to MSVC.

```bash
# Inside WSL Ubuntu 22.04+
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev \
                 libayatana-appindicator3-dev librsvg2-dev libssl-dev \
                 pkg-config libsndfile1 libfuse2
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli --version "^2"
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/gabriel-jung/podcodex && cd podcodex
make setup
make setup-pyinstaller
make bundle
```

### WSL2 gotchas

- The Tauri webview window renders through **WSLg** (Windows 11 only).
  On Windows 10 hosts, no native window — use `make dev-no-tauri` and
  open the browser at `http://localhost:5173`.
- Don't run from `/mnt/c/...` — fs translation slows PyInstaller 5-10×
  and breaks some symlinks. Clone into the WSL home directory.
- The produced `.deb` / `.AppImage` are Linux ELF binaries; copy them to
  a Linux host or run them inside WSL itself for testing.
- WSL networking quirks: if `/api/health` doesn't reach the sidecar from
  the webview, check that `127.0.0.1:18811` is reachable inside WSL
  (`curl http://127.0.0.1:18811/api/health`).

---

## Windows 11 native (.exe / .msi)

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
make bundle               # produces .msi + .exe (NSIS)
```

Outputs:

- `src-tauri\target\release\bundle\msi\PodCodex_<version>_x64_en-US.msi`
- `src-tauri\target\release\bundle\nsis\PodCodex_<version>_x64-setup.exe`

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
2. Launch the bundle from its installed location (e.g. open the `.app`,
   `dpkg -i` the `.deb`, install the `.msi`).
3. Confirm the splash phase messages appear during cold start (10-30 s
   on first launch each session).
4. Step through the [`deploy/SMOKE.md`](SMOKE.md) checklist.

If `/api/health` never returns and the splash stays on "Almost ready,
hang tight…" past ~60 s, run from a terminal to surface stderr:

- macOS: `PodCodex.app/Contents/MacOS/podcodex-app 2>&1 | head -50`
- Linux: `./podcodex-app 2>&1 | head -50`

The bundled sidecar logs to `<app_data>/logs/server.log`:

| OS      | `<app_data>` path                                        |
|---------|----------------------------------------------------------|
| macOS   | `~/Library/Application Support/com.podcodex.desktop/`    |
| Linux   | `~/.local/share/com.podcodex.desktop/`                   |
| Windows | `%APPDATA%\com.podcodex.desktop\`                        |

---

## Reporting build issues

Open an issue at <https://github.com/gabriel-jung/podcodex/issues> with:

- OS + version (`uname -a`, `sw_vers`, `winver`)
- `python --version`, `node --version`, `rustc --version`,
  `cargo tauri --version`
- The full `make bundle` output (use `tee build.log`)
- For PyInstaller failures: `packaging/build_work/podcodex-server/warn-podcodex-server.txt`
