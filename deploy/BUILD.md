# Building standalone bundles

How to produce a distributable PodCodex `.dmg` (macOS) or `.msi`
(Windows) from a clean checkout. Linux is not shipped; see [Linux](#linux-build-from-source)
below.

> **Status:** macOS arm64 verified end-to-end (~530 MB DMG, ~1.4 GB installed).
> Windows path is documented but not yet smoke-tested. Expect minor gaps;
> report what breaks.

The flow is the same on every shipped target:

1. **Freeze the Python backend** with PyInstaller (`--onedir`) →
   `src-tauri/binaries/server/` (CPU-only), bundled by Tauri as a
   resource rather than an externalBin file.
2. **Build the frontend** (`npm run build`).
3. **Bundle with Tauri** (`cargo tauri build`) → native installer.

yt-dlp ships inside the sidecar as its Python module; there is no separate binary. ffmpeg is **not** bundled; the app shells out to the user's system install (see [LICENSE_AUDIT.md](../LICENSE_AUDIT.md)).

`make bundle` chains all three. Use the individual `make bundle-*` targets when iterating on a single layer.

> **GPU acceleration:** the bundled installer is CPU-only. NVIDIA users
> activate an in-app CUDA backend (~2.4 GB download) from the GPU panel
> in settings. Pascal cards (sm_60-62) need the source-build path
> (`--extra gpu-pascal`); see [`PASCAL.md`](PASCAL.md).

---

## macOS (arm64 / x86_64)

```bash
brew install node rust uv ffmpeg
cargo install tauri-cli --version "^2"

git clone https://github.com/gabriel-jung/PodCodex && cd PodCodex
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
- `bundle_dmg.sh` can fail on the last step with nothing but
  `failed to run bundle_dmg.sh` in the tauri output. The real error is one
  line further down and tauri swallows it:

  ```
  hdiutil: couldn't unmount "diskN" - Resource busy
  ```

  The script mounts the staging image, has Finder lay the window out over
  AppleScript, then unmounts. Spotlight or Finder can still hold the volume
  at that point, and the script gives up on the first refusal. Recover
  without redoing the ~5 min of resource copying:

  ```bash
  cd src-tauri/target/release/bundle
  hdiutil detach /dev/diskN -force
  hdiutil convert macos/rw.*.dmg -format UDZO -imagekey zlib-level=9 \
      -o dmg/PodCodex_X.Y.Z_aarch64.dmg
  rm macos/rw.*.dmg
  ```

  Every failed run leaves its `rw.*.dmg` **attached**, and deleting the file
  does not return the space while the kernel holds it. `hdiutil info` lists
  the leaks; detach them before retrying, or the next run finds
  `/Volumes/PodCodex` taken and fails again for a different reason.
- `zlib-level=9` in the recovery above is not decoration: it is what tauri
  itself passes. Converting the same staging image without it yields a DMG
  about 50 MB larger, so dropping it would silently ship a fatter download
  than a normal `make bundle`.

---

## Linux (build from source)

PodCodex doesn't ship a Linux installer. Linux users run from source:

```bash
# Install Linux dev prereqs from the Makefile header (apt or dnf).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli --version "^2"
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/gabriel-jung/PodCodex && cd PodCodex
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

git clone https://github.com/gabriel-jung/PodCodex
cd PodCodex
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
uv sync --extra gpu-pascal   # cu126 wheels: Pascal only (sm_60-62), see PASCAL.md
```

Mutually exclusive; never enable both.

`PODCODEX_DEVICE=auto|cpu|cuda` overrides device selection at runtime. `cpu` skips GPU init even when CUDA is available; `cuda` makes model loaders that resolve a device (whisper, pyannote) raise if there is no CUDA. Default is `auto`. When the env var is unset, the choice made in the app's GPU panel (persisted as `device_override` in `<data_dir>/settings.json`) applies.

---

## Regenerated artifacts

A few files are checked in but auto-generated. Re-run after editing the source:

```bash
make types     # frontend/src/api/generated-types.ts from Pydantic models
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
| Frontend (TS / CSS) | `cd src-tauri && cargo tauri build` (its beforeBuildCommand runs `npm run build` in frontend/) |
| Rust shell only     | `cd src-tauri && cargo tauri build`                                |
| Python backend      | `make bundle-server && cd src-tauri && cargo tauri build`          |
| yt-dlp              | bump it in `uv.lock`, then `make bundle-server && cd src-tauri && cargo tauri build` |

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

The bundled sidecar logs to `<data_dir>/logs/server.log` (`~/Library/Application Support/podcodex/` on macOS, `%APPDATA%\podcodex\` on Windows, `~/.local/share/podcodex/` on Linux). Raw stdio of the sidecar and its step subprocesses (native crash output, faulthandler dumps, tracebacks from before logging starts) goes to `stdio.log` in the same folder, overwritten at each launch.

Report build issues at <https://github.com/gabriel-jung/PodCodex/issues>. Include:

- OS + version (`uname -a`, `sw_vers`, or `winver`).
- Tool versions: `python --version`, `node --version`, `rustc --version`, `cargo tauri --version`.
- Full `make bundle` output (use `tee build.log`).
- For PyInstaller failures, also attach `packaging/build_work/podcodex-server/warn-podcodex-server.txt`.

---

## Releasing (CI)

`release.yml` builds and publishes on a pushed `v*` tag. Bump first (`make bump VERSION=X.Y.Z`, see [CLAUDE.md § Versioning](../CLAUDE.md#versioning-bump-on-every-windows-msi-release)), commit, then tag. The tag name decides the flow:

| Tag | Result |
|-----|--------|
| `vX.Y.Z` | Stable release, becomes "latest". The README's `/releases/latest/download/...` links resolve to it. |
| `vX.Y.Z-beta.N`, `vX.Y.Z-rc.N` | Prerelease. "Latest" skips it, so the README's stable links keep serving the previous stable. Its assets are reachable at `/releases/download/<tag>/...`. |
| Actions UI, "Run workflow" | Draft, hidden until published by hand. |

After both platform builds, the `beta-channel` job copies the release's files to a rolling prerelease tagged `beta` when the build is newer than what that release holds (a stable `X.Y.Z` outranks every `X.Y.Z` prerelease). The README's beta links point there (`/releases/download/beta/<alias>`), so they always serve the newest build and never one older than stable. The release notes of `beta` start with `Build: <tag>`, which the job reads to compare versions; don't edit that line by hand.

The release workflow runs no tests. Check that the CI run on the commit you tag is green.

### Prerelease caveats

- **GPU backend download.** The app looks for its GPU manifest under its own version, `/releases/download/vX.Y.Z/cuda-libs.json`, which a prerelease tag doesn't match. It then tries the `beta` channel and uses it only when that manifest was built for the app's exact version, then "latest", whose manifest it refuses when built for another version. So a prerelease installs its GPU backend in one click. Every `X.Y.Z-beta.N` shares one version number, so the match also checks the build id: `build_server.py` bundles the commit a build is made from (`_podcodex_build.py`) and `package_gpu.py` stamps the same commit into the manifest. An older beta, superseded on the channel, is refused with a sentence saying to update rather than handed the newest beta's backend; update to the newest beta (the beta link always serves it). The same id guards an installed backend: `src-tauri/build.rs` stamps the commit into the desktop shell, which won't launch a backend whose manifest names another build (it falls back to CPU), and the GPU panel then asks to update, which downloads that build's server core instead of reusing the old one. A stable release can't pick up a beta's backend: its MSI is uploaded only after its own GPU files. `PODCODEX_GPU_MANIFEST_URL` still overrides the lookup (point it at `/releases/download/<tag>/cuda-libs.json`).
- **Windows MSI version.** An MSI version cannot carry `-beta.N`, so every prerelease of `X.Y.Z` and the stable `X.Y.Z` share one MSI version, and WiX skips replacing files on a same-version install. On a machine that has a prerelease installed, uninstall it before installing the next prerelease or the stable build (the README says so next to the beta links).
- **Updater.** The Tauri updater is off (`tauri.conf.json`), so prereleases reach only people who download them directly.

### Local `make bundle` and the DMG

`bundle_dmg.sh` sometimes fails locally (Finder layout over a mounted image); the `.app` under `src-tauri/target/release/bundle/macos/` is still complete, and CI builds the DMG. When the DMG step succeeds, Tauri deletes that `.app`; copy it out of the DMG to run it. A failed run can leave a multi-GB `rw.*.dmg` in `src-tauri/target/release/bundle/dmg/`; delete it.
