# PodCodex Desktop App — Development & Build
#
# Prerequisites:
#   All platforms:
#     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#     cargo install tauri-cli --version "^2"
#
#   macOS:
#     brew install node
#
#   Ubuntu/Debian (WSL2 included — dev only, no shipped Linux artifact):
#     sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev \
#       libayatana-appindicator3-dev librsvg2-dev libssl-dev pkg-config \
#       libsndfile1
#
#   Fedora (dev only):
#     sudo dnf install webkit2gtk4.1-devel gtk3-devel \
#       libappindicator-gtk3-devel librsvg2-devel openssl-devel pkg-config \
#       libsndfile
#
# Quick start:
#   make setup   # one-time: install all deps
#   make dev     # start FastAPI + Vite + Tauri (hot-reload)
#   make build   # production .dmg (macOS) or .msi (Windows) bundle

.PHONY: setup setup-python setup-frontend setup-pyinstaller dev dev-api dev-frontend dev-tauri dev-sidecar build bundle bundle-gpu bundle-server bundle-server-gpu package-gpu bundle-natives bundle-sign bundle-sign-only clean

# ── Setup ────────────────────────────────────────────────

setup: setup-python setup-frontend  ## One-time setup: install all dependencies
	@echo "\n✅ Setup complete. Run 'make dev' to start developing."

setup-python:  ## Install Python deps (desktop + pipeline + rag + youtube + mcp)
	uv sync --extra desktop --extra pipeline --extra rag --extra youtube --extra mcp

setup-frontend:  ## Install frontend deps
	cd frontend && npm install

# ── Development ──────────────────────────────────────────

dev:  ## Start all three services (FastAPI + Vite + Tauri) — Ctrl+C to stop
	@echo "Starting PodCodex dev environment..."
	@echo "  API:      http://127.0.0.1:18811"
	@echo "  Frontend: http://localhost:5173"
	@echo ""
	@trap 'kill 0' EXIT; \
	$(MAKE) dev-api & \
	$(MAKE) dev-frontend & \
	sleep 3 && $(MAKE) dev-tauri & \
	wait

dev-api:  ## Start FastAPI backend (port 18811, auto-reload)
	.venv/bin/uvicorn podcodex.api.app:app --host 127.0.0.1 --port 18811 --reload --reload-dir src

dev-frontend:  ## Start Vite dev server (port 5173)
	cd frontend && npm run dev

dev-tauri: dev-sidecar  ## Start Tauri webview (connects to Vite dev server)
	cd src-tauri && PODCODEX_SKIP_BACKEND_SPAWN=1 . $$HOME/.cargo/env && cargo tauri dev

dev-sidecar:  ## Drop a placeholder sidecar binary so cargo tauri dev compiles
	.venv/bin/python scripts/setup_dev_sidecar.py

dev-no-tauri:  ## Start just API + Vite (use browser at localhost:5173)
	@trap 'kill 0' EXIT; \
	$(MAKE) dev-api & \
	$(MAKE) dev-frontend & \
	wait

# ── Production Build ─────────────────────────────────────

setup-pyinstaller:  ## Install PyInstaller into the dev venv
	uv pip install pyinstaller --python .venv/bin/python

bundle-server: setup-pyinstaller  ## Freeze the CPU Python backend as a sidecar binary
	.venv/bin/python packaging/build_server.py

bundle-server-gpu: setup-pyinstaller  ## Freeze the GPU sidecar (--onedir, includes CUDA libs)
	.venv/bin/python packaging/build_server.py --gpu

package-gpu:  ## Split the GPU --onedir build into server-core + cuda-libs release archives
	.venv/bin/python packaging/package_gpu.py

bundle-gpu: bundle-server-gpu package-gpu  ## Build GPU sidecar and package release archives + manifest

bundle-natives:  ## Download yt-dlp static binary for the host (ffmpeg ships via imageio-ffmpeg)
	.venv/bin/python scripts/fetch_native_binaries.py

bundle: bundle-server bundle-natives  ## Full standalone .dmg / .msi (frontend + CPU sidecar + Tauri)
	cd frontend && npm run build
	cd src-tauri && . $$HOME/.cargo/env && cargo tauri build

bundle-sign:  ## Sign + notarize the macOS bundle (requires APPLE_SIGNING_IDENTITY)
	scripts/sign_and_notarize.sh

bundle-sign-only:  ## Sign without notarizing (faster iteration)
	scripts/sign_and_notarize.sh --no-notary

build: bundle  ## Alias for `bundle` — unsigned .dmg (macOS) or .msi (Windows) with bundled backend

# ── Utilities ────────────────────────────────────────────

clean:  ## Remove build artifacts
	rm -rf frontend/dist frontend/node_modules/.vite
	rm -rf src-tauri/target src-tauri/binaries
	rm -rf packaging/dist packaging/build_work

test:  ## Run Python tests
	.venv/bin/python -m pytest tests/ -x -q

types:  ## Regenerate frontend TS types from Pydantic models
	.venv/bin/python scripts/generate_types.py

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
