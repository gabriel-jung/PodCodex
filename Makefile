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
#   Ubuntu/Debian (WSL2 included):
#     sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev \
#       libayatana-appindicator3-dev librsvg2-dev libssl-dev pkg-config
#
#   Fedora:
#     sudo dnf install webkit2gtk4.1-devel gtk3-devel \
#       libappindicator-gtk3-devel librsvg2-devel openssl-devel pkg-config
#
# Quick start:
#   make setup   # one-time: install all deps
#   make dev     # start FastAPI + Vite + Tauri (hot-reload)
#   make build   # production .app / .exe bundle

.PHONY: setup setup-python setup-frontend dev dev-api dev-frontend dev-tauri build clean

# ── Setup ────────────────────────────────────────────────

setup: setup-python setup-frontend  ## One-time setup: install all dependencies
	@echo "\n✅ Setup complete. Run 'make dev' to start developing."

setup-python:  ## Install Python deps (core + desktop + ingest)
	uv pip install -e ".[desktop,ingest]" --python .venv/bin/python

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

dev-tauri:  ## Start Tauri webview (connects to Vite dev server)
	cd src-tauri && PODCODEX_SKIP_BACKEND_SPAWN=1 . $$HOME/.cargo/env && cargo tauri dev

dev-no-tauri:  ## Start just API + Vite (use browser at localhost:5173)
	@trap 'kill 0' EXIT; \
	$(MAKE) dev-api & \
	$(MAKE) dev-frontend & \
	wait

# ── Production Build ─────────────────────────────────────

build:  ## Build production .app / .exe
	cd frontend && npm run build
	cd src-tauri && . $$HOME/.cargo/env && cargo tauri build

# ── Utilities ────────────────────────────────────────────

clean:  ## Remove build artifacts
	rm -rf frontend/dist frontend/node_modules/.vite
	rm -rf src-tauri/target

test:  ## Run Python tests
	.venv/bin/python -m pytest tests/ -x -q

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
