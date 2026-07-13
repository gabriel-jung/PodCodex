# Installing the PodCodex Discord bot

Two install paths. Pick whichever fits your setup:

- [Path A: uv (native)](#path-a-uv-native): lightweight, good for a dedicated VPS or running alongside the desktop app
- [Path B: Docker](#path-b-docker): isolated runtime, good for multi-service hosts

After either path, see:

- [Access control](#access-control-passwords): optional; only if you host multiple shows and want each Discord server to see a different one
- [Transferring the index](#transferring-the-index): only if the bot runs on a different machine than the desktop app

---

## Before you start

You need four things before either install path:

### 1. A Discord bot application and token

1. Go to <https://discord.com/developers/applications> and click **New Application**.
2. Name it, open the **Bot** tab, and click **Reset Token** to reveal and copy the token. Save it. Discord only shows it once.
3. Leave **Privileged Gateway Intents** off. The bot uses only default intents.
4. Under **OAuth2 → URL Generator**, select:
   - Scopes: `bot` and `applications.commands`
   - Bot Permissions: `Send Messages`, `Embed Links`, `Read Message History`
5. Visit the generated URL and invite the bot to each Discord server you want it in.

**If the token ever leaks**, click **Reset Token** to invalidate it; the old one stops working immediately.

### 2. A host machine

Requirements:

- **Python 3.12** specifically (not 3.11, not 3.13). The Docker image bundles it. For the uv path, `uv sync` downloads and pins Python 3.12 into the project's virtual environment, so your system Python version doesn't matter.
- **1-10 GB RAM**, depending on the embedder. BGE-M3 (the default) uses ~2.5 GB; smaller models cut this to ~1 GB, larger ones (Perplexity 4B) need ~10 GB. See [Resource requirements](#resource-requirements).
- Outbound internet for the bot's connection to Discord's gateway. No inbound ports needed.

The bot is a read-only frontend over an existing index, so it does **not** need:

- `ffmpeg`: only needed during transcription (desktop app)
- A GPU or `CUDA`: only needed for local transcription/indexing (desktop app)
- `HF_TOKEN`: only needed for speaker diarization (desktop app)

### 3. An index to serve

The bot is read-only and doesn't build indexes itself. You need an existing LanceDB index, produced by the desktop app's **Index** step on any show.

The desktop app writes to its platform `<data_dir>/index/`: `~/.local/share/podcodex/index/` on Linux, `~/Library/Application Support/podcodex/index/` on macOS, `%APPDATA%\podcodex\index\` on Windows. The bot resolves the same path and looks there first, so in most setups there is nothing to configure. If it doesn't find an index there, it also checks `./deploy/index/` and `./index/` relative to its working directory. Set `PODCODEX_INDEX=/abs/path` to override explicitly.

- **Bot on the same machine as the desktop app** → nothing to do. The bot finds the desktop's index automatically.
- **Bot on a different machine (VPS, server)** → rsync the index over, see [Transferring the index](#transferring-the-index).

The bot logs the resolved path and the reason on startup (`IndexStore opened: <path> (<reason>)`). If you've never run the Index step, the bot will start but every command will return empty results.

### 4. About slash command sync

Discord propagates globally-scoped slash commands lazily; commands may take up to an hour to appear in servers after the bot's first start. They're available instantly in the Discord client's command picker once propagated.

For faster iteration during testing, start the bot with `--dev-guild <GUILD_ID>`; commands sync instantly to that one guild.

Once propagated, commands stay registered; restarts don't re-trigger the wait.

---

## Path A: uv (native)

### 1. Install uv

Follow the official install guide: <https://docs.astral.sh/uv/getting-started/installation/>.

### 2. Clone and sync

```bash
git clone https://github.com/gabriel-jung/PodCodex.git
cd PodCodex
uv sync --extra bot --extra rag --extra cpu --no-dev
```

This creates `.venv/` with all runtime dependencies (~1.5 GB).

Both trailing flags matter on a typical GPU-less VPS:

- `--extra cpu` pulls the CPU-only PyTorch wheel. Without it, torch resolves from PyPI, which on Linux means the CUDA build plus ~6 GB of `nvidia-*` and `triton` packages the host can never use.
- `--no-dev` skips the dev dependency group (Jupyter, pre-commit, pytest), which `uv sync` installs by default.

If the VPS does have an NVIDIA GPU and you want the embedder on it, swap `--extra cpu` for `--extra gpu` (or `--extra gpu-pascal` on GTX 10xx / P40 / P100, see [PASCAL.md](PASCAL.md)). The extras are mutually exclusive.

Already synced without these flags? Re-run the command above with `--reinstall` to drop the CUDA stack.

### 3. Create `.env` at the repo root

The bot's `load_dotenv` searches from the current working directory upward, so the repo root is the reliable location. Write the file directly:

```bash
cat > .env <<'EOF'
DISCORD_TOKEN=your-bot-token

# Optional, only if you want to point the bot at an index that is neither
# at <data_dir>/index (e.g. ~/.local/share/podcodex/index on
# Linux) nor at ./deploy/index / ./index.
# PODCODEX_INDEX=/absolute/path/to/index
EOF
```

Then open `.env` and fill in `DISCORD_TOKEN`. `.env` is gitignored.

**How the bot finds the index** (logged at startup):

1. `PODCODEX_INDEX` env var if set: always wins.
2. `<data_dir>/index/` if it exists with data: desktop app default (Linux: `~/.local/share/podcodex/index/`).
3. `./deploy/index/` or `./index/` relative to the bot's working directory: repo-local fallback.
4. Else: creates an empty `<data_dir>/index/`.

Check the startup log line (`IndexStore opened: <path> (<reason>)`) to confirm which one was picked.

### 4. Start the bot

```bash
uv run podcodex-bot
```

Initial start downloads the BGE-M3 model (~2.5 GB). Subsequent starts are instant.

Verify in Discord:

```text
/stats
/search question:hello world
```

### 5. Auto-restart (systemd)

Drop this at `/etc/systemd/system/podcodex-bot.service` (adjust paths and user):

```ini
[Unit]
Description=PodCodex Discord Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/PodCodex
EnvironmentFile=/home/ubuntu/PodCodex/.env
ExecStart=/home/ubuntu/.local/bin/uv run podcodex-bot
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and follow logs:

```bash
sudo systemctl enable --now podcodex-bot
sudo journalctl -u podcodex-bot -f
```

---

## Path B: Docker

Assumes Docker Engine with the compose plugin is installed.

```bash
git clone https://github.com/gabriel-jung/PodCodex.git
cd PodCodex/deploy
cp .env.example .env           # edit: DISCORD_TOKEN, optional provider key
docker compose up -d --build bot
docker compose logs -f bot
```

Update:

```bash
git pull && docker compose up -d --build bot
```

Notes:

- The host's `~/.local/share/podcodex/index/` is mounted into the container at `/root/.local/share/podcodex/index/`, matching the bot's default, so no `PODCODEX_INDEX` override is needed.
- To serve an index at a different host location, set `PODCODEX_INDEX_HOST=/abs/path` in `deploy/.env` before `docker compose up`.
- BGE-M3 lives in the `model_cache` named volume; survives rebuilds.
- `restart: unless-stopped` handles crashes and host reboots.
- Logs rotate at 50 MB × 3 files via the json-file driver.

---

## Access control (passwords)

**Optional. Skip this section entirely unless you need it.** By default every indexed show is visible to every Discord server the bot is in. That's the right setup for a personal bot, or when one bot serves one audience.

You only need passwords when you're running a single bot process that carries **multiple shows**, and each show should be available to a **different Discord server**. For example, one bot hosting Show A for server A's listeners and Show B for server B's listeners, without either side seeing the other show in `/stats` or `/search`.

How it works: passwords flip a show to invisible by default. An admin in each Discord server runs `/unlock password:****` once to reveal the corresponding show only there. Passwords live in `_show_passwords.lance` inside the index directory and ship with the index via rsync.

### Set, rotate, or remove passwords

Stop the bot, then run:

```bash
# Path A (uv)
uv run podcodex-bot --manage-passwords

# Path B (Docker)
docker compose run --rm bot --manage-passwords
```

Interactive prompt lists all indexed shows and lets you set, generate (`g`), or remove passwords. Generated passwords print once; copy them before dismissing.

Restart the bot so it picks up the new password map:

```bash
sudo systemctl restart podcodex-bot   # uv
docker compose restart bot            # docker
```

### Unlock in Discord

Per Discord server, an admin with `manage_guild` runs:

```text
/unlock password:****
/changepassword show:<name>   # rotate, DMs the new password
/lock show:<name>             # remove a show
```

All three responses are ephemeral; other users see nothing.

---

## Transferring the index

**Skip this section if the bot runs on the same machine as the desktop app.** The bot finds the index automatically.

Only transfer when the bot runs on a separate machine (e.g. a VPS).

The bot host (Linux) reads from `~/.local/share/podcodex/index/` by default. The source path on the desktop machine depends on its OS:

| Desktop OS | Source path |
| ---------- | ----------- |
| Linux      | `~/.local/share/podcodex/index/` |
| macOS      | `~/Library/Application Support/podcodex/index/` |
| Windows    | `%APPDATA%\podcodex\index\` (rsync not native, see [From a Windows desktop](#from-a-windows-desktop)) |

If you overrode `PODCODEX_DATA_DIR` on the desktop, the index lives under `<override>/index/` instead.

### 1. Create the target directory on the bot host

rsync does not create parent directories:

```bash
ssh user@host 'mkdir -p ~/.local/share/podcodex/index'
```

### 2. rsync from the indexing machine

**Run as a single line.** Multi-line paste without `\` continuations fails in zsh. Trailing slash on source matters (copies contents, not the dir itself). Use `--delete` so renamed/removed shows on the desktop don't leave stale tables on the bot.

From Linux desktop:

```bash
# Dry run first: prints what would transfer or delete, changes nothing
rsync -avn --delete --progress ~/.local/share/podcodex/index/ user@host:~/.local/share/podcodex/index/

# Real copy
rsync -av --delete --progress ~/.local/share/podcodex/index/ user@host:~/.local/share/podcodex/index/
```

From macOS desktop (note the escaped space):

```bash
rsync -avn --delete --progress ~/Library/Application\ Support/podcodex/index/ user@host:~/.local/share/podcodex/index/
rsync -av --delete --progress ~/Library/Application\ Support/podcodex/index/ user@host:~/.local/share/podcodex/index/
```

Safe to run while the bot is running; LanceDB is read-only on the bot side.

### Per-show sync

The full directory is the unit of transfer. Per-show selective sync is technically possible (rsync include/exclude on the `{show}__*.lance` tables), but `_collections.lance` and `_show_passwords.lance` are global registries; partial syncs leave them inconsistent. Transfer the whole directory.

### From a Windows desktop

rsync isn't native on Windows. Use the bundle path (below). `podcodex-export` resolves `%APPDATA%\podcodex\index\` automatically and produces a single file you can move by any means (scp, web upload, USB drive). WSL + rsync also works if you prefer parity with the Linux flow:

```powershell
wsl rsync -av --delete --progress /mnt/c/Users/<you>/AppData/Roaming/podcodex/index/ user@host:~/.local/share/podcodex/index/
```

### Alternative: bundle archive (selective, atomic)

When you don't have rsync access (no SSH symmetry, restricted firewall, web upload, USB) or want to deploy a *subset* of shows, use the `.podcodex` bundle format:

```bash
# Indexing machine: pick specific shows or use --all for parity with rsync
podcodex-export "Show A" "Show B" --index-only -o shows-index.podcodex
# or every show:
podcodex-export --all --index-only -o shows-index.podcodex

# Transfer (any path: scp, web, S3)
scp shows-index.podcodex user@host:/tmp/

# Bot host: replaces existing collections atomically
podcodex-import /tmp/shows-index.podcodex --on-conflict replace
```

Bundle format records each collection's embedding model + chunker in a manifest so the importer can warn if a model isn't installed. rsync stays the canonical option for "ship everything, fast updates"; bundle wins on selective deploy and atomic transfer without SSH.

---

## Command reference

| Command                              | Who      | Description                                      |
| ------------------------------------ | -------- | ------------------------------------------------ |
| `/search question`                   | Everyone | Hybrid keyword + semantic search (server defaults) |
| `/search-advanced question […]`      | Everyone | Search with full control over retrieval tuning   |
| `/exact query`                       | Everyone | Literal substring match (case-insensitive, like Ctrl+F) |
| `/exact-advanced query […]`          | Everyone | Literal match with source and date filters       |
| `/random`                            | Everyone | Random quote                                     |
| `/random-advanced […]`               | Everyone | Random quote with source and date filters        |
| `/stats [show]`                      | Everyone | Index overview: shows, episodes, segments, duration |
| `/episodes show`                     | Everyone | List episodes for a show with segment count + duration |
| `/speakers [show]`                   | Everyone | Chunk count and airtime per speaker              |
| `/help`                              | Everyone | Show available commands                          |
| `/setup [model] [top_k] …`           | Admin    | Configure server defaults                        |
| `/announcements [channel] [off]`     | Admin    | Channel for new-episode and bot-version updates  |
| `/unlock password`                   | Admin    | Unlock a show (password identifies the show)     |
| `/lock show`                         | Admin    | Remove a show from this server                   |
| `/changepassword show`               | Admin    | Rotate password for an unlocked show             |
| `/sync`                              | Admin    | Manually re-sync slash commands                  |
| `/admin-reload`                      | Admin    | Reconnect to the index and reload show passwords |

The bot polls the index for changes (default every 10 min, `--announce-interval`
to tune) and posts newly indexed episodes to each server's configured channel.
The first pass after enabling records the existing catalogue silently — only
episodes added afterwards are announced. Locked shows are only announced on
servers that unlocked them.

---

## Resource requirements

| Setup                | RAM         | Notes                                                  |
| -------------------- | ----------- | ------------------------------------------------------ |
| Bot (E5 small)       | ~1 GB       | Smallest, fastest, lower search quality                |
| Bot (BGE-M3)         | ~2.5 GB     | Default, good search quality                           |
| Bot (Perplexity 4B)  | ~10 GB      | Best quality, needs a beefier host                     |
| Many shows (20+)     | negligible  | LanceDB scales well, RAM stays mostly flat             |
