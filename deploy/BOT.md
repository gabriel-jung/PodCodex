# Installing the PodCodex Discord bot

Two install paths — pick whichever fits your setup:

- [Path A — uv (native)](#path-a--uv-native) — lightweight, good for a dedicated VPS or running alongside the desktop app
- [Path B — Docker](#path-b--docker) — isolated runtime, good for multi-service hosts

After either path, see:

- [Access control](#access-control-passwords) — optional; only if you host multiple shows and want each Discord server to see a different one
- [Transferring the index](#transferring-the-index) — only if the bot runs on a different machine than the desktop app

---

## Before you start

You need four things before either install path:

### 1. A Discord bot application and token

1. Go to <https://discord.com/developers/applications> and click **New Application**.
2. Name it, open the **Bot** tab, and click **Reset Token** to reveal and copy the token. Save it — Discord only shows it once.
3. Leave **Privileged Gateway Intents** off. The bot uses only default intents.
4. Under **OAuth2 → URL Generator**, select:
   - Scopes: `bot` and `applications.commands`
   - Bot Permissions: `Send Messages`, `Embed Links`, `Read Message History`, `Use Slash Commands`
5. Visit the generated URL and invite the bot to each Discord server you want it in.

**If the token ever leaks**, click **Reset Token** to invalidate it — the old one stops working immediately.

### 2. A host machine

Requirements:

- **Python 3.12** specifically (not 3.11, not 3.13). The Docker image bundles it. For the uv path, `uv sync` downloads and pins Python 3.12 into the project's virtual environment — your system Python version doesn't matter.
- **~3 GB RAM**, mostly for the BGE-M3 query embedder. See [Resource requirements](#resource-requirements).
- Outbound internet — the bot connects to Discord's gateway. No inbound ports needed.

The bot is a read-only frontend over an existing index, so it does **not** need:

- `ffmpeg` — only needed during transcription (desktop app)
- A GPU or `CUDA` — only needed for local transcription/indexing (desktop app)
- `HF_TOKEN` — only needed for speaker diarization (desktop app)

### 3. An index to serve

The bot is read-only — it doesn't build indexes itself. You need an existing LanceDB index, produced by the desktop app's **Index** step on any show.

The desktop app writes to `~/.local/share/podcodex/index/`. The bot looks there first, so in most setups there is nothing to configure. If it doesn't find an index there, it also checks `./deploy/index/` and `./index/` relative to its working directory. Set `PODCODEX_INDEX=/abs/path` to override explicitly.

- **Bot on the same machine as the desktop app** → nothing to do. The bot finds the desktop's index automatically.
- **Bot on a different machine (VPS, server)** → rsync the index over, see [Transferring the index](#transferring-the-index).

The bot logs the resolved path and the reason on startup (`IndexStore opened: <path> (<reason>)`). If you've never run the Index step, the bot will start but every command will return empty results.

### 4. About slash command sync

Discord propagates globally-scoped slash commands lazily — commands may take up to an hour to appear in servers after the bot's first start. They're available instantly in the Discord client's command picker once propagated.

For faster iteration during testing, start the bot with `--dev-guild <GUILD_ID>` — commands sync instantly to that one guild.

Once propagated, commands stay registered; restarts don't re-trigger the wait.

---

## Path A — uv (native)

### 1. Install uv

Follow the official install guide: <https://docs.astral.sh/uv/getting-started/installation/>.

### 2. Clone and sync

```bash
git clone https://github.com/<you>/PodCodex.git
cd PodCodex
uv sync --extra bot --extra rag
```

This creates `.venv/` with all runtime dependencies.

### 3. Create `.env` at the repo root

The bot's `load_dotenv` searches from the current working directory upward — the repo root is the reliable location. Write the file directly:

```bash
cat > .env <<'EOF'
DISCORD_TOKEN=your-bot-token

# Optional — only if you want to point the bot at an index that is neither
# at ~/.local/share/podcodex/index nor at ./deploy/index / ./index.
# PODCODEX_INDEX=/absolute/path/to/index
EOF
```

Then open `.env` and fill in `DISCORD_TOKEN`. `.env` is gitignored.

**How the bot finds the index** (logged at startup):

1. `PODCODEX_INDEX` env var if set — always wins.
2. `~/.local/share/podcodex/index/` if it exists with data — desktop app default.
3. `./deploy/index/` or `./index/` relative to the bot's working directory — repo-local fallback.
4. Else: creates an empty `~/.local/share/podcodex/index/`.

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

## Path B — Docker

Assumes Docker Engine with the compose plugin is installed.

```bash
git clone https://github.com/<you>/PodCodex.git
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

- The host's `~/.local/share/podcodex/index/` is mounted into the container at the same path, matching the bot's default — no `PODCODEX_INDEX` override needed.
- To serve an index at a different host location, set `PODCODEX_INDEX_HOST=/abs/path` in `deploy/.env` before `docker compose up`.
- BGE-M3 lives in the `model_cache` named volume — survives rebuilds.
- `restart: unless-stopped` handles crashes and host reboots.
- Logs rotate at 50 MB × 3 files via the json-file driver.

---

## Access control (passwords)

**Optional — skip this section entirely unless you need it.** By default every indexed show is visible to every Discord server the bot is in. That's the right setup for a personal bot, or when one bot serves one audience.

You only need passwords when you're running a single bot process that carries **multiple shows**, and each show should be available to a **different Discord server** — e.g. one bot hosting Show A for server A's listeners and Show B for server B's listeners, without either side seeing the other show in `/stats` or `/search`.

How it works: passwords flip a show to invisible by default. An admin in each Discord server runs `/unlock password:****` once to reveal the corresponding show only there. Passwords live in `_show_passwords.lance` inside the index directory — they ship with the index via rsync.

### Set, rotate, or remove passwords

Stop the bot, then run:

```bash
# Path A (uv)
uv run podcodex-bot --manage-passwords

# Path B (Docker)
docker compose run --rm bot --manage-passwords
```

Interactive prompt lists all indexed shows and lets you set, generate (`g`), or remove passwords. Generated passwords print once — copy them before dismissing.

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

All three responses are ephemeral — other users see nothing.

---

## Transferring the index

**Skip this section if the bot runs on the same machine as the desktop app** — the bot finds the index automatically.

Only transfer when the bot runs on a separate machine (e.g. a VPS). Both install paths read from `~/.local/share/podcodex/index/` by default, so the rsync target is the same regardless of which path you picked.

### 1. Create the target directory on the bot host

rsync does not create parent directories:

```bash
ssh user@host 'mkdir -p ~/.local/share/podcodex/index'
```

### 2. rsync from the indexing machine

**Run as a single line** — multi-line paste without `\` continuations fails in zsh. Trailing slash on source matters (copies contents, not the dir itself):

```bash
# Dry run first — prints what would transfer, changes nothing
rsync -avn --progress ~/.local/share/podcodex/index/ user@host:~/.local/share/podcodex/index/

# Real copy
rsync -av --progress ~/.local/share/podcodex/index/ user@host:~/.local/share/podcodex/index/
```

If your source is elsewhere (you overrode `PODCODEX_INDEX` on the desktop), swap the local path accordingly.

Safe to run while the bot is running — LanceDB is read-only on the bot side.

### Per-show sync

The full directory is the unit of transfer. Per-show selective sync is technically possible (rsync include/exclude on the `{show}__*.lance` tables), but `_collections.lance` and `_show_passwords.lance` are global registries — partial syncs leave them inconsistent. Transfer the whole directory.

### Alternative: bundle archive (selective, atomic)

When you don't have rsync access (no SSH symmetry, restricted firewall, web upload, USB) or want to deploy a *subset* of shows, use the `.podcodex` bundle format:

```bash
# Indexing machine — pick specific shows or use --all for parity with rsync
podcodex-export "Show A" "Show B" --index-only -o shows-index.podcodex
# or every show:
podcodex-export --all --index-only -o shows-index.podcodex

# Transfer (any path: scp, web, S3)
scp shows-index.podcodex user@host:/tmp/

# Bot host — replaces existing collections atomically
podcodex-import /tmp/shows-index.podcodex --on-conflict replace
```

Bundle format records each collection's embedding model + chunker in a manifest so the importer can warn if a model isn't installed. rsync stays the canonical option for "ship everything, fast updates" — bundle wins on selective deploy and atomic transfer without SSH.

---

## Command reference

| Command                              | Who      | Description                                      |
| ------------------------------------ | -------- | ------------------------------------------------ |
| `/search question [show] [...]`      | Everyone | Hybrid keyword + semantic search                 |
| `/exact query [show] [...]`          | Everyone | Literal substring match (+ accent variants, 1-edit typos) |
| `/random [show] [...]`               | Everyone | Random quote                                     |
| `/stats [show]`                      | Everyone | Index overview                                   |
| `/episodes show`                     | Everyone | List episodes for a show                         |
| `/help`                              | Everyone | Show available commands                          |
| `/setup [model] [top_k] …`           | Admin    | Configure server defaults                        |
| `/unlock password`                   | Admin    | Unlock a show (password identifies the show)     |
| `/lock show`                         | Admin    | Remove a show from this server                   |
| `/changepassword show`               | Admin    | Rotate password for an unlocked show             |
| `/sync`                              | Admin    | Manually re-sync slash commands                  |

---

## Resource requirements

| Setup              | RAM     | Notes                                   |
| ------------------ | ------- | --------------------------------------- |
| Bot                | ~2.5 GB | BGE-M3 for query embedding              |
| Many shows (20+)   | ~3 GB   | LanceDB scales well; RAM stays flat     |
