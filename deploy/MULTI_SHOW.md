# Deploying the Bot

How to serve one or more podcast shows from a VPS, with optional per-server access control.

## Architecture

```text
Local machine (GPU)                   VPS
───────────────────                   ───
Desktop app                           podcodex-bot
  transcribe (WhisperX)    rsync →      LanceDB index (read-only)
  correct / translate                   BGE-M3 on CPU (query embedding)
  index (BGE-M3 GPU)                    Discord slash commands
  ~/.local/share/podcodex/index/
```

**Key principles:**

- **Compute locally, serve remotely** — GPU work (transcription, indexing) happens on your machine. The VPS only embeds queries (CPU, ~2.5 GB RAM) and serves results.
- **One bot, one process, one token** — serves all Discord servers from one container.
- **Shows invisible by default** — access control via `shows.toml`; admins unlock with `/unlock`.
- **No database on VPS** — the LanceDB index directory is self-contained. rsync it and done.

---

## Initial setup

### 1. Index shows locally

Use the desktop app: add a show, run Transcribe → Correct → Index. The Index step writes the LanceDB index into:

```
~/.local/share/podcodex/index/
```

Each show gets its own collection table inside that directory. You can index multiple shows — they all go into the same directory.

### 2. Transfer the index to VPS

```bash
# First time — full copy
rsync -av --progress \
  ~/.local/share/podcodex/index/ \
  vps:/path/to/deploy/data/index/

# After adding or updating a show — incremental
rsync -av --progress \
  ~/.local/share/podcodex/index/ \
  vps:/path/to/deploy/data/index/
```

rsync is safe to run while the bot is running — LanceDB is read-only on the bot side.

### 3. Configure environment

```bash
cd deploy
cp .env.example .env.production
# Edit .env.production:
#   DISCORD_TOKEN=<your bot token>
#   PODCODEX_INDEX=/app/data/index   ← already set in example
```

If you want `/ask` (LLM synthesis), add the key for your provider:

```bash
# In .env.production — add whichever provider you'll use
OPENAI_API_KEY=sk-...
# or MISTRAL_API_KEY=...
# or ANTHROPIC_API_KEY=...
```

### 4. Install and start

**Option A — uv (simpler, recommended for single-service VPS):**

```bash
# On the VPS
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --python 3.12 \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "podcodex[bot,rag]"

# Run (reads DISCORD_TOKEN and PODCODEX_INDEX from env / .env file)
podcodex-bot
```

Use a systemd unit for restarts:

```ini
# /etc/systemd/system/podcodex-bot.service
[Unit]
Description=PodCodex Discord Bot
After=network.target

[Service]
EnvironmentFile=/path/to/deploy/.env.production
ExecStart=/root/.local/bin/podcodex-bot
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now podcodex-bot
journalctl -u podcodex-bot -f
```

**Option B — Docker (useful for multi-service VPS or isolated environments):**

```bash
cd deploy
docker compose build bot
docker compose up -d bot
docker compose logs -f bot
```

Both options download BGE-M3 (~2.5 GB) on first run — subsequent starts are instant.

### 5. Verify

In any Discord server the bot has been invited to:

```
/stats          → shows indexed collections
/search question:hello world   → returns results
```

---

## Access control (multi-show, multi-server)

Skip this section if you're running a private bot with no password protection.

### Register shows

Run `--add-show` once per show — it prompts for name + password and appends to `shows.toml`:

```bash
docker compose run --rm bot --add-show --shows-config /app/shows.toml
# Show name: Les Pieds sur terre
# Password: ****
# Confirm password: ****
# Added 'Les Pieds sur terre' to shows.toml
```

Repeat for each show. The password you set is what server admins type in `/unlock`.

### Mount shows.toml

`docker-compose.yml` already mounts `./shows.toml:/app/shows.toml:ro`. If the file exists, access control activates automatically — all shows are invisible until unlocked.

### Restart to pick up changes

```bash
docker compose restart bot
```

### Unlock shows in Discord

In each Discord server, an admin with `manage_guild` runs:

```
/unlock show:Les Pieds sur terre password:****
```

The bot verifies the hash and grants access. Response is ephemeral. `/lock` removes a show.

---

## Configuring /ask (LLM synthesis)

`/ask` requires a provider to be configured per server (by an admin):

```
/setup ask_provider:openai ask_model:gpt-4o-mini
/setup ask_provider:mistral
/setup ask_provider:anthropic ask_model:claude-haiku-4-5-20251001
```

The API key is read from the env var for the provider (`OPENAI_API_KEY`, `MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`) — it is never stored in `server_config.json`. Set it in `.env.production` before starting the bot.

The ask cooldown (default 30s) can also be adjusted:

```
/setup ask_cooldown:60
```

---

## Adding a new show later

1. Desktop app: index the new show locally
2. `rsync` the index directory to VPS (incremental — only new files transferred)
3. If using access control: `docker compose run --rm bot --add-show --shows-config /app/shows.toml`
4. `docker compose restart bot` (to reload `shows.toml`)
5. Share the password with the server admin — they run `/unlock`

---

## Updating the bot

```bash
git pull
docker compose build bot
docker compose up -d bot   # zero-downtime rolling restart
```

The index and model cache volumes are preserved across rebuilds.

---

## Security notes

- `/unlock` and `/lock` responses are ephemeral — other users see nothing
- Password is never stored — only sha256 hash in `shows.toml`
- API keys for `/ask` are env-var only — not persisted to disk by the bot
- `/setup` blocks `show_add`/`show_remove` when access control is active — use `/unlock`/`/lock`

---

## Resource requirements

| Setup                  | RAM     | Notes                                          |
| ---------------------- | ------- | ---------------------------------------------- |
| Bot only (no /ask)     | ~2.5 GB | BGE-M3 for query embedding                     |
| Bot + /ask             | ~2.5 GB | LLM call is external API, no extra RAM         |
| Many shows (20+)       | ~3 GB   | LanceDB scales well; RAM stays flat            |
| Separate embedding svc | ~200 MB | Extract BGE-M3 into shared HTTP service (TEI)  |

---

## Command reference

| Command                              | Who      | Description                                      |
| ------------------------------------ | -------- | ------------------------------------------------ |
| `/search question [show] [...]`      | Everyone | Hybrid keyword + semantic search                 |
| `/ask question [show] [...]`         | Everyone | LLM-synthesized answer from transcript passages  |
| `/exact query [show] [...]`          | Everyone | Literal substring match                          |
| `/random [show] [...]`               | Everyone | Random quote                                     |
| `/stats [show]`                      | Everyone | Index overview                                   |
| `/episodes show`                     | Everyone | List episodes for a show                         |
| `/help`                              | Everyone | Show available commands                          |
| `/setup [model] [top_k] [ask_*] …`  | Admin    | Configure server defaults                        |
| `/unlock show password`              | Admin    | Unlock a show for this server                    |
| `/lock show`                         | Admin    | Remove a show from this server                   |
| `/sync`                              | Admin    | Manually re-sync slash commands                  |
