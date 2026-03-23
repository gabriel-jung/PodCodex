# Deploying Multiple Shows

How to serve multiple podcast shows from a single VPS with data isolation between Discord servers.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│  VPS                                                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  podcodex-bot --shows-config shows.toml              │   │
│  │  (single process, single Discord token)              │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│                  ┌──────┴──────┐                              │
│                  │   Qdrant    │                              │
│                  │  (shared)   │                              │
│                  └─────────────┘                              │
│                  Collections:                                 │
│                  ├─ les_pieds_sur_terre__bge-m3__semantic      │
│                  └─ transfert__bge-m3__semantic                │
└─────────────────────────────────────────────────────────────┘
```

**Key principles:**

- **One bot, one process, one token** — serves all Discord servers
- **Shows are invisible by default** — empty `allowed_shows` = no access
- **Password-gated discovery** — server admins unlock shows via `/unlock`
- **Complete data isolation** — a server can only query shows it has unlocked; no enumeration of other shows possible
- **Self-service** — server admins lock/unlock independently, no bot owner intervention needed

## Step-by-step setup

### 1. Index shows locally

On your machine, process each show's transcripts:

```bash
podcodex vectorize "/path/to/Les Pieds sur terre/"
podcodex vectorize "/path/to/Transfert/"
```

### 2. Copy SQLite DBs to VPS

```bash
scp "/path/to/Les Pieds sur terre/vectors.db" vps:/path/to/deploy/data/pieds_sur_terre.db
scp "/path/to/Transfert/vectors.db" vps:/path/to/deploy/data/transfert.db
```

### 3. Sync to Qdrant on VPS

```bash
docker compose run --rm bot podcodex sync --db /app/data/pieds_sur_terre.db
docker compose run --rm bot podcodex sync --db /app/data/transfert.db
```

### 4. Generate password hashes

```bash
podcodex-bot --hash-password
# Enter password: ****
# sha256:a1b2c3d4...
```

### 5. Create shows.toml

```toml
[shows.les_pieds_sur_terre]
name = "Les Pieds sur terre"
password_hash = "sha256:..."

[shows.transfert]
name = "Transfert"
password_hash = "sha256:..."
```

See `deploy/shows.example.toml` for a full example.

### 6. Update docker-compose.yml

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.17.0
    ports:
      - "127.0.0.1:6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    deploy:
      resources:
        limits:
          memory: 512M
    restart: unless-stopped
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "3"

  bot:
    build:
      context: ..
      dockerfile: deploy/Dockerfile
    depends_on:
      qdrant:
        condition: service_started
    env_file:
      - .env.production
    environment:
      - QDRANT_URL=http://qdrant:6333
    command: ["podcodex-bot", "--shows-config", "/app/shows.toml"]
    volumes:
      - ./data:/app/data
      - ./shows.toml:/app/shows.toml:ro
      - model_cache:/root/.cache
    deploy:
      resources:
        limits:
          memory: 3G
    restart: unless-stopped
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "3"

volumes:
  qdrant_data:
  model_cache:
```

### 7. Deploy

```bash
cd deploy
docker compose up -d --build
```

### 8. Unlock shows in Discord servers

In each Discord server, an admin (with `manage_guild` permission) runs `/unlock` with the show name and password. The bot verifies the hash and adds the show to that server's `allowed_shows`. Responses are ephemeral — other users see nothing.

`/lock` removes a show from the server.

## Adding a new show later

1. `podcodex vectorize` locally
2. `scp vectors.db` to VPS
3. `docker compose run --rm bot podcodex sync --db /app/data/<show>.db`
4. `podcodex-bot --hash-password` → add entry to `shows.toml`
5. `docker compose restart bot` (to reload `shows.toml`)
6. Share the password with the server admin — they run `/unlock` themselves

## Security

- **`/unlock` and `/lock` responses are ephemeral** — other users in the channel see nothing (not even that the command was used)
- **Password never stored by the bot** — only the sha256 hash is in `shows.toml`; the bot compares and discards immediately
- **Password travels over TLS** to Discord servers. Discord can see it server-side, which is acceptable for a podcast access key
- **`/unlock` requires `manage_guild` permission** — regular users cannot unlock shows
- **`/setup` blocks show management** when access control is on — `show_add`/`show_remove`/`show_clear` are rejected with a message pointing to `/unlock` and `/lock`
- **Bot owner can rotate passwords** in `shows.toml` without affecting already-unlocked servers
- **Backward compatible** — without `--shows-config`, the bot behaves exactly as before (no access control, all shows visible)
- **Future option:** DM-based `/unlock` for stronger isolation if needed

## Scaling

| Shows  | RAM    | Notes                                                  |
| ------ | ------ | ------------------------------------------------------ |
| 1      | ~3 GB  | Current setup, no `--shows-config` needed              |
| 2-20   | ~3 GB  | Single bot + `shows.toml` (this guide)                 |
| 20-100 | ~4 GB  | Add a shared embedding service (TEI or FastAPI)        |
| 100+   | ~25 GB | One bot per show + shared embedding service (see below)|

The bottleneck at scale is the embedding model: BGE-M3 uses ~2.5 GB RAM. One bot = one model load = fine. If you ever need separate bot processes per show (separate tokens for full process-level isolation), extract the embedding model into a shared HTTP service so each bot is only ~200 MB.

## Bot commands reference

| Command                          | Who         | Description                                |
| -------------------------------- | ----------- | ------------------------------------------ |
| `/unlock show password`          | Admin       | Unlock a show for this server              |
| `/lock show`                     | Admin       | Remove a show from this server             |
| `/setup`                         | Admin       | View/change model, top-k, source, compact  |
| `/search question [show] [...]`  | Everyone    | Hybrid keyword + semantic search           |
| `/exact query [show] [...]`      | Everyone    | Literal substring match                    |
| `/random [show] [...]`           | Everyone    | Random quote                               |
| `/stats [show]`                  | Everyone    | Index overview                             |
| `/episodes show`                 | Everyone    | List episodes for a show                   |
| `/help`                          | Everyone    | Show available commands                    |

All user-facing commands are scoped to `allowed_shows` — users can only see and search shows that have been unlocked in their server.
