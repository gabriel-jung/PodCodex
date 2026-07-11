# Discord update announcements — design

Date: 2026-07-11
Status: approved (design), pending spec review

## Goal

The Discord bot proactively announces, in a per-server channel:

1. **New episodes** — when episodes are added to the index (the primary want).
2. **Bot version change** — when the running bot's version differs from the last announced one.

## Hard constraint: the bot is index-only

The bot process sees only the LanceDB index (rsynced from the desktop). It has no
`config.json`, no show folders, no push signal from the pipeline. Every fact it
announces must be derivable from the index it already reads. Consequences:

- New-episode detection is **poll + diff**, not push. The bot already tracks
  `index_mtime()` to invalidate ACL state; the announcer reuses that signal.
- The producing app/pipeline version is **not** in the index, so "version" means
  the **bot's own** `podcodex.__version__`. Stamping a pipeline version into the
  index is explicitly out of scope (see Non-goals).

## Data provenance (no invented info)

Every announced field maps to real data — a user choice made in the app or the
episode's own data/metadata — or it is **omitted**. Nothing is synthesized or
guessed. There is no generative step anywhere in this feature (the bot is
retrieval-only), so this is enforced structurally:

| Announced field | Source | If absent |
|---|---|---|
| Show name | index `show` (chosen in the app) | (always present) |
| Episode title | `episode_title` from `.episode_meta.json` (RSS/YouTube title) | fall back to the real stem, humanized — never a made-up title |
| Date (`month`) | `pub_date` metadata | omit the date entirely |
| Artwork thumbnail | `artwork_url` metadata | no thumbnail |
| "N new episodes" | exact diff count | (always real) |
| Bot version | `podcodex.__version__` | (always real) |

Rule for implementation: never fabricate a date, title, count, or link to fill a
gap; leave it out. This mirrors the project-wide stance that pipeline output
reflects only user choices and episode data/metadata, never inference.

## Architecture

Two new pieces plus wiring in `bot.py`.

### `bot/announce.py` (new module)

- **`AnnounceStore(db_path)`** — durable state in its own SQLite file
  (`announce_state.db`, next to `search_cache.db`), isolated from the search
  cache. Schema:
  - `baselined(collection TEXT PRIMARY KEY)` — collections whose back-catalogue
    has been recorded silently.
  - `seen_episodes(collection TEXT, stem TEXT, PRIMARY KEY(collection, stem))`.
  - `meta(key TEXT PRIMARY KEY, value TEXT)` — holds `announced_version`.
  - Method `observe(collection, current: set[str]) -> list[str]`:
    - collection not in `baselined` → insert all `current` into `seen_episodes`,
      mark baselined, **return `[]`** (first-run silence: never dump the back
      catalogue).
    - else → `new = current - seen`; insert `new`; return sorted `new`.
  - `get_meta(key)` / `set_meta(key, value)`.
  - Thread-safe like `SearchCacheStore` (single lock; sub-ms ops).

- **Pure helpers** (no Discord I/O, unit-testable):
  - `build_new_episodes_embed(show, episodes) -> discord.Embed` — grouped card:
    title `📣 N new episode(s) — {show}`, show artwork as thumbnail (from the
    first episode carrying `artwork_url`), body = `• {title} · {month}` lines
    (newest first, capped with a `+K more` tail if very large).
  - `build_version_embed(version) -> discord.Embed` — `🔖 PodCodex bot v{version}`.

### `bot.py` wiring

- **`ServerSettings.announce_channel_id: int = 0`** — 0 = disabled (opt-in).
  Persisted in `server_config.json` (existing mechanism).
- **`BotConfig.announce_interval_minutes: int = 10`** — CLI `--announce-interval`.
- **`AnnounceStore`** constructed in `__init__` (path = `server_config_path.parent
  / "announce_state.db"`).
- **`/announcements` command** (admin, `default_permissions(manage_guild=True)`):
  - `channel: discord.TextChannel = None`, `off: bool` choice.
  - `off:true` → clear (`announce_channel_id = 0`). `channel:` set → store its id.
  - no args → report current channel (or "disabled").
  - Writes `ServerSettings` + `_save_server_config()`.
- **`@tasks.loop` announcer** — interval set from config via `change_interval`
  before `.start()` in `setup_hook` (after the index is opened). Each tick:
  1. `index_mtime()` vs the announcer's **own** watermark
     (`_announce_mtime_seen`, separate from `_refresh_if_stale`'s
     `_index_mtime_seen` — otherwise a user command's refresh advances the shared
     watermark and the loop never sees the change) → unchanged → return.
  2. `reconnect()` + `_reload_shows()` (same refresh path as `_refresh_if_stale`).
  3. For each collection in `get_all_collection_info()`: `current =
     set(list_episodes(col))`; `new = AnnounceStore.observe(col, current)`.
  4. For collections with new stems, fetch display rows (`get_episode_stats`) and
     group by show → `{show: [episode rows]}`.
  5. For each guild with `announce_channel_id`: compute that guild's accessible
     collections (`_resolve_show_collections(ALL, guild_settings)`); post one
     grouped embed per accessible show that has new episodes.
- **Version announce** in `on_ready`:
  - `stored = AnnounceStore.get_meta("announced_version")`.
  - `stored is None` → set silently (baseline; no announce on first ever run).
  - `stored != __version__` → post `build_version_embed` to each configured
    channel, then `set_meta`.

## Data flow

```
rsync new index → mtime rises → loop tick detects → observe() diffs per collection
  → new stems → group by show → per guild: access-filter → post grouped embed
```

State is global (per collection); access filtering happens at post time per guild,
so a locked show never leaks to a server that has not unlocked it. A server that
unlocks a show later gets only its *future* episodes (past ones are already in
`seen_episodes`) — acceptable.

## Error handling

- Send failures (channel deleted, missing permissions, HTTP error) are logged and
  swallowed per channel; the loop and other channels continue.
- A tick that throws is caught so the loop never dies; next tick retries.
- `observe()` persists new stems **only after** they are computed, but state is
  written regardless of whether any channel is configured — so configuring a
  channel later never replays the back catalogue.
- Note: state advances even if a post fails, so a hard send failure means that
  batch is not retried. Acceptable for v1 (announcements are best-effort, not a
  guaranteed feed).

## Testing

Unit (pure, no Discord gateway):
- `AnnounceStore`: first-run baseline returns `[]` and records all; subsequent
  `observe` returns only genuinely new stems; re-observing the same set returns
  `[]`; version meta get/set + None baseline.
- Access filtering: given new-episodes-per-collection and two guild settings (one
  with a locked show unlocked, one without), the accessible subset is correct.
- `build_new_episodes_embed`: grouped title count, newest-first ordering,
  thumbnail from artwork, `+K more` tail.
- Version-change detection: None → silent; unchanged → no announce; changed →
  announce once.

The `tasks.loop` timer itself is not unit-tested; its body is factored into a
plain async method exercised directly (mirrors the existing FakeBot simulation
pattern in `test_bot_simulation.py`).

## Non-goals (v1)

- No per-episode rich cards for batches (grouped-per-show only, per decision).
- No index/pipeline version stamp — bot version only.
- No DMs, no editing/deleting past announcements, no retry queue for failed sends.
- No custom message templates or per-show announce toggles.

## Open decisions (resolved)

- Announce channel config: dedicated `/announcements` command with a native
  channel picker (not folded into `/setup`).
- Poll cadence: 10 minutes default, `--announce-interval` to override.
- Multiple episodes at once: one grouped message per show.
- Scope: new episodes + bot version (on change).
