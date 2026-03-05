"""
podcodex.bot.bot — Discord bot for podcast transcript search.

Usage:
    podcodex-bot [--strategy bge_speaker] [--top-k 5] [--qdrant-url URL]
                 [--server-config server_config.json]

Slash commands:
    /search <question> [show] [alpha=0.5]  — vector/hybrid search
    /exact  <question> [show]              — BM25 keyword search
    /setup  [strategy] [top_k]            — admin: configure per-server defaults

Per-server settings (strategy, top_k) are stored in a JSON file on disk and
override the global CLI defaults for that server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import discord
from discord import app_commands
from loguru import logger

from podcodex.bot.config import BotConfig
from podcodex.rag.retriever import Retriever
from podcodex.rag.store import QdrantStore


# ──────────────────────────────────────────────
# Per-server settings
# ──────────────────────────────────────────────


@dataclass
class ServerSettings:
    strategy: str = "bge_speaker"
    top_k: int = 5


# ──────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _speaker(chunk: dict) -> str:
    return chunk.get("speaker") or chunk.get("dominant_speaker") or "Unknown"


def _score_bar(score: float, width: int = 8) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


# ──────────────────────────────────────────────
# Context formatting
# ──────────────────────────────────────────────

_MAX_CONTEXT_N = 8  # max chunks each side
_MAX_CHARS = 1900  # Discord message limit with some headroom


def _format_context(
    neighbors: list[dict], start: float, n: int, show: str, episode: str
) -> tuple[str, bool]:
    """
    Format surrounding chunks as a Discord message.

    Returns:
        (content, has_more) — has_more is True when there are chunks
        beyond the current window that a further expansion would reveal.
    """
    pos = next(
        (i for i, c in enumerate(neighbors) if abs(c.get("start", -1) - start) < 0.1),
        None,
    )
    if pos is None:
        return "Could not locate this chunk in the episode.", False

    lo = max(0, pos - n)
    hi = min(len(neighbors), pos + n + 1)
    has_more = (lo > 0 or hi < len(neighbors)) and n < _MAX_CONTEXT_N

    before = neighbors[lo:pos]
    current = neighbors[pos]
    after = neighbors[pos + 1 : hi]

    header = f"**{show} — {episode}**"
    if show == "" and episode == "":
        header = "*Context*"
    lines = [header + f"  ·  ±{n} turns\n"]

    for c in before:
        lines.append(f"*{_speaker(c)}* · {_fmt_time(c.get('start', 0))}\n{c['text']}")
    lines.append(
        f"**▶ {_speaker(current)}** · {_fmt_time(current.get('start', 0))}\n"
        f"**{current['text']}**"
    )
    for c in after:
        lines.append(f"*{_speaker(c)}* · {_fmt_time(c.get('start', 0))}\n{c['text']}")

    content = "\n\n".join(lines)
    if len(content) > _MAX_CHARS:
        content = content[:_MAX_CHARS] + "\n\n*…(truncated)*"
        has_more = False  # no point expanding if we're already hitting the limit

    return content, has_more


# ──────────────────────────────────────────────
# Context UI — initial button on result embed
# ──────────────────────────────────────────────


class _ExpandButton(discord.ui.Button):
    """Button on the result embed that opens the context window (±2 turns)."""

    def __init__(self, collection: str, episode: str, show: str, start: float):
        super().__init__(label="Show context ↕", style=discord.ButtonStyle.secondary)
        self._collection = collection
        self._episode = episode
        self._show = show
        self._start = start

    async def callback(self, interaction: discord.Interaction) -> None:
        bot: PodCodexBot = interaction.client  # type: ignore[assignment]
        loop = asyncio.get_event_loop()
        neighbors = await loop.run_in_executor(
            None,
            lambda: bot.store.fetch_episode_chunks(self._collection, self._episode),
        )
        content, has_more = _format_context(
            neighbors, self._start, 2, self._show, self._episode
        )
        view = _ContextView(
            self._collection,
            self._episode,
            self._show,
            self._start,
            neighbors,
            n=2,
            has_more=has_more,
        )
        await interaction.response.send_message(content, view=view, ephemeral=True)


class _ExpandView(discord.ui.View):
    """View attached to each result embed (holds the initial Show context button)."""

    def __init__(self, collection: str, episode: str, show: str, start: float):
        super().__init__(timeout=300)
        self.add_item(_ExpandButton(collection, episode, show, start))


# ──────────────────────────────────────────────
# Context UI — expandable context message
# ──────────────────────────────────────────────


class _ExpandMoreButton(discord.ui.Button):
    """Button inside the context window that widens the visible window."""

    def __init__(
        self,
        collection: str,
        episode: str,
        show: str,
        start: float,
        neighbors: list[dict],
        n: int,
    ):
        super().__init__(
            label=f"Show more ↕  (±{n})", style=discord.ButtonStyle.secondary
        )
        self._collection = collection
        self._episode = episode
        self._show = show
        self._start = start
        self._neighbors = neighbors
        self._n = n

    async def callback(self, interaction: discord.Interaction) -> None:
        content, has_more = _format_context(
            self._neighbors, self._start, self._n, self._show, self._episode
        )
        view = _ContextView(
            self._collection,
            self._episode,
            self._show,
            self._start,
            self._neighbors,
            n=self._n,
            has_more=has_more,
        )
        await interaction.response.edit_message(content=content, view=view)


class _ContextView(discord.ui.View):
    """View for the ephemeral context message — offers further expansion."""

    def __init__(
        self,
        collection: str,
        episode: str,
        show: str,
        start: float,
        neighbors: list[dict],
        n: int,
        has_more: bool,
    ):
        super().__init__(timeout=300)
        if has_more:
            self.add_item(
                _ExpandMoreButton(collection, episode, show, start, neighbors, n + 2)
            )


# ──────────────────────────────────────────────
# Result embed
# ──────────────────────────────────────────────


def _result_embed(
    chunk: dict, rank: int, collection: str, label: str
) -> tuple[discord.Embed, _ExpandView]:
    show = chunk.get("show", "")
    episode = chunk.get("episode", "")
    speaker = _speaker(chunk)
    start = chunk.get("start", 0.0)
    end = chunk.get("end", 0.0)
    text = chunk.get("text", "")
    score = chunk.get("score", 0.0)

    embed = discord.Embed(description=f'*"{text}"*', color=discord.Color.blurple())
    if show:
        embed.set_author(name=f"🎙 {show}")
    embed.title = episode or "(untitled)"
    embed.add_field(name="Speaker", value=speaker, inline=True)
    embed.add_field(
        name="Timestamp", value=f"{_fmt_time(start)} → {_fmt_time(end)}", inline=True
    )
    embed.add_field(
        name="Relevance", value=f"{_score_bar(score)} {score:.0%}", inline=True
    )
    embed.set_footer(text=f"#{rank}  •  {label}")

    return embed, _ExpandView(collection, episode, show, start)


# ──────────────────────────────────────────────
# Strategy choices (shared across commands)
# ──────────────────────────────────────────────

_STRATEGY_CHOICES = [
    app_commands.Choice(name="BGE speaker turns — hybrid", value="bge_speaker"),
    app_commands.Choice(name="BGE semantic chunks — hybrid", value="bge_semantic"),
    app_commands.Choice(name="E5 semantic — dense", value="e5_semantic"),
    app_commands.Choice(name="Pplx context — dense", value="pplx_context"),
]


# ──────────────────────────────────────────────
# Bot
# ──────────────────────────────────────────────


class PodCodexBot(discord.Client):
    def __init__(self, config: BotConfig, server_config_path: Path):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.config = config
        self.server_config_path = server_config_path
        self.tree = app_commands.CommandTree(self)

        self._store: QdrantStore | None = None
        self._retrievers: dict[str, Retriever] = {}  # strategy → Retriever
        self._server_cfg: dict[int, ServerSettings] = self._load_server_config()

        self._register_commands()

    # ── Guild config persistence ───────────────

    def _load_server_config(self) -> dict[int, ServerSettings]:
        if not self.server_config_path.exists():
            return {}
        raw = json.loads(self.server_config_path.read_text())
        return {int(sid): ServerSettings(**v) for sid, v in raw.items()}

    def _save_server_config(self) -> None:
        self.server_config_path.write_text(
            json.dumps(
                {str(k): asdict(v) for k, v in self._server_cfg.items()}, indent=2
            )
        )

    def _server_settings(self, server_id: int | None) -> ServerSettings:
        if server_id and server_id in self._server_cfg:
            return self._server_cfg[server_id]
        return ServerSettings(strategy=self.config.strategy, top_k=self.config.top_k)

    def _effective_settings(
        self, server_id: int | None, strategy: str = "", top_k: int = 0
    ) -> ServerSettings:
        """Server defaults overridden by any per-query values that were explicitly set."""
        base = self._server_settings(server_id)
        return ServerSettings(
            strategy=strategy or base.strategy,
            top_k=top_k or base.top_k,
        )

    # ── Lazy-loaded heavy objects ──────────────

    @property
    def store(self) -> QdrantStore:
        if self._store is None:
            self._store = QdrantStore(url=self.config.qdrant_url)
        return self._store

    def retriever(self, strategy: str) -> Retriever:
        if strategy not in self._retrievers:
            self._retrievers[strategy] = Retriever(strategy, store=self.store)
        return self._retrievers[strategy]

    # ── Lifecycle ─────────────────────────────

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Slash commands synced")

    async def on_ready(self) -> None:
        logger.success(f"Logged in as {self.user} (id={self.user.id})")

    # ── Command registration ───────────────────

    def _register_commands(self) -> None:
        # /search
        @self.tree.command(
            name="search", description="Search podcast transcripts (vector/hybrid)"
        )
        @app_commands.describe(
            question="What are you looking for?",
            show="Restrict to a specific show (leave empty for all)",
            alpha="0 = keywords only → 1 = semantic only  (default 0.5)",
            strategy="Embedding model — overrides server default for this query",
            top_k="Number of results (overrides server default)",
        )
        @app_commands.choices(strategy=_STRATEGY_CHOICES)
        async def search(
            interaction: discord.Interaction,
            question: str,
            show: str = "",
            alpha: app_commands.Range[float, 0.0, 1.0] = 0.5,
            strategy: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
        ) -> None:
            await self._handle_search(
                interaction, question, show or None, alpha, strategy, top_k
            )

        search.autocomplete("show")(self._show_autocomplete)

        # /exact
        @self.tree.command(
            name="exact", description="Keyword search in podcast transcripts (BM25)"
        )
        @app_commands.describe(
            question="Keywords to search for",
            show="Restrict to a specific show (leave empty for all)",
            strategy="Embedding model — overrides server default for this query",
            top_k="Number of results (overrides server default)",
        )
        @app_commands.choices(strategy=_STRATEGY_CHOICES)
        async def exact(
            interaction: discord.Interaction,
            question: str,
            show: str = "",
            strategy: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
        ) -> None:
            await self._handle_exact(
                interaction, question, show or None, strategy, top_k
            )

        exact.autocomplete("show")(self._show_autocomplete)

        # /setup  (admin only)
        @self.tree.command(
            name="setup", description="Configure bot defaults for this server"
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            strategy="Embedding strategy / model to use",
            top_k="Number of results returned per query",
        )
        @app_commands.choices(strategy=_STRATEGY_CHOICES)
        async def setup(
            interaction: discord.Interaction,
            strategy: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
        ) -> None:
            await self._handle_setup(interaction, strategy or None, top_k or None)

    # ── Autocomplete ───────────────────────────

    async def _show_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        # Use strategy from the interaction namespace if the user already picked one
        strategy = (
            getattr(interaction.namespace, "strategy", "")
            or self._server_settings(interaction.guild_id).strategy
        )
        loop = asyncio.get_event_loop()
        collections = await loop.run_in_executor(
            None, lambda: self._indexed_collections(strategy)
        )
        suffix = f"__{strategy}"
        shows = [c.removesuffix(suffix) for c in collections]
        return [
            app_commands.Choice(name=s, value=s)
            for s in shows
            if current.lower() in s.lower()
        ][:25]

    # ── /search handler ────────────────────────

    async def _handle_search(
        self,
        interaction: discord.Interaction,
        question: str,
        show: str | None,
        alpha: float,
        strategy: str,
        top_k: int,
    ) -> None:
        settings = self._effective_settings(interaction.guild_id, strategy, top_k)
        label = f"α={alpha:.2f}  •  {settings.strategy}"
        await self._run_query(
            interaction, question, show, settings, alpha=alpha, label=label
        )

    # ── /exact handler ─────────────────────────

    async def _handle_exact(
        self,
        interaction: discord.Interaction,
        question: str,
        show: str | None,
        strategy: str,
        top_k: int,
    ) -> None:
        settings = self._effective_settings(interaction.guild_id, strategy, top_k)
        if not settings.strategy.startswith("bge"):
            await interaction.response.send_message(
                f"BM25 search requires a BGE strategy. "
                f"Current: `{settings.strategy}`. "
                f"Pick a BGE strategy or ask an admin to run `/setup`.",
                ephemeral=True,
            )
            return
        await self._run_query(
            interaction,
            question,
            show,
            settings,
            alpha=0.0,
            label=f"exact / BM25  •  {settings.strategy}",
        )

    # ── /setup handler ─────────────────────────

    async def _handle_setup(
        self,
        interaction: discord.Interaction,
        strategy: str | None,
        top_k: int | None,
    ) -> None:
        guild_id = interaction.guild_id
        current = self._server_settings(guild_id)

        if strategy is None and top_k is None:
            await interaction.response.send_message(
                f"**Current settings for this server**\n"
                f"Strategy: `{current.strategy}`\n"
                f"Top-k: `{current.top_k}`",
                ephemeral=True,
            )
            return

        updated = ServerSettings(
            strategy=strategy or current.strategy,
            top_k=top_k or current.top_k,
        )
        self._server_cfg[guild_id] = updated
        self._save_server_config()
        logger.info(f"Guild {guild_id} config updated: {updated}")

        await interaction.response.send_message(
            f"✅ Settings updated\n"
            f"Strategy: `{updated.strategy}`\n"
            f"Top-k: `{updated.top_k}`",
            ephemeral=True,
        )

    # ── Shared query runner ────────────────────

    async def _run_query(
        self,
        interaction: discord.Interaction,
        question: str,
        show: str | None,
        settings: ServerSettings,
        alpha: float,
        label: str,
    ) -> None:
        await interaction.response.defer()

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: self._search(question, show, settings, alpha)
            )
        except Exception as exc:
            logger.error(f"Query error: {exc}")
            await interaction.followup.send(f"Search failed: {exc}", ephemeral=True)
            return

        if not results:
            await interaction.followup.send("No results found.", ephemeral=True)
            return

        for rank, (chunk, col) in enumerate(results, 1):
            embed, view = _result_embed(chunk, rank, col, label)
            await interaction.followup.send(embed=embed, view=view)

    # ── Search logic ───────────────────────────

    def _indexed_collections(self, strategy: str) -> list[str]:
        suffix = f"__{strategy}"
        return [c for c in self.store.list_collections() if c.endswith(suffix)]

    def _search(
        self,
        question: str,
        show: str | None,
        settings: ServerSettings,
        alpha: float,
    ) -> list[tuple[dict, str]]:
        suffix = f"__{settings.strategy}"
        collections = (
            [f"{show}{suffix}"]
            if show
            else self._indexed_collections(settings.strategy)
        )

        if not collections:
            logger.warning(f"No collections found for strategy '{settings.strategy}'")
            return []

        ret = self.retriever(settings.strategy)
        all_results: list[tuple[dict, str]] = []
        for col in collections:
            hits = ret.retrieve(question, col, top_k=settings.top_k, alpha=alpha)
            all_results.extend((hit, col) for hit in hits)

        all_results.sort(key=lambda x: x[0].get("score", 0.0), reverse=True)
        return all_results[: settings.top_k]


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(prog="podcodex-bot")
    parser.add_argument(
        "--strategy",
        default="bge_speaker",
        help="Default embedding strategy (default: bge_speaker)",
    )
    parser.add_argument(
        "--top-k", default=5, type=int, help="Default results per query (default: 5)"
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL (default: QDRANT_URL env or localhost:6333)",
    )
    parser.add_argument(
        "--server-config",
        default="server_config.json",
        help="Path to per-server settings file (default: server_config.json)",
    )
    args = parser.parse_args()

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "DISCORD_TOKEN is not set. Add it to your .env file or environment."
        )

    config = BotConfig(
        token=token,
        strategy=args.strategy,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
    )
    bot = PodCodexBot(config, server_config_path=Path(args.server_config))
    logger.info(
        f"Starting PodCodex bot (strategy={config.strategy}, top_k={config.top_k})…"
    )
    bot.run(config.token, log_handler=None)


if __name__ == "__main__":
    main()
