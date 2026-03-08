"""
podcodex.bot.bot — Discord bot for podcast transcript search.

Usage:
    podcodex-bot [--model bge-m3] [--top-k 5] [--qdrant-url URL]
                 [--server-config server_config.json]

Slash commands:
    /search <question> [show] [alpha=0.5] [model] [top_k]
    /exact  <question> [show] [model] [top_k]
    /setup  [model] [top_k]  — admin: configure per-server defaults

Per-server settings are stored in a JSON file on disk.
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
from podcodex.rag.defaults import ALPHA, DEFAULT_MODEL, MODELS, TOP_K
from podcodex.rag.retriever import Retriever
from podcodex.rag.store import QdrantStore, collection_name


# ──────────────────────────────────────────────
# Per-server settings
# ──────────────────────────────────────────────


@dataclass
class ServerSettings:
    model: str = DEFAULT_MODEL
    top_k: int = TOP_K


# ──────────────────────────────────────────────
# Model choices (generated from registry — never hardcode here)
# ──────────────────────────────────────────────

_MODEL_CHOICES = [
    app_commands.Choice(name=spec.description, value=spec.key)
    for spec in MODELS.values()
]


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

_MAX_CONTEXT_N = 8
_MAX_CHARS = 1900


def _format_context(
    neighbors: list[dict], start: float, n: int, show: str, episode: str
) -> tuple[str, bool]:
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
        has_more = False

    return content, has_more


# ──────────────────────────────────────────────
# Context UI
# ──────────────────────────────────────────────


class _ExpandButton(discord.ui.Button):
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
    def __init__(self, collection: str, episode: str, show: str, start: float):
        super().__init__(timeout=300)
        self.add_item(_ExpandButton(collection, episode, show, start))


class _ExpandMoreButton(discord.ui.Button):
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
        self._retrievers: dict[str, Retriever] = {}
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
        return ServerSettings(model=self.config.model, top_k=self.config.top_k)

    def _effective_settings(
        self, server_id: int | None, model: str = "", top_k: int = 0
    ) -> ServerSettings:
        base = self._server_settings(server_id)
        return ServerSettings(
            model=model or base.model,
            top_k=top_k or base.top_k,
        )

    # ── Lazy-loaded heavy objects ──────────────

    @property
    def store(self) -> QdrantStore:
        if self._store is None:
            self._store = QdrantStore(url=self.config.qdrant_url)
        return self._store

    def retriever(self, model: str) -> Retriever:
        if model not in self._retrievers:
            self._retrievers[model] = Retriever(model=model, store=self.store)
        return self._retrievers[model]

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
            model="Embedding model — overrides server default for this query",
            top_k="Number of results (overrides server default)",
        )
        @app_commands.choices(model=_MODEL_CHOICES)
        async def search(
            interaction: discord.Interaction,
            question: str,
            show: str = "",
            alpha: app_commands.Range[float, 0.0, 1.0] = ALPHA,
            model: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
        ) -> None:
            await self._handle_search(
                interaction, question, show or None, alpha, model, top_k
            )

        search.autocomplete("show")(self._show_autocomplete)

        # /exact
        @self.tree.command(
            name="exact", description="Keyword (BM25) search in podcast transcripts"
        )
        @app_commands.describe(
            question="Keywords to search for",
            show="Restrict to a specific show (leave empty for all)",
            model="Embedding model — overrides server default for this query",
            top_k="Number of results (overrides server default)",
        )
        @app_commands.choices(model=_MODEL_CHOICES)
        async def exact(
            interaction: discord.Interaction,
            question: str,
            show: str = "",
            model: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
        ) -> None:
            await self._handle_exact(interaction, question, show or None, model, top_k)

        exact.autocomplete("show")(self._show_autocomplete)

        # /setup  (admin only)
        @self.tree.command(
            name="setup", description="Configure bot defaults for this server"
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            model="Default embedding model",
            top_k="Number of results returned per query",
        )
        @app_commands.choices(model=_MODEL_CHOICES)
        async def setup(
            interaction: discord.Interaction,
            model: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
        ) -> None:
            await self._handle_setup(interaction, model or None, top_k or None)

    # ── Autocomplete ───────────────────────────

    async def _show_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        model = (
            getattr(interaction.namespace, "model", "")
            or self._server_settings(interaction.guild_id).model
        )
        loop = asyncio.get_event_loop()
        collections = await loop.run_in_executor(
            None, lambda: self.store.list_collections(model=model)
        )
        suffix = f"__{model}"
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
        model: str,
        top_k: int,
    ) -> None:
        settings = self._effective_settings(interaction.guild_id, model, top_k)
        label = f"α={alpha:.2f}  •  {MODELS[settings.model].label}"
        await self._run_query(
            interaction, question, show, settings, alpha=alpha, label=label
        )

    # ── /exact handler ─────────────────────────

    async def _handle_exact(
        self,
        interaction: discord.Interaction,
        question: str,
        show: str | None,
        model: str,
        top_k: int,
    ) -> None:
        settings = self._effective_settings(interaction.guild_id, model, top_k)
        label = f"exact / BM25  •  {MODELS[settings.model].label}"
        await self._run_query(
            interaction, question, show, settings, alpha=0.0, label=label
        )

    # ── /setup handler ─────────────────────────

    async def _handle_setup(
        self,
        interaction: discord.Interaction,
        model: str | None,
        top_k: int | None,
    ) -> None:
        guild_id = interaction.guild_id
        current = self._server_settings(guild_id)

        if model is None and top_k is None:
            await interaction.response.send_message(
                f"**Current settings for this server**\n"
                f"Model: `{current.model}`\n"
                f"Top-k: `{current.top_k}`",
                ephemeral=True,
            )
            return

        updated = ServerSettings(
            model=model or current.model,
            top_k=top_k or current.top_k,
        )
        self._server_cfg[guild_id] = updated
        self._save_server_config()
        logger.info(f"Guild {guild_id} config updated: {updated}")

        await interaction.response.send_message(
            f"✅ Settings updated\nModel: `{updated.model}`\nTop-k: `{updated.top_k}`",
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

    def _search(
        self,
        question: str,
        show: str | None,
        settings: ServerSettings,
        alpha: float,
    ) -> list[tuple[dict, str]]:
        model = settings.model
        collections = (
            [collection_name(show, model)]
            if show
            else self.store.list_collections(model=model)
        )

        if not collections:
            logger.warning(f"No collections found for model '{model}'")
            return []

        ret = self.retriever(model)
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
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Default embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        default=TOP_K,
        type=int,
        help=f"Default results per query (default: {TOP_K})",
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
        model=args.model,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
    )
    bot = PodCodexBot(config, server_config_path=Path(args.server_config))
    logger.info(f"Starting PodCodex bot (model={config.model}, top_k={config.top_k})…")
    bot.run(config.token, log_handler=None)


if __name__ == "__main__":
    main()
