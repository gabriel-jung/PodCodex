"""Show-to-collection resolution shared by every command and the announcer."""

from __future__ import annotations


from podcodex.bot.access import ResolvedShows
from podcodex.bot.config import ServerSettings
from podcodex.rag.defaults import (
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    MODELS,
)
from podcodex.rag.search_service import (
    SearchCollection,
    load_show_rag_prefs,
    resolve_collections,
)


def explicit_combo(
    model: str, chunker: str, settings: ServerSettings
) -> tuple[str, str] | None:
    """The (model, chunker) a user typed on a command, or None if neither.

    Kept apart from ``settings`` rather than merged into it, so a typo'd
    combo still falls back through show.toml prefs and the server default
    instead of collapsing straight to first-by-name. The command and its
    autocompletes both build it here, so suggestions match what runs.
    """
    if not (model or chunker):
        return None
    return (model or settings.model, chunker or settings.chunker)


class ResolutionMixin:
    """Collection-resolution methods mixed into PodCodexBot (bot.py).

    Expects on self: ``config``, ``_model_label``, ``_show_allowed``.
    """

    def _loadable_models(self) -> set[str]:
        """Models the bot will load: ``--models`` when given, else ``--model``
        plus any model an allowed guild chose with /setup (``model_set``), so
        a later smaller ``--model`` does not take that guild's search away.
        Settings saved before the marker existed count as no choice: they
        cannot tell a pick from a model an older ``--model`` stamped, and
        loading a stamped large model is the memory blowup this list exists
        to prevent (the bot logs those guilds at startup).
        """
        if self.config.models:
            return set(self.config.models)
        chosen = {
            s.model
            for gid, s in list(self._server_cfg.items())
            if s.model_set and s.model in MODELS and self.guild_allowed(gid)
        }
        return {self.config.model, *chosen}

    def _loadable(self, col_info: dict[str, dict]) -> dict[str, dict]:
        """The collections whose model this bot will load."""
        allowed = self._loadable_models()
        return {n: i for n, i in col_info.items() if i.get("model") in allowed}

    def _empty_collections_message(
        self,
        col_info: dict[str, dict],
        shows: ResolvedShows | None = None,
        *,
        embeds: bool = False,
    ) -> str:
        """Explain to the user why no collections matched.

        Distinguishes: empty index, locked show, unknown show, a show (or the
        whole index) indexed only under models this bot does not load, and
        nothing visible here. Resolution falls back across models, so the
        guild's default model is never the reason by itself. ``embeds``: the
        command encodes a query, so the bot's model list applies.
        """
        if not col_info:
            return (
                "Nothing has been indexed yet. "
                "Add a show in the desktop app and run the **Index** step."
            )

        if shows and shows.is_locked:
            # About the requested show only. The old wording stated
            # server-wide state the data contradicts, so an admin who had
            # already unlocked two other shows was told none were unlocked.
            # It also says nothing about which other shows exist.
            return (
                "That show is not unlocked on this Discord server. "
                "An admin can unlock it with `/unlock password:****`."
            )

        loadable = self._loadable(col_info) if embeds else col_info

        def shows_in(info: dict[str, dict]) -> set[str]:
            # The label resolve_collections matches on, name fallback included.
            return {(i.get("show") or n).strip().lower() for n, i in info.items()}

        if shows and shows.is_specific:
            wanted = {s.strip().lower() for s in shows.shows}
            named = ", ".join(f"**{s}**" for s in shows.shows)
            if not wanted & shows_in(col_info):
                return f"No show called {named} is indexed."
            if not wanted & shows_in(loadable):
                return f"{named} is indexed with a model this bot does not load."
            return f"{named} is not available on this server."

        if not loadable:
            return "Nothing here is indexed with a model this bot loads."
        return "No shows are available to search here."

    def _resolve_show_collections(
        self,
        shows: ResolvedShows,
        settings: ServerSettings,
        col_info: dict[str, dict],
        *,
        explicit: tuple[str, str] | None = None,
        embeds: bool = False,
    ) -> list[SearchCollection]:
        """One :class:`SearchCollection` per accessible show.

        The single collection-resolution path for /search, /exact, /random and
        the stats commands, delegating the picking to the shared
        :func:`resolve_collections`. Precedence per show: ``explicit`` (a user
        model+chunker typed on an ``-advanced`` command), the show's
        ``show.toml`` RAG prefs, this server's default model+chunker, the
        global default (``DEFAULT_MODEL``/``DEFAULT_CHUNKING``), then the
        first collection by name so a show indexed only under a non-default
        model stays reachable. ``embeds`` (the command encodes a query: only
        /search) first drops collections under models this bot does not load;
        literal search, random and stats read the index without a model.
        ``is_locked`` yields ``[]`` (no preview leaks);
        locked-but-unlocked shows pass the access filter.
        """
        if shows.is_locked:
            return []
        wanted = list(shows.shows) if shows.is_specific else None
        picked = resolve_collections(
            self._loadable(col_info) if embeds else col_info,
            shows=wanted,
            show_prefs=load_show_rag_prefs(),
            override=explicit,
            default=(
                settings.model or DEFAULT_MODEL,
                settings.chunker or DEFAULT_CHUNKING,
            ),
        )
        if not self._locked_show_ids:
            return picked  # nothing protected: no lookups on the event loop
        return [c for c in picked if self._row_allowed(c.show_id, c.show, settings)]
