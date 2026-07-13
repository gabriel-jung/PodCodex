"""Show-to-collection resolution shared by every command and the announcer."""

from __future__ import annotations


from podcodex.bot.access import ResolvedShows
from podcodex.bot.config import ServerSettings
from podcodex.rag.defaults import (
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
)
from podcodex.rag.search_service import (
    SearchCollection,
    load_show_rag_prefs,
    resolve_collections,
)


class ResolutionMixin:
    """Collection-resolution methods mixed into PodCodexBot (bot.py).

    Expects on self: ``_model_label``, ``_show_allowed``.
    """

    def _empty_collections_message(
        self,
        col_info: dict[str, dict],
        settings: ServerSettings,
        shows: ResolvedShows | None = None,
    ) -> str:
        """Explain to the user why no collections matched.

        Distinguishes: empty index, wrong model, locked/no-unlock, missing show.
        """
        if not col_info:
            return (
                "Nothing has been indexed yet. "
                "Add a show in the desktop app and run the **Index** step."
            )

        model_label = self._model_label(settings.model)
        same_model = {
            info["model"]
            for info in col_info.values()
            if info["chunker"] == settings.chunker
        }

        if settings.model not in same_model:
            others = sorted(m for m in same_model if m)
            hint = (
                f"Available models: {', '.join(others)}. "
                f"Switch with `/setup model:{others[0]}` or pass `model:` to this command."
                if others
                else "No other models available either — index something first."
            )
            return (
                f"No shows are indexed with the **{model_label}** embedding model. "
                f"{hint}"
            )

        if shows and shows.is_locked:
            return (
                "No shows are unlocked for this Discord server. "
                "An admin can unlock one with `/unlock password:****`."
            )

        if shows and shows.is_specific:
            missing = ", ".join(f"**{s}**" for s in shows.shows)
            return f"{missing} is not indexed with the **{model_label}** model on this server."

        return "No shows are available to search here."

    def _resolve_show_collections(
        self,
        shows: ResolvedShows,
        settings: ServerSettings,
        col_info: dict[str, dict],
        *,
        explicit: tuple[str, str] | None = None,
    ) -> list[SearchCollection]:
        """One :class:`SearchCollection` per accessible show.

        The single collection-resolution path for /search, /exact, /random and
        the stats commands, delegating the picking to the shared
        :func:`resolve_collections`. Precedence per show: ``explicit`` (a user
        model+chunker typed on an ``-advanced`` command), the show's
        ``show.toml`` RAG prefs, this server's default model+chunker, the
        global default (``DEFAULT_MODEL``/``DEFAULT_CHUNKING``), then the
        first collection by name so a show indexed only under a non-default
        model stays reachable. ``is_locked`` yields ``[]`` (no preview leaks);
        locked-but-unlocked shows pass the access filter.
        """
        if shows.is_locked:
            return []
        wanted = list(shows.shows) if shows.is_specific else None
        return [
            c
            for c in resolve_collections(
                col_info,
                shows=wanted,
                show_prefs=load_show_rag_prefs(),
                override=explicit,
                default=(
                    settings.model or DEFAULT_MODEL,
                    settings.chunker or DEFAULT_CHUNKING,
                ),
            )
            if self._show_allowed(c.show, settings)
        ]
