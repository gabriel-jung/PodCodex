"""Typed models for the RAG read boundary.

A ``Hit`` is one chunk read back from a LanceDB index, optionally enriched
by retrieval (score, exact-match metadata). ``index_store._row_to_chunk``
is the single place hits are born; everything downstream (retriever,
search_service, bot, MCP, API) consumes these models.

The write path (chunker → embedder → ``save_chunks``) stays dict-based:
``save_chunks`` owns coalescing/normalization and runs thousands of times
per episode. Write-side schema drift is guarded by the chunker conformance
test in ``tests/test_hit_model.py``: any new chunker key must be declared
here or that test fails.

Both models use ``extra="allow"``: old on-disk indexes can carry meta keys
this version no longer writes, and they must keep loading. Unknown keys
land in ``model_extra``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from podcodex.core._utils import normalize_pub_date, resolve_episode_title


class SpeakerTurn(BaseModel):
    """One speaker turn inside a multi-turn chunk (``Hit.speakers``)."""

    model_config = ConfigDict(extra="allow")

    # Empty, not a sentinel: the chunker writes "UNKNOWN" explicitly, and the
    # display layer (merge_display_turns / display_speaker) owns how an unnamed
    # speaker is rendered. A default sentinel here would ship raw to users.
    speaker: str = ""
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    # Character offsets into the chunk text, chunker bookkeeping. Absent on
    # speaker_chunks output and on some legacy indexes.
    start_char: int | None = None
    end_char: int | None = None


class Hit(BaseModel):
    """One chunk read back from the index, plus optional retrieval fields."""

    model_config = ConfigDict(extra="allow")

    # Fixed LanceDB columns (index_store._chunk_schema).
    chunk_index: int = -1
    show: str = ""
    episode: str = ""
    source: str = ""
    pub_date: str = ""
    dominant_speaker: str = ""
    start: float = 0.0
    end: float = 0.0
    text: str = ""

    # Meta extras (chunker._meta_fields), packed into the meta JSON column
    # by save_chunks and re-inflated by _row_to_chunk. All optional.
    episode_title: str = ""
    episode_number: int | None = None
    broadcast_number: int | None = None
    description: str = ""
    artwork_url: str = ""
    audio_url: str = ""
    youtube_id: str = ""
    # Chunker only writes timed=False (subtitle imports without timing);
    # absence means timed. Consumers always read with a True default.
    timed: bool = True
    speakers: list[SpeakerTurn] | None = None
    token_count: int | None = None
    # Pre-normalization indexes stored the raw RSS date under this key.
    rss_pub_date: str = ""

    # Retrieval enrichment (search_vector / search_fts / search_literal /
    # Retriever.exact). None or False on plain loads.
    score: float | None = None
    match_text: str | None = None
    accent_match: bool = False
    fuzzy_match: bool = False

    # Set only by the random-path turn-flatten (Retriever.random), which
    # narrows a multi-turn chunk to a single turn. None on stored rows:
    # save_chunks folds the chunker's `speaker` key into dominant_speaker.
    speaker: str | None = None

    def to_chunk_dict(self) -> dict:
        """Write-shape dict for feeding back into ``save_chunks`` (indexer
        cache reuse across models). Retrieval fields are always dropped;
        ``exclude_defaults`` drops absent extras so stored meta doesn't
        accumulate junk. Legacy ``model_extra`` keys survive the roundtrip."""
        out = self.model_dump(
            exclude_defaults=True,
            exclude={
                "chunk_index",
                "score",
                "match_text",
                "accent_match",
                "fuzzy_match",
            },
        )
        # embedder.encode_passages subscripts chunk["text"] and save_chunks keys
        # off episode/show, so these must survive exclude_defaults even when the
        # value happens to equal the field default.
        out["text"] = self.text
        out["episode"] = self.episode
        out["show"] = self.show
        return out

    @property
    def display_title(self) -> str:
        """Human-readable episode title: RSS title, else humanized stem."""
        return resolve_episode_title(self.episode_title, self.episode)

    @property
    def speaker_label(self) -> str:
        """Raw speaker name: turn speaker when flattened, else dominant."""
        return self.speaker or self.dominant_speaker

    @property
    def effective_pub_date(self) -> str:
        """Publication date as YYYY-MM-DD, tolerating legacy indexes that
        only carry the raw RSS form under ``rss_pub_date``."""
        if self.pub_date:
            return self.pub_date
        return normalize_pub_date(self.rss_pub_date) or ""
