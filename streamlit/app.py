"""
podcodex — Podcast transcription, translation and synthesis app.

Run with: streamlit run streamlit/app.py
"""

import os
import sys
import warnings

import streamlit as st
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv
from utils import normalize_path

load_dotenv()

# ── Logging setup ──
# Remove default loguru handler, reconfigure with color and configurable level.
# Use the sidebar debug toggle or set PODCODEX_DEBUG=1 env var.
logger.remove()
_LOG_LEVEL = (
    "DEBUG"
    if (st.session_state.get("debug_logging") or os.environ.get("PODCODEX_DEBUG"))
    else "INFO"
)
logger.add(sys.stderr, level=_LOG_LEVEL, colorize=True)

warnings.filterwarnings(
    "ignore", message=".*Torchaudio.*I/O functions.*", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")

# ──────────────────────────────────────────────
# Tab configuration
# ──────────────────────────────────────────────

TAB_CONFIG = {
    "transcribe": ("🎙️ Transcribe", "transcript"),
    "polish": ("✨ Polish", "polished"),
    "index": ("🗂️ Index", "indexed"),
    "translate": ("🌍 Translate", "translation"),
    "synthesize": ("🔊 Synthesize", "generated"),
    "search": ("🔍 Search", ""),
}

# Tabs shown only in podcast mode (show folder + episode management)
PODCAST_ONLY_TABS = {"index", "search"}

TAB_DESCRIPTIONS = {
    "transcribe": "Transcribe an audio file, diarize speakers, and export a clean transcript.",
    "polish": "Fix transcription errors, proper nouns, and spelling using LLMs.",
    "index": "Vectorize episodes for semantic search.",
    "translate": "Translate a transcript (raw or polished) to another language.",
    "synthesize": "Clone speaker voices and synthesize a translated podcast episode.",
    "search": "Query across your indexed episodes using hybrid semantic search.",
}

WORKFLOWS = [
    (
        "🎙️ → ✨ → 🌍 → 🔊",
        "Full pipeline",
        "Transcribe, polish, translate, then synthesize with cloned voices.",
    ),
    (
        "🎙️ → 🌍 → 🔊",
        "Transcribe & translate",
        "Transcribe, translate directly (no polishing), then synthesize.",
    ),
    (
        "🌍 → 🔊",
        "Translate & synthesize",
        "Start from an existing transcript, translate it, then synthesize.",
    ),
    (
        "🌍",
        "Translate only",
        "Import an existing transcript and translate it.",
    ),
]


def get_tab_label(tab_id: str) -> str:
    """Return the tab display label, prefixed with ✅ if the tab's data is loaded."""
    name, state_key = TAB_CONFIG[tab_id]
    is_complete = bool(state_key and st.session_state.get(state_key))
    return f"{'✅ ' if is_complete else ''}{name}"


def init_session_state():
    """Initialize all session-state keys with their defaults (no-op if already set)."""
    defaults = {
        # app mode
        "podcast_mode": False,
        # show-level
        "show_folder": "",
        "show_name": "",
        "show_meta": None,  # ShowMeta loaded from show.toml
        # episode-level
        "audio_path": None,
        "output_dir": str(Path.cwd() / "Transcriptions"),
        "episode_stem": "",
        "episode_title": "",
        "transcript_only": False,
        "transcript": None,
        "polished": None,
        "translations": {},
        "translation": None,
        "generated": None,
        "indexed": None,
        "current_tab": "transcribe",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────


def _select_episode(episode) -> None:
    """Load an episode's data into session state and switch to the transcribe tab.

    For transcript-only episodes (no audio file), a pseudo audio path is set
    so that ``AudioPaths.from_audio`` resolves paths correctly from the
    episode output directory.
    """
    logger.info(
        "Loading episode: {} (transcript_only={})",
        episode.stem,
        episode.audio_path is None,
    )
    from podcodex.core import AudioPaths
    from podcodex.core.transcribe import load_transcript, load_transcript_full
    from podcodex.core.polish import load_polished
    from podcodex.core.translate import load_translation, list_translations

    st.session_state.output_dir = str(episode.output_dir)
    st.session_state.episode_stem = episode.stem
    st.session_state.episode_title = episode.title or episode.stem
    st.session_state.transcript_only = episode.audio_path is None
    # Pseudo audio path for transcript-only episodes keeps AudioPaths working
    audio_path = (
        str(episode.path)
        if episode.path
        else str(episode.output_dir / f"{episode.stem}.audio")
    )
    st.session_state.audio_path = audio_path
    st.session_state.generated = None
    st.session_state.indexed = episode.indexed if episode.indexed else None
    st.session_state.current_tab = "transcribe"

    od = str(episode.output_dir)

    # Auto-detect nodiar mode
    if episode.path:
        nodiar = False
        p_diar = AudioPaths.from_audio(episode.path, output_dir=od)
        p_nodiar = AudioPaths.from_audio(episode.path, output_dir=od, nodiar=True)
        if p_nodiar.transcript_best.exists() and not p_diar.transcript_best.exists():
            nodiar = True
        elif p_diar.transcript_best.exists():
            full = load_transcript_full(episode.path, output_dir=od)
            nodiar = not full.get("meta", {}).get("diarized", True)
    else:
        # Transcript-only: detect nodiar from file naming (same logic as audio)
        pseudo = episode.output_dir / f"{episode.stem}.audio"
        p_diar = AudioPaths.from_audio(pseudo, output_dir=od)
        p_nodiar = AudioPaths.from_audio(pseudo, output_dir=od, nodiar=True)
        nodiar = (
            p_nodiar.transcript_best.exists() and not p_diar.transcript_best.exists()
        )
    st.session_state["skip_diarization"] = nodiar
    st.session_state["_prev_skip_diarization"] = nodiar
    logger.debug("nodiar={}, audio_path={}", nodiar, audio_path)

    # Load transcript, polished, and translations using the (pseudo) audio path
    st.session_state.transcript = (
        load_transcript(audio_path, output_dir=od, nodiar=nodiar)
        if episode.transcribed
        else None
    )
    st.session_state.polished = (
        load_polished(audio_path, output_dir=od, nodiar=nodiar)
        if episode.polished
        else None
    )
    langs = list_translations(audio_path, output_dir=od, nodiar=nodiar)
    st.session_state.translations = {
        lang: load_translation(audio_path, lang, output_dir=od, nodiar=nodiar)
        for lang in langs
    }

    n_segs = len(st.session_state.transcript) if st.session_state.transcript else 0
    logger.info(
        "Episode loaded: {} segments, polished={}, translations={}",
        n_segs,
        bool(st.session_state.polished),
        list(langs),
    )

    # Backward-compat: keep `translation` pointing to the first available translation
    first = next(iter(st.session_state.translations.values()), None)
    st.session_state.translation = first

    st.rerun()


def _load_show_meta_for_folder(folder: str) -> None:
    """Load show.toml for a folder and populate session state."""
    from podcodex.ingest.show import load_show_meta

    meta = load_show_meta(Path(folder)) if folder and Path(folder).is_dir() else None
    st.session_state.show_meta = meta
    if meta:
        st.session_state.show_name = meta.name
        logger.info("Loaded show.toml: name={}, rss={}", meta.name, bool(meta.rss_url))
    else:
        logger.debug("No show.toml found in {}", folder)


def _render_show_settings(folder_input: str) -> None:
    """Render the show settings expander with show.toml fields."""
    from podcodex.ingest.show import ShowMeta, save_show_meta

    meta = st.session_state.get("show_meta")
    with st.expander("⚙️ Show settings", expanded=False):
        rss_url = st.text_input(
            "RSS feed URL",
            value=meta.rss_url if meta else "",
            placeholder="https://example.com/feed.xml",
            key="show_rss_url",
        )
        col_lang, col_speakers = st.columns(2)
        with col_lang:
            language = st.text_input(
                "Language",
                value=meta.language if meta else "",
                placeholder="es",
                key="show_language",
                help="ISO 639-1 language code.",
            )
        with col_speakers:
            speakers_str = st.text_input(
                "Speakers",
                value=", ".join(meta.speakers) if meta else "",
                placeholder="Alice, Bob",
                key="show_speakers",
                help="Comma-separated speaker names.",
            )
        if st.button("💾 Save show.toml", use_container_width=True):
            speakers = [s.strip() for s in speakers_str.split(",") if s.strip()]
            new_meta = ShowMeta(
                name=st.session_state.show_name,
                rss_url=rss_url.strip(),
                language=language.strip(),
                speakers=speakers,
            )
            save_show_meta(Path(folder_input), new_meta)
            st.session_state.show_meta = new_meta
            st.toast("show.toml saved.", icon="✅")


def _render_sidebar() -> None:
    """Render the sidebar, adapting to simple or podcast mode."""
    with st.sidebar:
        podcast_mode = st.toggle(
            "📡 Podcast mode",
            value=st.session_state.podcast_mode,
            key="_podcast_toggle",
            help="Enable show folder, RSS feeds, episode management, indexing and search.",
        )
        if podcast_mode != st.session_state.podcast_mode:
            st.session_state.podcast_mode = podcast_mode
            # When switching to simple mode from podcast, reset tab if on a podcast-only tab
            if (
                not podcast_mode
                and st.session_state.get("current_tab") in PODCAST_ONLY_TABS
            ):
                st.session_state.current_tab = "transcribe"
            st.rerun()

        if podcast_mode:
            _render_sidebar_podcast()
        else:
            _render_sidebar_simple()

        # ── Debug logging toggle ──
        st.divider()
        debug = st.toggle(
            "🐛 Debug logging",
            value=st.session_state.get("debug_logging", False),
            key="_debug_toggle",
            help="Show detailed logs in the terminal.",
        )
        if debug != st.session_state.get("debug_logging", False):
            st.session_state.debug_logging = debug
            st.rerun()


def _render_sidebar_simple() -> None:
    """Sidebar for simple mode: just a file upload."""
    st.markdown("## 🎙️ Audio file")
    st.caption("Drop an audio file to transcribe, polish, translate or synthesize.")

    uploaded = st.file_uploader(
        "Audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        label_visibility="collapsed",
    )
    dest_dir = st.text_input(
        "Output folder",
        value=str(Path.cwd() / "Transcriptions"),
        help="Where to save the episode folder and all outputs.",
    )
    dest_dir = normalize_path(dest_dir)
    if uploaded and st.session_state.get("audio_filename") != uploaded.name:
        stem = Path(uploaded.name).stem
        save_dir = Path(dest_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        audio_dest = save_dir / uploaded.name
        audio_dest.write_bytes(uploaded.read())
        ep_output_dir = save_dir / stem
        ep_output_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.audio_path = str(audio_dest)
        st.session_state.audio_filename = uploaded.name
        st.session_state.output_dir = str(ep_output_dir)
        st.session_state.base_output_dir = str(save_dir)
        st.session_state.episode_stem = stem
        st.session_state.episode_title = stem
        st.session_state.transcript_only = False
        st.session_state.transcript = None
        st.session_state.translation = None
        st.session_state.generated = None
        st.session_state.pop("trim_applied", None)
        st.session_state.current_tab = "transcribe"
        st.rerun()


def _render_sidebar_podcast() -> None:
    """Sidebar for podcast mode: show folder, episode list, settings."""
    st.markdown("## 📂 Show")

    folder_input = st.text_input(
        "Local folder",
        value=st.session_state.show_folder,
        placeholder="/path/to/my-podcast",
        help="Local folder for this show. Audio files and all outputs are stored here, one subfolder per episode. Can be empty — populate it from an RSS feed.",
    )
    folder_input = normalize_path(folder_input)

    # Reload show.toml when folder changes
    if folder_input != st.session_state.get("_prev_show_folder", ""):
        st.session_state.show_name = ""
        _load_show_meta_for_folder(folder_input)
        # Clear show settings and RSS widget keys so they pick up new values
        for k in (
            "show_rss_url",
            "show_language",
            "show_speakers",
            "_rss_feed",
            "_itunes_results",
            "rss_search_query",
            "rss_url_inline",
        ):
            st.session_state.pop(k, None)
    st.session_state["_prev_show_folder"] = folder_input
    st.session_state.show_folder = folder_input

    default_name = st.session_state.show_name or (
        Path(folder_input).name if folder_input else ""
    )
    show_name_input = st.text_input(
        "Show name",
        value=default_name,
        placeholder="My Podcast",
        help="Display name for the show. Overrides the folder name when it is not descriptive enough.",
    )
    st.session_state.show_name = show_name_input

    # Show settings expander (show.toml fields)
    if folder_input and Path(folder_input).is_dir():
        _render_show_settings(folder_input)

    if folder_input:
        folder = Path(folder_input)
        if not folder.is_dir():
            st.warning("Folder not found.")
        else:
            col_overview, col_refresh = st.columns([3, 1])
            with col_overview:
                ep_loaded = st.session_state.get("audio_path") or st.session_state.get(
                    "episode_stem"
                )
                if ep_loaded:
                    if st.button("← Show overview", use_container_width=True):
                        for key in (
                            "audio_path",
                            "transcript",
                            "polished",
                            "translation",
                            "generated",
                            "indexed",
                        ):
                            st.session_state[key] = None
                        st.session_state.translations = {}
                        st.session_state.episode_stem = ""
                        st.session_state.episode_title = ""
                        st.session_state.transcript_only = False
                        _scan_folder_cached.clear()
                        st.rerun()
            with col_refresh:
                if st.button(
                    "🔄", help="Refresh episode status", use_container_width=True
                ):
                    _scan_folder_cached.clear()
                    st.rerun()
            _render_sidebar_search(show_name_input)
            _render_episode_list(folder)


@st.cache_data(ttl=60, show_spinner=False)
def _scan_folder_cached(folder: str) -> list:
    """Cached wrapper for ``scan_folder`` — refreshes every 60s or on manual clear."""
    from podcodex.ingest.folder import scan_folder

    return scan_folder(Path(folder))


def _render_episode_list(folder: Path) -> None:
    """Render the scrollable episode list in the sidebar with status badges."""
    episodes = _scan_folder_cached(str(folder))
    if not episodes:
        st.info("No episodes found.")
        return

    n_polished = sum(1 for e in episodes if e.polished)
    n_transcribed = sum(1 for e in episodes if e.transcribed and not e.polished)
    n_indexed = sum(1 for e in episodes if e.indexed)
    n_pending = len(episodes) - sum(1 for e in episodes if e.transcribed)
    summary = f"{len(episodes)} episodes — 🟢 {n_polished} · 🟡 {n_transcribed} · 🔴 {n_pending}"
    if n_indexed:
        summary += f" · ⚡ {n_indexed} indexed"
    st.caption(summary)

    active_stem = st.session_state.get("episode_stem", "")
    with st.container(height=400, border=False):
        for ep in episodes:
            is_active = ep.stem == active_stem
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                if ep.validated_polished:
                    badge = "🟢"
                elif ep.polished:
                    badge = "🟡"
                elif ep.transcribed:
                    badge = "🟡"
                elif ep.audio_path is None:
                    badge = "⚪"  # no audio (feed-only or transcript-only without pipeline progress)
                else:
                    badge = "🔴"
                has_unvalidated = (
                    ep.raw_transcript or ep.raw_polished or bool(ep.raw_translations)
                )
                warn_mark = " ⚠️" if has_unvalidated else ""
                indexed_mark = " ⚡" if ep.indexed else ""
                display_name = ep.title or ep.stem
                label = f"**{display_name}**" if is_active else display_name
                st.markdown(f"{badge} {label}{warn_mark}{indexed_mark}")
            with col_btn:
                if is_active:
                    st.button("Open", key=f"open_{ep.stem}", disabled=True)
                else:
                    if st.button("Open", key=f"open_{ep.stem}"):
                        _select_episode(ep)


def _render_sidebar_search(show_name: str) -> None:
    """Render the sidebar search form that redirects to the RAG search tab."""
    with st.form("sidebar_search_form", clear_on_submit=False):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            search_q = st.text_input(
                "Search episodes",
                placeholder="🔍 Search episodes…",
                label_visibility="collapsed",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "→", help="Search across all indexed episodes"
            )
        if submitted and search_q.strip():
            st.session_state["rag_search_query"] = search_q.strip()
            st.session_state.requested_tab = "search"
            st.rerun()
    st.divider()


# ──────────────────────────────────────────────
# RSS and import helpers
# ──────────────────────────────────────────────


def _search_itunes(query: str, limit: int = 8) -> list[dict]:
    """Search the iTunes/Apple Podcasts directory for podcasts.

    Returns a list of dicts with keys: name, artist, feed_url, artwork_url.
    """
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode(
        {
            "term": query,
            "media": "podcast",
            "limit": limit,
        }
    )
    url = f"https://itunes.apple.com/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "podcodex/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        import json

        data = json.loads(resp.read())
    results = []
    for r in data.get("results", []):
        feed = r.get("feedUrl", "")
        if not feed:
            continue
        results.append(
            {
                "name": r.get("collectionName", ""),
                "artist": r.get("artistName", ""),
                "feed_url": feed,
                "artwork_url": r.get("artworkUrl60", ""),
            }
        )
    return results


def _render_rss_setup(folder: str, show_name: str) -> None:
    """Render the RSS feed setup UI: podcast search + manual URL input."""
    from podcodex.ingest.show import ShowMeta, save_show_meta

    meta = st.session_state.get("show_meta")

    with st.expander("📡 Add RSS Feed", expanded=True):
        st.caption("Search for a podcast or paste an RSS feed URL directly.")

        # ── Podcast search ──
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            search_query = st.text_input(
                "Search podcasts",
                placeholder="Search by podcast name…",
                key="rss_search_query",
                label_visibility="collapsed",
            )
        with col_btn:
            search_clicked = st.button(
                "🔍 Search",
                use_container_width=True,
                disabled=not search_query.strip(),
            )

        if search_clicked and search_query.strip():
            with st.spinner("Searching…"):
                try:
                    st.session_state["_itunes_results"] = _search_itunes(
                        search_query.strip()
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state["_itunes_results"] = []

        results = st.session_state.get("_itunes_results")
        if results:
            for i, pod in enumerate(results):
                col_art, col_info, col_use = st.columns([1, 5, 2])
                with col_art:
                    if pod["artwork_url"]:
                        st.image(pod["artwork_url"], width=50)
                with col_info:
                    st.markdown(f"**{pod['name']}**")
                    st.caption(pod["artist"])
                with col_use:
                    if st.button(
                        "Use this feed",
                        key=f"itunes_{i}",
                        use_container_width=True,
                    ):
                        new_meta = ShowMeta(
                            name=pod["name"] or show_name,
                            rss_url=pod["feed_url"],
                            language=meta.language if meta else "",
                            speakers=meta.speakers if meta else [],
                        )
                        save_show_meta(Path(folder), new_meta)
                        st.session_state.show_meta = new_meta
                        st.session_state.show_name = new_meta.name
                        st.session_state.pop("show_rss_url", None)
                        st.session_state.pop("_itunes_results", None)
                        st.toast(f"Feed saved — {pod['name']}", icon="✅")
                        st.rerun()
        elif results is not None:
            st.info("No podcasts found. Try a different search term.")

        # ── Manual URL fallback ──
        st.divider()
        st.caption("Or paste a feed URL directly:")
        col_url, col_save = st.columns([4, 1])
        with col_url:
            new_url = st.text_input(
                "RSS feed URL",
                placeholder="https://example.com/feed.xml",
                key="rss_url_inline",
                label_visibility="collapsed",
            )
        with col_save:
            if st.button(
                "Add feed",
                use_container_width=True,
                type="primary",
                disabled=not new_url.strip(),
            ):
                new_meta = ShowMeta(
                    name=show_name,
                    rss_url=new_url.strip(),
                    language=meta.language if meta else "",
                    speakers=meta.speakers if meta else [],
                )
                save_show_meta(Path(folder), new_meta)
                st.session_state.show_meta = new_meta
                st.session_state.pop("show_rss_url", None)
                st.toast("RSS feed saved to show.toml.", icon="✅")
                st.rerun()


def _open_feed_episode(rss_ep, folder: str) -> None:
    """Open a feed-only episode (no local audio) by creating its metadata and folder."""
    from podcodex.ingest.rss import episode_stem as rss_stem, save_episode_meta
    from podcodex.ingest.folder import EpisodeInfo

    stem = rss_stem(rss_ep)
    output_dir = Path(folder) / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    save_episode_meta(output_dir, rss_ep)

    ep = EpisodeInfo(
        audio_path=None,
        stem=stem,
        output_dir=output_dir,
        title=rss_ep.title,
    )
    _scan_folder_cached.clear()
    _select_episode(ep)


def _get_feed_episodes(folder: str, rss_url: str) -> list | None:
    """Return cached feed episodes, auto-fetching on first load if needed."""
    from podcodex.ingest.rss import fetch_feed, load_feed_cache, save_feed_cache

    feed = st.session_state.get("_rss_feed") or load_feed_cache(Path(folder))
    if feed:
        return feed

    # Auto-fetch on first visit when URL is configured but no cache exists
    logger.info("Auto-fetching RSS feed for {}", rss_url)
    try:
        feed = fetch_feed(rss_url)
        save_feed_cache(Path(folder), feed)
        st.session_state["_rss_feed"] = feed
        return feed
    except Exception as e:
        logger.warning("Auto-fetch failed: {}", e)
        return None


def _render_rss_controls(folder: str, rss_url: str) -> None:
    """Render the RSS feed URL and fetch/change controls."""
    from podcodex.ingest.rss import fetch_feed, save_feed_cache

    col_url, col_fetch = st.columns([5, 1])
    with col_url:
        st.caption(f"📡 {rss_url}")
    with col_fetch:
        if st.button("🔄 Fetch", use_container_width=True, help="Refresh the feed"):
            with st.spinner("Fetching feed…"):
                try:
                    feed_episodes = fetch_feed(rss_url)
                    save_feed_cache(Path(folder), feed_episodes)
                    st.session_state["_rss_feed"] = feed_episodes
                    _scan_folder_cached.clear()
                    logger.info("Fetched {} episodes from RSS", len(feed_episodes))
                    st.toast(f"Fetched {len(feed_episodes)} episodes.", icon="✅")
                    st.rerun()
                except Exception as e:
                    logger.error("Feed fetch failed: {}", e)
                    st.error(f"Feed fetch failed: {e}")


# ──────────────────────────────────────────────
# Show overview dashboard
# ──────────────────────────────────────────────


def _render_show_overview() -> None:
    """Render the show dashboard: RSS feed, import, and unified episode list."""
    from podcodex.ingest.rss import episode_stem as rss_stem, download_audio

    folder = st.session_state.get("show_folder", "")
    show_name = st.session_state.get("show_name") or Path(folder).name
    local_episodes = _scan_folder_cached(folder)
    meta = st.session_state.get("show_meta")
    rss_url = meta.rss_url if meta else ""

    n_local = len(local_episodes)
    n_transcribed = sum(1 for e in local_episodes if e.transcribed)
    n_polished = sum(1 for e in local_episodes if e.polished)
    n_indexed = sum(1 for e in local_episodes if e.indexed)

    # ── RSS setup or controls ──
    if not rss_url:
        _render_rss_setup(folder, show_name)
    else:
        _render_rss_controls(folder, rss_url)

    # ── Summary stats ──
    summary = f"**{show_name}** — {n_local} local episode{'s' if n_local != 1 else ''}"
    if n_transcribed:
        summary += f" · 📝 {n_transcribed} transcribed"
    if n_polished:
        summary += f" · ✨ {n_polished} polished"
    if n_indexed:
        summary += f" · ⚡ {n_indexed} indexed"
    st.markdown(summary)

    st.divider()

    # ── Unified episode table (local + RSS) ──
    feed_episodes = _get_feed_episodes(folder, rss_url) if rss_url else None

    local_by_stem = {ep.stem: ep for ep in local_episodes}
    active_stem = st.session_state.get("episode_stem", "")

    if feed_episodes:
        # Merge: iterate feed episodes (keeps feed order = newest first),
        # then append any local-only episodes not in the feed.
        feed_slugs = []
        not_downloaded = [
            ep
            for ep in feed_episodes
            if ep.audio_url and rss_stem(ep) not in local_by_stem
        ]

        col_title, col_dl_all = st.columns([4, 2])
        with col_title:
            st.markdown(f"**{len(feed_episodes)} episodes in feed**")
        with col_dl_all:
            if not_downloaded:
                if st.button(
                    f"⬇️ Download all ({len(not_downloaded)})",
                    use_container_width=True,
                    help=f"Download {len(not_downloaded)} episodes that aren't local yet.",
                ):
                    progress = st.progress(0, text="Downloading…")
                    downloaded = 0
                    for j, ep in enumerate(not_downloaded):
                        slug = rss_stem(ep)
                        progress.progress(
                            (j + 1) / len(not_downloaded),
                            text=f"Downloading {ep.title}…",
                        )
                        try:
                            download_audio(ep, Path(folder))
                            downloaded += 1
                            logger.info("Downloaded {}", slug)
                        except Exception as e:
                            logger.error("Download failed for {}: {}", slug, e)
                            st.warning(f"Failed: {slug} — {e}")
                    progress.empty()
                    _scan_folder_cached.clear()
                    st.toast(
                        f"Downloaded {downloaded}/{len(not_downloaded)} episodes.",
                        icon="✅",
                    )
                    st.rerun()

        # Header row
        cols = st.columns([1, 4, 2, 3, 2])
        for col, label in zip(cols, ["#", "Episode", "Date", "Status", ""]):
            col.markdown(f"**{label}**")

        for i, rss_ep in enumerate(feed_episodes):
            slug = rss_stem(rss_ep)
            feed_slugs.append(slug)
            local = local_by_stem.get(slug)
            is_active = slug == active_stem

            cols = st.columns([1, 4, 2, 3, 2])
            with cols[0]:
                ep_num = rss_ep.episode_number
                st.caption(str(ep_num) if ep_num is not None else "—")
            with cols[1]:
                name = f"**{rss_ep.title}**" if is_active else rss_ep.title
                st.markdown(name)
            with cols[2]:
                st.caption(rss_ep.pub_date[:10] if rss_ep.pub_date else "—")
            with cols[3]:
                if local:
                    _render_local_status_compact(local)
                else:
                    st.caption("Feed only" + (" · 🎧" if rss_ep.audio_url else ""))
            with cols[4]:
                if local:
                    if is_active:
                        st.button("Open", key=f"ov_{i}_{slug}", disabled=True)
                    elif st.button("Open", key=f"ov_{i}_{slug}"):
                        _select_episode(local)
                else:
                    col_open, col_dl = st.columns(2)
                    with col_open:
                        if is_active:
                            st.button("Open", key=f"ov_{i}_{slug}", disabled=True)
                        elif st.button("Open", key=f"ov_{i}_{slug}"):
                            _open_feed_episode(rss_ep, folder)
                    with col_dl:
                        if rss_ep.audio_url:
                            if st.button(
                                "⬇️", key=f"dl_{i}_{slug}", help="Download audio"
                            ):
                                with st.spinner("Downloading…"):
                                    try:
                                        download_audio(rss_ep, Path(folder))
                                        _scan_folder_cached.clear()
                                        logger.info("Downloaded {}", slug)
                                        st.toast(f"Downloaded {slug}.", icon="✅")
                                        st.rerun()
                                    except Exception as e:
                                        logger.error(
                                            "Download failed for {}: {}", slug, e
                                        )
                                        st.error(f"Download failed: {e}")

        # Local-only episodes not in the feed
        local_only = [ep for ep in local_episodes if ep.stem not in set(feed_slugs)]
        if local_only:
            st.divider()
            st.caption(f"{len(local_only)} local episode(s) not in feed:")
            _render_local_episode_table(local_only, active_stem, key_prefix="lo")

    elif local_episodes:
        # No feed — show local episodes only
        _render_local_episode_table(local_episodes, active_stem, key_prefix="ov")

    else:
        st.info(
            "No episodes yet. "
            + ("Click **Fetch** above to load the feed." if rss_url else "")
        )


def _render_local_status_compact(ep) -> None:
    """Show compact pipeline status for one local episode."""
    parts = []
    if ep.audio_path is None and ep.transcribed:
        parts.append("📄")  # transcript-only (imported)
    elif ep.audio_path is None:
        parts.append("⚪")  # metadata only
    if ep.transcribed:
        parts.append("📝")
    if ep.polished:
        parts.append("✨")
    if ep.translations:
        parts.append("🌍 " + ", ".join(t.capitalize() for t in ep.translations))
    if ep.indexed:
        parts.append("⚡")
    if not parts:
        st.caption("Downloaded")
    else:
        st.markdown(" · ".join(parts))


def _render_local_episode_table(
    episodes: list, active_stem: str, key_prefix: str
) -> None:
    """Render a local episode table with pipeline status columns."""
    cols = st.columns([4, 3, 1, 3, 1, 1])
    for col, label in zip(
        cols, ["Episode", "Transcription", "Polish", "Translations", "Index", ""]
    ):
        col.markdown(f"**{label}**")

    for ep in episodes:
        cols = st.columns([4, 3, 1, 3, 1, 1])
        is_active = ep.stem == active_stem
        display_name = ep.title or ep.stem
        label = f"**{display_name}**" if is_active else display_name
        if ep.audio_path is None:
            label += " 📄"
        cols[0].markdown(label)

        # Pipeline progress
        if ep.audio_path is None:
            pipeline = "📄 Imported" if ep.transcribed else "⬜ No transcript"
        elif ep.transcribed:
            pipeline = "✅ Done"
        elif ep.mapped:
            pipeline = "🟡 Ready to export"
        elif ep.assigned:
            pipeline = "🟡 Needs speaker map"
        elif ep.diarized:
            pipeline = "🟠 Needs assignment"
        elif ep.segments_ready:
            pipeline = "🟠 Needs diarization"
        else:
            pipeline = "⬜ Not started"
        cols[1].markdown(pipeline)

        if ep.validated_polished:
            cols[2].markdown("✅")
        elif ep.raw_polished:
            cols[2].markdown("⚠️ Raw")
        else:
            cols[2].markdown("⬜")
        if ep.translations:
            cols[3].markdown(", ".join(lang.capitalize() for lang in ep.translations))
        else:
            cols[3].markdown("—")
        cols[4].markdown("⚡" if ep.indexed else "—")
        with cols[5]:
            if is_active:
                st.button("Open", key=f"{key_prefix}_{ep.stem}", disabled=True)
            else:
                if st.button("Open", key=f"{key_prefix}_{ep.stem}"):
                    _select_episode(ep)


# ──────────────────────────────────────────────
# Getting started
# ──────────────────────────────────────────────


def _render_getting_started():
    """Render the onboarding panel, adapting to the current mode."""
    podcast_mode = st.session_state.podcast_mode

    if podcast_mode:
        # Podcast mode: prompt to set a show folder
        st.info(
            "Pick a **local folder** for this show in the sidebar — "
            "this is where audio files and all processing outputs will be stored. "
            "You can start with an empty folder and populate it from an RSS feed."
        )
        return

    # Simple mode: workflow overview
    with st.expander("🚀 Getting started", expanded=True):
        st.caption(
            "Upload an audio file in the sidebar to get started, "
            "or switch to **Podcast mode** for show management, RSS feeds, and search."
        )
        for emoji_flow, title, desc in WORKFLOWS:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"**{emoji_flow}**")
            with col2:
                st.markdown(f"**{title}** — {desc}")

        st.divider()
        st.markdown("**Where to start:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🎙️ **Have an audio file?**\nUpload it in the sidebar.")
        with col2:
            st.info(
                "📄 **Have a transcript?**\nGo to **Polish** or **Translate** to import it."
            )
        with col3:
            st.info("🌍 **Have a translation?**\nGo to **Synthesize** and import it.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    st.set_page_config(layout="wide", page_title="podcodex")

    st.markdown(
        """<style>
        /* Tighten vertical spacing between containers */
        div[data-testid="stVerticalBlock"] > div { padding-top: 0.25rem; padding-bottom: 0.25rem; }
        /* Reduce top padding in main content area */
        .block-container { padding-top: 2rem; }
        /* Compact expander headers */
        details[data-testid="stExpander"] summary { padding: 0.4rem 0.6rem; }
        </style>""",
        unsafe_allow_html=True,
    )

    init_session_state()

    _render_sidebar()

    podcast_mode = st.session_state.podcast_mode

    show_name = st.session_state.get("show_name") or "podcodex"
    st.title(show_name)
    st.markdown("*Podcast transcription, translation & voice synthesis*")
    st.divider()

    ep_loaded = st.session_state.get("audio_path") or st.session_state.get(
        "episode_stem"
    )
    nothing_loaded = not any(
        [
            ep_loaded,
            st.session_state.get("transcript"),
            st.session_state.get("translation"),
        ]
    )

    # Show overview (podcast mode only) or getting started
    if podcast_mode and st.session_state.get("show_folder") and nothing_loaded:
        _render_show_overview()
        st.divider()
    elif nothing_loaded:
        _render_getting_started()
        st.divider()
    elif ep_loaded:
        ep_name = (
            st.session_state.get("episode_title")
            or st.session_state.get("episode_stem")
            or (
                Path(st.session_state.audio_path).stem
                if st.session_state.get("audio_path")
                else ""
            )
        )
        label = f"Episode: {ep_name}"
        if st.session_state.get("transcript_only"):
            if st.session_state.get("transcript"):
                label += " (transcript only)"
            else:
                label += " (no audio)"
        st.subheader(label)

    # Tab navigation — filter tabs by mode
    visible_tabs = [
        tab_id
        for tab_id in TAB_CONFIG
        if podcast_mode or tab_id not in PODCAST_ONLY_TABS
    ]

    # Handle requested_tab before rendering widget
    if "requested_tab" in st.session_state:
        requested = st.session_state.pop("requested_tab")
        if requested in visible_tabs:
            st.session_state.current_tab = requested

    # Ensure current_tab is valid for the current mode
    if st.session_state.get("current_tab") not in visible_tabs:
        st.session_state.current_tab = "transcribe"

    current_tab = st.segmented_control(
        label="Navigation",
        options=visible_tabs,
        format_func=get_tab_label,
        default=st.session_state.get("current_tab", "transcribe"),
        label_visibility="collapsed",
    )

    if current_tab:
        st.session_state.current_tab = current_tab
    else:
        current_tab = st.session_state.get("current_tab", "transcribe")

    st.caption(TAB_DESCRIPTIONS[current_tab])
    st.divider()

    if current_tab == "transcribe":
        import streamlit_transcribe as ui

        ui.render()
    elif current_tab == "polish":
        import streamlit_polish as ui

        ui.render()
    elif current_tab == "index":
        import streamlit_index as ui

        ui.render()
    elif current_tab == "translate":
        import streamlit_translate as ui

        ui.render()
    elif current_tab == "synthesize":
        import streamlit_synthesize as ui

        ui.render()
    elif current_tab == "search":
        import streamlit_search as ui

        ui.render()


if __name__ == "__main__":
    main()
