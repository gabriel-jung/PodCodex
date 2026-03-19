"""
podcodex — Podcast transcription, translation and synthesis app.

Run with: streamlit run streamlit/app.py
"""

import warnings
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from utils import normalize_path

load_dotenv()

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
        # show-level
        "show_folder": "",
        "show_name": "",
        # episode-level
        "audio_path": None,
        "output_dir": str(Path.cwd() / "Transcriptions"),
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
    """Load an episode's data into session state and switch to the transcribe tab."""
    from podcodex.core import AudioPaths
    from podcodex.core.transcribe import load_transcript, load_transcript_full
    from podcodex.core.polish import load_polished
    from podcodex.core.translate import load_translation, list_translations

    st.session_state.audio_path = str(episode.path)
    st.session_state.output_dir = str(episode.output_dir)
    st.session_state.generated = None
    st.session_state.indexed = episode.indexed if episode.indexed else None
    st.session_state.current_tab = "transcribe"

    # Auto-detect nodiar mode from existing transcript metadata
    od = str(episode.output_dir)
    p_diar = AudioPaths.from_audio(episode.path, output_dir=od)
    p_nodiar = AudioPaths.from_audio(episode.path, output_dir=od, nodiar=True)
    nodiar = False
    if p_nodiar.transcript_best.exists() and not p_diar.transcript_best.exists():
        # Only nodiar transcript exists — auto-enable
        nodiar = True
    elif p_diar.transcript_best.exists():
        # Diarized transcript exists — check its meta
        full = load_transcript_full(episode.path, output_dir=od)
        nodiar = not full.get("meta", {}).get("diarized", True)
    st.session_state["skip_diarization"] = nodiar
    st.session_state["_prev_skip_diarization"] = nodiar

    st.session_state.transcript = (
        load_transcript(episode.path, output_dir=od, nodiar=nodiar)
        if episode.transcribed
        else None
    )
    st.session_state.polished = (
        load_polished(episode.path, output_dir=od, nodiar=nodiar)
        if episode.polished
        else None
    )

    langs = list_translations(episode.path, output_dir=od, nodiar=nodiar)
    st.session_state.translations = {
        lang: load_translation(episode.path, lang, output_dir=od, nodiar=nodiar)
        for lang in langs
    }
    # Backward-compat: keep `translation` pointing to the first available translation
    first = next(iter(st.session_state.translations.values()), None)
    st.session_state.translation = first

    st.rerun()


def _render_sidebar() -> None:
    """Render the sidebar: show folder, episode list, search, and single-file upload."""
    with st.sidebar:
        st.markdown("## 📂 Show")

        folder_input = st.text_input(
            "Show folder",
            value=st.session_state.show_folder,
            placeholder="/path/to/show/folder",
            help="Folder containing your audio files. Outputs for each episode are saved in a subfolder named after the episode.",
        )
        folder_input = normalize_path(folder_input)

        # Reset show name when folder changes
        if folder_input != st.session_state.get("_prev_show_folder", ""):
            st.session_state.show_name = ""
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

        if folder_input:
            folder = Path(folder_input)
            if not folder.is_dir():
                st.warning("Folder not found.")
            else:
                col_overview, col_refresh = st.columns([3, 1])
                with col_overview:
                    if st.session_state.get("audio_path"):
                        if st.button("← Show overview", use_container_width=True):
                            for key in (
                                "audio_path",
                                "transcript",
                                "polished",
                                "translation",
                                "generated",
                            ):
                                st.session_state[key] = None
                            st.session_state.translations = {}
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

        st.divider()

        with st.expander(
            "📁 Upload a single file",
            expanded=not bool(folder_input),
        ):
            st.caption("Use this to process a single audio file without a show folder.")
            uploaded = st.file_uploader(
                "Audio file",
                type=["mp3", "wav", "m4a", "ogg", "flac"],
                label_visibility="collapsed",
            )
            dest_dir = st.text_input(
                "Destination folder",
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
                st.session_state.transcript = None
                st.session_state.translation = None
                st.session_state.generated = None
                st.session_state.pop("trim_applied", None)
                st.session_state.current_tab = "transcribe"
                st.rerun()


@st.cache_data(ttl=60, show_spinner=False)
def _scan_folder_cached(folder: str) -> list:
    """Cached wrapper for ``scan_folder`` — refreshes every 60s or on manual clear."""
    from podcodex.ingest.folder import scan_folder

    return scan_folder(Path(folder))


def _render_episode_list(folder: Path) -> None:
    """Render the scrollable episode list in the sidebar with status badges."""
    episodes = _scan_folder_cached(str(folder))
    if not episodes:
        st.info("No audio files found.")
        return

    n_polished = sum(1 for e in episodes if e.polished)
    n_transcribed = sum(1 for e in episodes if e.transcribed and not e.polished)
    n_indexed = sum(1 for e in episodes if e.indexed)
    n_pending = len(episodes) - sum(1 for e in episodes if e.transcribed)
    summary = f"{len(episodes)} episodes — 🟢 {n_polished} · 🟡 {n_transcribed} · 🔴 {n_pending}"
    if n_indexed:
        summary += f" · ⚡ {n_indexed} indexed"
    st.caption(summary)

    active_path = st.session_state.get("audio_path")
    with st.container(height=400, border=False):
        for ep in episodes:
            is_active = bool(active_path and Path(active_path) == ep.path)
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                if ep.validated_polished:
                    badge = "🟢"
                elif ep.polished:  # raw polished exists
                    badge = "🟡"
                elif ep.transcribed:
                    badge = "🟡"
                else:
                    badge = "🔴"
                has_unvalidated = (
                    ep.raw_transcript or ep.raw_polished or bool(ep.raw_translations)
                )
                warn_mark = " ⚠️" if has_unvalidated else ""
                indexed_mark = " ⚡" if ep.indexed else ""
                label = f"**{ep.stem}**" if is_active else ep.stem
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
# Show overview dashboard
# ──────────────────────────────────────────────


def _render_show_overview() -> None:
    """Render the show dashboard table with per-episode pipeline status."""
    folder = st.session_state.get("show_folder", "")
    show_name = st.session_state.get("show_name") or Path(folder).name
    episodes = _scan_folder_cached(folder)

    n_transcribed = sum(1 for e in episodes if e.transcribed)
    n_polished = sum(1 for e in episodes if e.polished)
    n_indexed = sum(1 for e in episodes if e.indexed)

    st.markdown(
        f"**{show_name}** — {len(episodes)} episode{'s' if len(episodes) != 1 else ''} · "
        f"📝 {n_transcribed} transcribed · ✨ {n_polished} polished · ⚡ {n_indexed} indexed"
    )
    st.divider()

    if not episodes:
        st.info("No audio files found in this folder.")
        return

    # Header row
    cols = st.columns([4, 3, 1, 3, 1, 1])
    for col, label in zip(
        cols, ["Episode", "Transcribing", "Polishing", "Translating", "Indexing", ""]
    ):
        col.markdown(f"**{label}**")

    active_path = st.session_state.get("audio_path")
    for ep in episodes:
        cols = st.columns([4, 3, 1, 3, 1, 1])
        is_active = bool(active_path and Path(active_path) == ep.path)
        label = f"**{ep.stem}**" if is_active else ep.stem
        cols[0].markdown(label)

        # Pipeline progress: segments → diarized → assigned → mapped → exported
        if ep.transcribed:
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
                st.button("Open", key=f"ov_open_{ep.stem}", disabled=True)
            else:
                if st.button("Open", key=f"ov_open_{ep.stem}"):
                    _select_episode(ep)


# ──────────────────────────────────────────────
# Getting started
# ──────────────────────────────────────────────


def _render_getting_started():
    """Render the onboarding panel with workflow options and entry points."""
    with st.expander("🚀 Getting started", expanded=True):
        st.caption(
            "podcodex supports several workflows depending on what you already have."
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
            st.info("🎙️ **Have an audio file?**\nStart in the **Transcribe** tab.")
        with col2:
            st.info(
                "📄 **Have a transcript?**\nGo to **Refine & Translate** to correct or translate it."
            )
        with col3:
            st.info(
                "🌍 **Have a translation?**\nGo to the **Synthesize** tab and import your translation."
            )


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

    show_name = st.session_state.get("show_name") or "podcodex"
    st.title(show_name)
    st.markdown("*Podcast transcription, translation & voice synthesis*")
    st.divider()

    nothing_loaded = not any(
        [
            st.session_state.get("audio_path"),
            st.session_state.get("transcript"),
            st.session_state.get("translation"),
        ]
    )
    if st.session_state.get("show_folder") and nothing_loaded:
        _render_show_overview()
        st.divider()
    elif nothing_loaded:
        _render_getting_started()
        st.divider()
    elif st.session_state.get("audio_path"):
        ep_name = Path(st.session_state.audio_path).stem
        st.subheader(f"Episode: {ep_name}")

    # Tab navigation — handle requested_tab before rendering widget
    if "requested_tab" in st.session_state:
        st.session_state.current_tab = st.session_state.pop("requested_tab")

    current_tab = st.segmented_control(
        label="Navigation",
        options=list(TAB_CONFIG.keys()),
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
