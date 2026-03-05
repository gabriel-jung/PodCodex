"""
podcodex — Show-level view wrapping the pipeline tabs.

Run with: streamlit run streamlit/show_app.py
"""

import warnings
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings(
    "ignore", message=".*Torchaudio.*I/O functions.*", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")

# ──────────────────────────────────────────────
# Tab configuration (mirrors app.py)
# ──────────────────────────────────────────────

TAB_CONFIG = {
    "transcribe": ("🎙️ Transcribe", "transcript"),
    "translate": ("🌍 Translate", "translation"),
    "synthesize": ("🔊 Synthesize", "generated"),
}

TAB_DESCRIPTIONS = {
    "transcribe": "Transcribe an audio file, diarize speakers, and export a clean transcript.",
    "translate": "Translate a transcript using an API, local LLM, or manual copy/paste. Can be used standalone without the Transcribe step.",
    "synthesize": "Clone speaker voices and synthesize a translated podcast episode.",
}


def get_tab_label(tab_id: str) -> str:
    """Build tab label with checkmark if step is complete."""
    name, state_key = TAB_CONFIG[tab_id]
    is_complete = bool(st.session_state.get(state_key))
    return f"{'✅ ' if is_complete else ''}{name}"


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────


def init_session_state():
    defaults = {
        # show-level
        "show_folder": "",
        "show_name": "",
        # episode-level (same as app.py)
        "audio_path": None,
        "output_dir": str(Path.cwd() / "Transcriptions"),
        "transcript": None,
        "translation": None,
        "generated": None,
        "current_tab": "transcribe",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ──────────────────────────────────────────────
# Episode selection
# ──────────────────────────────────────────────


def _select_episode(episode) -> None:
    st.session_state.audio_path = str(episode.path)
    st.session_state.output_dir = str(episode.output_dir)
    st.session_state.transcript = None
    st.session_state.translation = None
    st.session_state.generated = None
    st.session_state.current_tab = "transcribe"
    st.rerun()


# ──────────────────────────────────────────────
# Show panel
# ──────────────────────────────────────────────


def _render_show_panel() -> None:
    col1, col2 = st.columns([3, 2])
    with col1:
        folder_input = st.text_input(
            "Show folder",
            value=st.session_state.show_folder,
            placeholder="/path/to/show/folder",
        )
    with col2:
        default_name = st.session_state.show_name or (
            Path(folder_input).name if folder_input else ""
        )
        show_name_input = st.text_input(
            "Show name",
            value=default_name,
            placeholder="my_podcast",
        )

    st.session_state.show_folder = folder_input
    st.session_state.show_name = show_name_input

    if not folder_input:
        return

    folder = Path(folder_input)
    if not folder.is_dir():
        st.warning(f"Folder not found: {folder_input}")
        return

    from podcodex.ingest.folder import scan_folder

    episodes = scan_folder(folder)
    if not episodes:
        st.info("No audio files found in this folder.")
        return

    active_path = st.session_state.get("audio_path")
    for ep in episodes:
        is_active = bool(active_path and Path(active_path) == ep.path)
        col_name, col_status, col_btn = st.columns([4, 2, 1])
        with col_name:
            if is_active:
                st.markdown(f"**{ep.stem}**")
            else:
                st.text(ep.stem)
        with col_status:
            if ep.indexed:
                st.markdown("🟢 Indexed")
            elif ep.transcribed:
                st.markdown("🟡 Transcribed")
            else:
                st.markdown("🔴 Pending")
        with col_btn:
            if is_active:
                st.button("Open", key=f"open_{ep.stem}", disabled=True)
            else:
                if st.button("Open", key=f"open_{ep.stem}"):
                    _select_episode(ep)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    st.set_page_config(layout="wide", page_title="podcodex — show view")
    init_session_state()

    st.title("podcodex")
    st.markdown("*Podcast transcription, translation & voice synthesis*")

    _render_show_panel()

    if not st.session_state.get("audio_path"):
        return

    st.divider()

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
    elif current_tab == "translate":
        import streamlit_translate as ui

        ui.render()
    elif current_tab == "synthesize":
        import streamlit_synthesize as ui

        ui.render()


if __name__ == "__main__":
    main()
