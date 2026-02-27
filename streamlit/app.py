"""
podcodex — Podcast transcription, translation and synthesis app.
"""

import warnings
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Suppress known harmless warnings
warnings.filterwarnings(
    "ignore", message=".*Torchaudio.*I/O functions.*", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")

# ──────────────────────────────────────────────
# Tab configuration
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

WORKFLOWS = [
    (
        "🎙️ → 🌍 → 🔊",
        "Full pipeline",
        "Transcribe an audio file, translate it, then synthesize a new episode with cloned voices.",
    ),
    (
        "🌍 → 🔊",
        "Translate & synthesize",
        "Start from an existing transcript JSON. Translate it, then synthesize.",
    ),
    (
        "🎙️ → 🌍",
        "Transcribe & translate",
        "Transcribe and translate only — no synthesis.",
    ),
    (
        "🌍",
        "Translate only",
        "Import an existing transcript and translate it using any mode.",
    ),
]


def get_tab_label(tab_id: str) -> str:
    """Build tab label with checkmark if step is complete."""
    name, state_key = TAB_CONFIG[tab_id]
    is_complete = bool(st.session_state.get(state_key))
    return f"{'✅ ' if is_complete else ''}{name}"


def init_session_state():
    defaults = {
        "audio_path": None,
        "output_dir": str(Path.cwd() / "Transcriptions"),
        "transcript": None,
        "translation": None,
        "generated": None,
        "current_tab": "transcribe",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_getting_started():
    """Show getting started guide when nothing is loaded yet."""
    with st.container(border=True):
        st.markdown("### 🚀 Getting Started")
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
                "📄 **Have a transcript?**\nGo directly to the **Translate** tab and import your JSON."
            )
        with col3:
            st.info(
                "🌍 **Have a translation?**\nGo to the **Synthesize** tab and import your translation."
            )


def main():
    st.set_page_config(layout="wide", page_title="podcodex")
    init_session_state()

    st.title("podcodex")
    st.markdown("*Podcast transcription, translation & voice synthesis*")
    st.divider()

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

    # Persist selection — fall back to session state if widget returns None
    if current_tab:
        st.session_state.current_tab = current_tab
    else:
        current_tab = st.session_state.get("current_tab", "transcribe")

    # Tab description
    st.caption(TAB_DESCRIPTIONS[current_tab])

    st.divider()

    # Getting started — show only when nothing loaded
    nothing_loaded = not any(
        [
            st.session_state.get("audio_path"),
            st.session_state.get("transcript"),
            st.session_state.get("translation"),
        ]
    )
    if nothing_loaded and current_tab == "transcribe":
        _render_getting_started()
        st.divider()

    # Tab content
    if current_tab == "transcribe":
        import streamlit_transcribe as ui_transcribe

        ui_transcribe.render()

    elif current_tab == "translate":
        import streamlit_translate as ui_translate

        ui_translate.render()

    elif current_tab == "synthesize":
        import streamlit_synthesize as ui_synthesize

        ui_synthesize.render()


if __name__ == "__main__":
    main()
