"""
podcodex.ui.streamlit_index — Index tab (vectorize episodes for search)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from podcodex.core._utils import AudioPaths

try:
    from podcodex.rag.defaults import (
        CHUNK_SIZE,
        CHUNK_THRESHOLD,
        CHUNKING_STRATEGIES,
        DEFAULT_CHUNKING,
        DEFAULT_MODEL,
        MODELS,
    )
    from podcodex.rag.localstore import LocalStore
    from podcodex.rag.store import collection_name

    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False


def render() -> None:
    st.header("Index")
    st.caption("Vectorize episodes so they can be searched across your show.")

    if not _RAG_AVAILABLE:
        st.error(
            "RAG dependencies not installed. "
            "Run: `pip install 'podcodex[rag]'` then restart the app."
        )
        return

    from utils import get_episode_paths

    show_name = st.session_state.get("show_name", "")
    output_dir = st.session_state.get("output_dir", str(Path.cwd() / "Transcriptions"))
    paths = get_episode_paths()

    with st.container(border=True):
        if not paths:
            st.info("Load an episode from the sidebar to index it.")
        else:
            _render_index_section(paths, output_dir, show_name)


def _render_index_section(paths: "AudioPaths", output_dir: str, show_name: str) -> None:
    """Render the episode indexing UI: source/model/chunker selection and index button."""
    output_dir_path = Path(output_dir)
    _paths = paths

    col_source, col_lang = st.columns(2)
    with col_source:
        source = st.selectbox(
            "Source",
            options=["polished", "transcript"],
            index=0,
            key="rag_source",
            help="'polished' uses the corrected source-language transcript (recommended).",
        )
    with col_lang:
        custom_lang = st.text_input(
            "Or use a translation",
            placeholder="language name, e.g. english",
            key="rag_custom_lang",
            help="If set, overrides the source selector.",
        )
    if custom_lang.strip():
        source = custom_lang.strip()

    show_input = st.text_input(
        "Show name",
        value=show_name,
        key="rag_index_show",
        help="All episodes from this show share one vector collection.",
    )

    model_labels = {k: v.label for k, v in MODELS.items()}
    col_models, col_chunkers = st.columns(2)
    with col_models:
        model_keys = st.multiselect(
            "Embedding models",
            options=list(MODELS.keys()),
            default=[DEFAULT_MODEL],
            format_func=lambda k: model_labels[k],
            key="rag_models",
            help="\n".join(f"**{v.label}**: {v.description}" for v in MODELS.values()),
        )
    with col_chunkers:
        _skip_diar = st.session_state.get("skip_diarization", False)
        _chunker_options = [
            k for k in CHUNKING_STRATEGIES if not (_skip_diar and k == "speaker")
        ]
        chunkings = st.multiselect(
            "Chunking strategies",
            options=_chunker_options,
            default=[DEFAULT_CHUNKING],
            format_func=lambda k: CHUNKING_STRATEGIES[k],
            key="rag_chunkings",
            help="Speaker chunking is unavailable without diarization."
            if _skip_diar
            else None,
        )

    # Semantic params (shown when "semantic" is selected)
    if "semantic" in chunkings:
        with st.expander("Semantic chunking settings", expanded=False):
            col_chunk_size, col_chunk_threshold = st.columns(2)
            with col_chunk_size:
                chunk_size = st.number_input(
                    "Max tokens per chunk",
                    min_value=64,
                    max_value=1024,
                    value=CHUNK_SIZE,
                    step=32,
                    key="rag_chunk_size",
                )
            with col_chunk_threshold:
                chunk_threshold = st.slider(
                    "Split threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(CHUNK_THRESHOLD),
                    step=0.05,
                    key="rag_chunk_threshold",
                    help="Lower = more chunks (more splits). Higher = fewer, larger chunks.",
                )
    else:
        chunk_size = CHUNK_SIZE
        chunk_threshold = CHUNK_THRESHOLD

    # Check indexed status for each (model, chunker) combination
    episode_stem = output_dir_path.name
    db_path = _paths.vectors_db
    indexed: set[str] = set()
    pending: list[str] = []
    if show_input.strip() and model_keys and chunkings:
        try:
            local = LocalStore(db_path=db_path)
            for mk in model_keys:
                for ck in chunkings:
                    coll = collection_name(show_input, model=mk, chunker=ck)
                    if local.episode_is_indexed(coll, episode_stem):
                        indexed.add(f"{model_labels[mk]} / {ck}")
                    else:
                        pending.append(f"{model_labels[mk]} / {ck}")
        except (OSError, sqlite3.Error):
            pending = [
                f"{model_labels[mk]} / {ck}" for mk in model_keys for ck in chunkings
            ]

    if indexed and not pending:
        st.success(f"All {len(indexed)} combination(s) indexed.")
        st.session_state.indexed = True
    elif indexed:
        st.info(
            f"{len(indexed)} indexed, {len(pending)} pending: " + ", ".join(pending)
        )
    elif model_keys and chunkings:
        st.info("Episode is not yet indexed.")

    col_btn, col_force = st.columns([4, 1])
    with col_force:
        overwrite = st.checkbox(
            "Overwrite",
            key="rag_overwrite",
            value=False,
            help="Delete and recreate existing collections.",
        )

    all_done = indexed and not pending
    with col_btn:
        if st.button(
            "🗂️ Index episode",
            use_container_width=True,
            type="primary",
            disabled=not model_keys
            or not chunkings
            or bool(all_done and not overwrite),
        ):
            if not show_input.strip():
                st.error("Show name is required.")
                return
            _run_indexing(
                _paths,
                output_dir,
                show_input,
                source,
                model_keys,
                chunkings,
                chunk_size,
                chunk_threshold,
                overwrite,
            )


def _run_indexing(
    paths: AudioPaths,
    output_dir: str,
    show: str,
    source: str,
    model_keys: list[str],
    chunkings: list[str],
    chunk_size: int,
    chunk_threshold: float,
    overwrite: bool,
) -> None:
    """Run the vectorization pipeline for all (model, chunker) combinations.

    Writes to LocalStore (SQLite). Search reads directly from this store.
    """
    from podcodex.cli import _resolve_source, vectorize_batch

    output_dir_path = Path(output_dir)
    episode_stem = output_dir_path.name
    _paths = paths
    transcript_path = _paths.transcript_best

    if not transcript_path.exists():
        st.error(
            f"Transcript not found: `{transcript_path.name}`. "
            "Transcribe this episode first."
        )
        return

    source_path = _resolve_source(transcript_path, source)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    transcript = data if isinstance(data, dict) else {"meta": {}, "segments": data}

    episode = transcript.get("meta", {}).get("episode") or episode_stem
    transcript.setdefault("meta", {})
    transcript["meta"].setdefault("show", show)
    transcript["meta"].setdefault("episode", episode)

    db_path = _paths.vectors_db
    local = LocalStore(db_path=db_path)

    total = len(model_keys) * len(chunkings)
    progress = st.progress(0, text="Indexing…")

    def on_progress(step: int, total: int, label: str) -> None:
        progress.progress(step / max(total, 1), text=f"{label}…")

    try:
        n = vectorize_batch(
            transcript,
            show,
            episode,
            model_keys,
            chunkings,
            local,
            chunk_size=chunk_size,
            threshold=chunk_threshold,
            overwrite=overwrite,
            on_progress=on_progress,
        )
    except Exception as e:
        progress.empty()
        st.error(f"Indexing failed: {e}")
        return

    progress.empty()
    # Touch the .rag_indexed marker so the sidebar shows ⚡
    (output_dir_path / ".rag_indexed").touch()
    st.session_state.indexed = True
    st.toast(f"Indexed {total} combination(s) ({n} chunks embedded).", icon="✅")
    st.rerun()
