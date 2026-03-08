"""
podcodex.ui.streamlit_rag — Search tab (index & query)
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from utils import fmt_time

try:
    from podcodex.rag.defaults import (
        ALPHA,
        CHUNK_SIZE,
        CHUNK_THRESHOLD,
        CHUNKING_STRATEGIES,
        DEFAULT_CHUNKING,
        DEFAULT_MODEL,
        MODELS,
        TOP_K,
    )
    from podcodex.rag.store import _normalize_show

    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False


def render():
    st.header("Search")
    st.caption("Index episodes for vector search, then query across your show.")

    if not _RAG_AVAILABLE:
        st.error(
            "RAG dependencies not installed. "
            "Run: `pip install 'podcodex[rag]'` then restart the app."
        )
        return

    show_name = st.session_state.get("show_name", "")
    audio_path = st.session_state.get("audio_path")
    output_dir = st.session_state.get("output_dir", str(Path.cwd() / "Transcriptions"))

    # ── Section 1: Index ──
    with st.container(border=True):
        st.markdown("### ⚙️ Index Episode")
        if not audio_path:
            st.info("Load an episode from the sidebar to index it.")
        else:
            _render_index_section(audio_path, output_dir, show_name)

    # ── Section 2: Search ──
    with st.container(border=True):
        st.markdown("### 🔍 Search")
        _render_search_section(show_name)


def _render_index_section(audio_path: str, output_dir: str, show_name: str):
    output_dir_path = Path(output_dir)
    is_indexed = (output_dir_path / ".rag_indexed").exists()

    if is_indexed:
        st.success("Episode is indexed.")
    else:
        st.info("Episode is not yet indexed.")

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

    col_show, col_model = st.columns(2)
    with col_show:
        show_input = st.text_input(
            "Show name",
            value=show_name,
            key="rag_index_show",
            help="All episodes from this show share one vector collection.",
        )
    with col_model:
        model_labels = {k: v.label for k, v in MODELS.items()}
        model_key = st.selectbox(
            "Embedding model",
            options=list(MODELS.keys()),
            index=list(MODELS.keys()).index(DEFAULT_MODEL),
            format_func=lambda k: model_labels[k],
            key="rag_model",
            help="\n".join(f"**{v.label}**: {v.description}" for v in MODELS.values()),
        )

    with st.expander("Chunking settings", expanded=False):
        chunking = st.radio(
            "Strategy",
            options=list(CHUNKING_STRATEGIES.keys()),
            index=list(CHUNKING_STRATEGIES.keys()).index(DEFAULT_CHUNKING),
            format_func=lambda k: CHUNKING_STRATEGIES[k],
            horizontal=True,
            key="rag_chunking",
        )
        if chunking == "semantic":
            col_cs, col_ct = st.columns(2)
            with col_cs:
                chunk_size = st.number_input(
                    "Max tokens per chunk",
                    min_value=64,
                    max_value=1024,
                    value=CHUNK_SIZE,
                    step=32,
                    key="rag_chunk_size",
                )
            with col_ct:
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

    col_btn, col_force = st.columns([4, 1])
    with col_force:
        overwrite = st.checkbox(
            "Overwrite",
            key="rag_overwrite",
            value=False,
            help="Delete and recreate the collection if it already exists.",
        )

    with col_btn:
        if st.button(
            "🗂️ Index episode",
            use_container_width=True,
            type="primary",
            disabled=bool(is_indexed and not overwrite),
        ):
            if not show_input.strip():
                st.error("Show name is required.")
                return
            _run_indexing(
                audio_path,
                output_dir,
                show_input,
                source,
                model_key,
                chunking,
                chunk_size,
                chunk_threshold,
                overwrite,
            )


def _run_indexing(
    audio_path: str,
    output_dir: str,
    show: str,
    source: str,
    model_key: str,
    chunking: str,
    chunk_size: int,
    chunk_threshold: float,
    overwrite: bool,
):
    from podcodex.cli import _resolve_source
    from podcodex.rag.chunker import semantic_chunks, speaker_chunks
    from podcodex.rag.embedder import get_embedder
    from podcodex.rag.store import QdrantStore, _normalize_show

    output_dir_path = Path(output_dir)
    episode_stem = output_dir_path.name
    transcript_path = output_dir_path / f"{episode_stem}.transcript.json"

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

    with st.spinner(f"Embedding '{episode}' ({source}, {model_key})…"):
        try:
            if chunking == "speaker":
                chunks = speaker_chunks(transcript)
            else:
                chunks = semantic_chunks(
                    transcript, chunk_size=chunk_size, threshold=chunk_threshold
                )
            if not chunks:
                st.warning("No chunks produced — check the source file.")
                return

            embedder = get_embedder(model_key)
            embeddings = embedder.encode_passages(chunks)

            store = QdrantStore()
            col = _normalize_show(show)
            store.create_collection(col, model=model_key, overwrite=overwrite)
            store.upsert(col, chunks, embeddings)

            (output_dir_path / ".rag_indexed").touch()
        except Exception as e:
            st.error(f"Indexing failed: {e}")
            return

    st.toast(f"Indexed {len(chunks)} chunks into '{col}'.", icon="✅")
    st.rerun()


def _render_search_section(show_name: str):
    show_input = st.text_input(
        "Show name",
        value=show_name,
        key="rag_search_show",
        help="Name of the show collection to search.",
    )
    query = st.text_input(
        "Query",
        placeholder="What was said about neural networks?",
        key="rag_search_query",
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider(
            "Results", min_value=1, max_value=20, value=TOP_K, key="rag_top_k"
        )
    with col2:
        alpha = st.slider(
            "Search mode",
            min_value=0.0,
            max_value=1.0,
            value=float(ALPHA),
            step=0.1,
            format="%.1f",
            key="rag_alpha",
            help="0 = keyword (BM25) · 1 = semantic (vector) · 0.5 = hybrid",
        )

    if st.button(
        "🔍 Search",
        use_container_width=True,
        type="primary",
        disabled=not query.strip() or not show_input.strip(),
    ):
        _run_search(query, show_input, top_k, alpha)

    results = st.session_state.get("rag_results")
    if results is not None:
        if results:
            _render_results(results)
        else:
            st.info("No results. Make sure the show is indexed.")


def _run_search(query: str, show: str, top_k: int, alpha: float):
    from podcodex.rag.retriever import Retriever

    with st.spinner("Searching…"):
        try:
            col = _normalize_show(show)
            retriever = Retriever()
            results = retriever.retrieve(query, col, top_k=top_k, alpha=alpha)
            st.session_state.rag_results = results
        except Exception as e:
            st.error(f"Search failed: {e}")


def _render_results(results: list[dict]):
    st.divider()
    st.markdown(f"**{len(results)} result(s)**")

    show_folder = st.session_state.get("show_folder", "")

    for i, res in enumerate(results):
        score = res.get("score", 0.0)
        start = res.get("start", 0.0)
        end = res.get("end", 0.0)
        episode = res.get("episode", "")
        speakers_turns = res.get("speakers", [])

        with st.container(border=True):
            st.markdown(
                f"**{episode}** · {fmt_time(start)}–{fmt_time(end)}"
                f" <span style='float:right;color:gray'>{score:.2f}</span>",
                unsafe_allow_html=True,
            )

            # Show speaker-attributed turns if available, otherwise plain text
            if speakers_turns:
                for turn in speakers_turns:
                    st.markdown(
                        f"**{turn.get('speaker', '?')}**: {turn.get('text', '')}"
                    )
            else:
                st.write(res.get("text", ""))

            audio_path = _find_audio(show_folder, episode)
            if audio_path:
                st.audio(str(audio_path), start_time=int(start))


def _find_audio(show_folder: str, episode: str) -> "Path | None":
    """Locate the audio file for a given episode stem in the show folder."""
    if not show_folder or not episode:
        return None
    from podcodex.ingest.folder import AUDIO_EXTENSIONS

    folder = Path(show_folder)
    if not folder.is_dir():
        return None
    for ext in AUDIO_EXTENSIONS:
        candidate = folder / f"{episode}{ext}"
        if candidate.exists():
            return candidate
    return None
