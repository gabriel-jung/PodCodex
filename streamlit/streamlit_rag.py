"""
podcodex.ui.streamlit_rag — Search tab (index & query)
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from podcodex.ingest.folder import find_audio
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
    from podcodex.rag.localstore import LocalStore
    from podcodex.rag.store import collection_name

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
        chunkings = st.multiselect(
            "Chunking strategies",
            options=list(CHUNKING_STRATEGIES.keys()),
            default=[DEFAULT_CHUNKING],
            format_func=lambda k: CHUNKING_STRATEGIES[k],
            key="rag_chunkings",
        )

    # Semantic params (shown when "semantic" is selected)
    if "semantic" in chunkings:
        with st.expander("Semantic chunking settings", expanded=False):
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

    # Check indexed status for each (model, chunker) combination
    episode_stem = output_dir_path.name
    db_path = output_dir_path / "vectors.db"
    indexed: set[str] = set()
    pending: list[str] = []
    if show_input.strip() and model_keys and chunkings:
        try:
            local = LocalStore(db_path=db_path)
            for mk in model_keys:
                for ck in chunkings:
                    col = collection_name(show_input, model=mk, chunker=ck)
                    if local.episode_is_indexed(col, episode_stem):
                        indexed.add(f"{model_labels[mk]} / {ck}")
                    else:
                        pending.append(f"{model_labels[mk]} / {ck}")
        except Exception:
            pending = [
                f"{model_labels[mk]} / {ck}" for mk in model_keys for ck in chunkings
            ]

    if indexed and not pending:
        st.success(f"All {len(indexed)} combination(s) indexed.")
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
                audio_path,
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
    audio_path: str,
    output_dir: str,
    show: str,
    source: str,
    model_keys: list[str],
    chunkings: list[str],
    chunk_size: int,
    chunk_threshold: float,
    overwrite: bool,
):
    from podcodex.cli import _resolve_source, vectorize_batch
    from podcodex.rag.store import QdrantStore

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

    db_path = output_dir_path / "vectors.db"
    local = LocalStore(db_path=db_path)
    store = QdrantStore()

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
            store,
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
    st.toast(f"Indexed {total} combination(s) ({n} chunks embedded).", icon="✅")
    st.rerun()


def _render_search_section(show_name: str):
    audio_path = st.session_state.get("audio_path")
    episode_stem = Path(audio_path).stem if audio_path else None

    if show_name:
        show_input = show_name
    else:
        show_input = st.text_input(
            "Show name",
            value="",
            key="rag_search_show",
            help="Name of the show collection to search.",
        )

    if episode_stem:
        episode_only = st.toggle(
            f"This episode only ({episode_stem})",
            value=True,
            key="rag_episode_only",
        )
        scope = (
            f"**{episode_stem}**"
            if episode_only
            else f"all episodes in **{show_input}**"
        )
    else:
        episode_only = False
        scope = f"all episodes in **{show_input}**" if show_input else ""

    if scope:
        st.caption(f"Searching in {scope}")

    col_query, col_btn = st.columns([6, 1])
    with col_query:
        query = st.text_input(
            "Query",
            placeholder="What was said about neural networks?",
            key="rag_search_query",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button(
            "🔍",
            use_container_width=True,
            type="primary",
            disabled=not query.strip() or not show_input.strip(),
            help="Search",
        )

    col_alpha, col_model, col_chunking, col_topk = st.columns([2, 2, 2, 1])
    with col_alpha:
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
    with col_model:
        model_labels = {k: v.label for k, v in MODELS.items()}
        model_key = st.selectbox(
            "Model",
            options=list(MODELS.keys()),
            index=list(MODELS.keys()).index(DEFAULT_MODEL),
            format_func=lambda k: model_labels[k],
            key="rag_search_model",
        )
    with col_chunking:
        chunking = st.selectbox(
            "Chunker",
            options=list(CHUNKING_STRATEGIES.keys()),
            index=list(CHUNKING_STRATEGIES.keys()).index(DEFAULT_CHUNKING),
            key="rag_search_chunking",
        )
    with col_topk:
        top_k = st.slider(
            "Results", min_value=1, max_value=20, value=TOP_K, key="rag_top_k"
        )

    if search_clicked:
        ep_filter = episode_stem if episode_only else None
        _run_search(
            query, show_input, top_k, alpha, model_key, chunking, episode=ep_filter
        )

    results = st.session_state.get("rag_results")
    if results is not None:
        if results:
            _render_results(results)
        else:
            st.info("No results. Make sure the show is indexed.")


def _run_search(
    query: str,
    show: str,
    top_k: int,
    alpha: float,
    model_key: str,
    chunking: str,
    episode: str | None = None,
):
    from podcodex.rag.retriever import Retriever

    with st.spinner("Searching…"):
        try:
            col = collection_name(show, model=model_key, chunker=chunking)
            retriever = Retriever(model=model_key)
            results = retriever.retrieve(
                query, col, top_k=top_k, alpha=alpha, episode=episode
            )
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
            score_dot = "🟢" if score > 0.8 else "🟡" if score > 0.5 else "⚪"
            st.markdown(
                f"{score_dot} **{episode}** · {fmt_time(start)}–{fmt_time(end)}"
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

            audio_path = find_audio(show_folder, episode)
            if audio_path:
                st.audio(str(audio_path), start_time=int(start))
