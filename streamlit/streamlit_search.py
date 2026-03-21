"""
podcodex.ui.streamlit_search — Search tab (query across indexed episodes)
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from podcodex.ingest.folder import find_audio
from utils import fmt_time

try:
    from podcodex.rag.defaults import (
        ALPHA,
        CHUNKING_STRATEGIES,
        DEFAULT_CHUNKING,
        DEFAULT_MODEL,
        MODELS,
        TOP_K,
    )
    from podcodex.rag.store import collection_name, qdrant_available

    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False


def render():
    st.header("Search")
    st.caption("Query across your indexed episodes using hybrid semantic search.")

    if not _RAG_AVAILABLE:
        st.error(
            "RAG dependencies not installed. "
            "Run: `pip install 'podcodex[rag]'` then restart the app."
        )
        return

    if not st.session_state.get("_qdrant_ok"):
        st.session_state["_qdrant_ok"] = qdrant_available()

    if not st.session_state["_qdrant_ok"]:
        _render_qdrant_offline()
        return

    show_name = st.session_state.get("show_name", "")
    _render_search_section(show_name)


def _render_qdrant_offline():
    """Show Qdrant-offline state with sync button."""
    st.warning(
        "Qdrant is not reachable. Start it with `docker compose up -d` "
        "then sync your local database."
    )
    col_db, col_show = st.columns(2)
    with col_db:
        db_path = st.text_input(
            "vectors.db path",
            value=st.session_state.get("_sync_db", ""),
            key="sync_db_input",
            help="Path to the SQLite vectors.db file to sync from.",
        )
    with col_show:
        show = st.text_input(
            "Show name (optional)",
            key="sync_show_input",
            help="Sync only this show. Leave blank for all.",
        )

    col_retry, col_sync = st.columns(2)
    with col_retry:
        if st.button("🔄 Retry connection", use_container_width=True):
            st.session_state["_qdrant_ok"] = qdrant_available()
            st.rerun()
    with col_sync:
        if st.button(
            "📤 Sync to Qdrant",
            use_container_width=True,
            type="primary",
            disabled=not db_path.strip(),
        ):
            _run_sync(db_path.strip(), show.strip() or None)


def _run_sync(db_path: str, show: str | None):
    """Sync LocalStore → Qdrant."""
    import numpy as np

    from podcodex.rag.localstore import LocalStore
    from podcodex.rag.store import QdrantStore

    if not qdrant_available():
        st.error("Qdrant is still not reachable.")
        return

    with st.spinner("Syncing to Qdrant…"):
        try:
            local = LocalStore(db_path=db_path)
            store = QdrantStore()
            collections = local.list_collections()
            if show:
                # collection names are "{normalized_show}__{model}__{chunker}"
                from podcodex.rag.store import _normalize_show as norm

                prefix = norm(show) + "__"
                collections = [c for c in collections if c.startswith(prefix)]
            total = 0
            for col_name in collections:
                info = local.get_collection_info(col_name)
                model = info["model"] if info else "bge-m3"
                store.create_collection(col_name, model=model, overwrite=False)
                episodes = local.list_episodes(col_name)
                for ep in episodes:
                    cached = local.load_chunks(col_name, ep)
                    if not cached:
                        continue
                    payload = [
                        {k: v for k, v in c.items() if k != "embedding"} for c in cached
                    ]
                    embeddings = np.stack([c["embedding"] for c in cached])
                    store.delete_episode_points(col_name, ep)
                    store.upsert(col_name, payload, embeddings)
                    total += len(cached)
            st.session_state["_qdrant_ok"] = True
            st.toast(
                f"Synced {total} chunks across {len(collections)} collection(s).",
                icon="✅",
            )
            st.rerun()
        except Exception as e:
            st.error(f"Sync failed: {e}")


def _render_search_section(show_name: str):
    """Render the search UI: query input, filters, and result display."""
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
    """Execute a hybrid search and store results in session state."""
    from podcodex.rag.retriever import Retriever

    with st.spinner("Searching…"):
        try:
            coll = collection_name(show, model=model_key, chunker=chunking)
            retriever = Retriever(model=model_key)
            results = retriever.retrieve(
                query, coll, top_k=top_k, alpha=alpha, episode=episode
            )
            st.session_state.rag_results = results
        except Exception as e:
            st.error(f"Search failed: {e}")


def _render_results(results: list[dict]):
    """Display search results with score badges, speaker turns, and audio previews."""
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
