"""
podcodex.ui.streamlit_polish — Polish tab (source correction)
"""

import json
from pathlib import Path

import streamlit as st

from podcodex.core import transcribe
from podcodex.core import polish as polish_mod
from utils import fmt_time

# OpenAI-compatible provider presets
_PROVIDERS = {
    "Mistral": {"url": "https://api.mistral.ai/v1", "model": "mistral-small-latest"},
    "OpenAI": {"url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "Custom": {"url": "", "model": ""},
}


def _on_provider_change():
    provider = st.session_state.get("polish_api_provider", "Mistral")
    preset = _PROVIDERS.get(provider, {})
    if preset["url"]:
        st.session_state["polish_api_base_url"] = preset["url"]
        st.session_state["polish_api_model"] = preset["model"]


def render():
    st.header("Polish")
    st.caption("Correct transcription errors and proper nouns in the source language.")

    if not st.session_state.get("transcript"):
        st.info(
            "No transcript loaded. Go to the **🎙️ Transcribe** tab first, or load an episode from the sidebar."
        )
        return

    audio_path = st.session_state.get("audio_path")
    output_dir = st.session_state.get("output_dir", str(Path.cwd() / "Transcriptions"))

    if not audio_path:
        st.warning("Session lost — please reload the page.")
        st.session_state.transcript = None
        st.rerun()

    transcript = st.session_state.transcript
    simplified = transcribe.simplify_transcript(transcript)

    # ── Section 1: Configuration ──
    with st.container(border=True):
        col_title, col_force = st.columns([4, 1])
        with col_title:
            st.markdown("### ⚙️ Configuration")
            n_orig = len(transcript)
            n_simplified = len(simplified)
            if n_simplified < n_orig:
                st.caption(
                    f"**{n_orig}** segments → **{n_simplified}** after merging consecutive same-speaker segments"
                )
        with col_force:
            force = st.checkbox(
                "Force",
                key="force_polish",
                value=False,
                help="Re-run even if a polished file already exists.",
            )

        mode = st.radio(
            "Backend",
            options=["api", "ollama", "manual"],
            format_func=lambda x: {
                "api": "🌐 API",
                "ollama": "🖥️ Ollama",
                "manual": "✍️ Manual",
            }[x],
            horizontal=True,
            key="polish_mode",
        )

        source_lang = st.text_input(
            "Source language",
            value="French",
            key="polish_source_lang",
            help="Full language name of the original podcast (e.g. 'French', 'Spanish').",
        )
        if "polish_context" not in st.session_state:
            show_name = st.session_state.get("show_name", "")
            if show_name:
                st.session_state.polish_context = f"Podcast: {show_name}"
        context = st.text_area(
            "Context",
            placeholder="e.g. French podcast about film music, hosted by Alice and Bob. Names: Guillermo del Toro, Alexandre Desplat.",
            help="Optional context for the LLM — greatly improves proper noun correction.",
            height=100,
            key="polish_context",
        )

    # ── Section 2: Mode-specific settings ──
    with st.container(border=True):
        already_done = polish_mod.polished_exists(audio_path, output_dir=output_dir)
        btn_disabled = already_done and not force
        if already_done and not force:
            st.info("Polished file already exists. Check **Force** to redo it.")

        if mode == "api":
            st.markdown("### 🌐 API Settings")
            st.selectbox(
                "Provider",
                list(_PROVIDERS.keys()),
                key="polish_api_provider",
                on_change=_on_provider_change,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(
                    "Model",
                    value=st.session_state.get(
                        "polish_api_model", _PROVIDERS["Mistral"]["model"]
                    ),
                    key="polish_api_model",
                )
            with col2:
                st.text_input(
                    "API base URL",
                    value=st.session_state.get(
                        "polish_api_base_url", _PROVIDERS["Mistral"]["url"]
                    ),
                    key="polish_api_base_url",
                )
            st.text_input(
                "API key",
                type="password",
                key="polish_api_key_input",
                placeholder="Leave empty to use API_KEY from .env",
            )

            if st.button(
                "✨ Polish",
                use_container_width=True,
                type="primary",
                disabled=btn_disabled,
            ):
                kwargs = _build_kwargs(mode, source_lang, context)
                with st.spinner(f"Processing {len(simplified)} segments..."):
                    try:
                        result = polish_mod.polish_segments(simplified, **kwargs)
                        polish_mod.save_polished(
                            audio_path, result, output_dir=output_dir
                        )
                        st.session_state.polished = polish_mod.load_polished(
                            audio_path, output_dir=output_dir
                        )
                        st.success(f"Done — {len(result)} segments processed.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        elif mode == "ollama":
            st.markdown("### 🖥️ Ollama Settings")
            st.text_input(
                "Ollama model",
                value="qwen3:14b",
                key="polish_ollama_model",
                help="Run `ollama list` to see available models.",
            )
            st.caption("⚠️ Reliable JSON output requires models ≥ 14B parameters.")

            if st.button(
                "✨ Polish",
                use_container_width=True,
                type="primary",
                disabled=btn_disabled,
            ):
                kwargs = _build_kwargs(mode, source_lang, context)
                with st.spinner(f"Processing {len(simplified)} segments..."):
                    try:
                        result = polish_mod.polish_segments(simplified, **kwargs)
                        polish_mod.save_polished(
                            audio_path, result, output_dir=output_dir
                        )
                        st.session_state.polished = polish_mod.load_polished(
                            audio_path, output_dir=output_dir
                        )
                        st.success(f"Done — {len(result)} segments processed.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        elif mode == "manual":
            st.markdown("### ✍️ Manual Correction")
            st.caption(
                "Copy each prompt into any LLM (ChatGPT, Claude, etc.), paste the JSON result back."
            )

            batch_minutes = st.slider(
                "Max duration per batch (minutes)",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key="polish_batch_minutes",
            )

            batches = polish_mod.build_manual_polish_prompts_batched(
                simplified,
                batch_minutes=batch_minutes,
                context=context,
                source_lang=source_lang,
            )
            n_batches = len(batches)

            if "polish_batch_idx" not in st.session_state:
                st.session_state.polish_batch_idx = 0
            if "polish_batch_results" not in st.session_state:
                st.session_state.polish_batch_results = {}

            if st.session_state.get("polish_n_batches") != n_batches:
                st.session_state.polish_batch_idx = 0
                st.session_state.polish_batch_results = {}
                st.session_state.polish_n_batches = n_batches

            idx = st.session_state.polish_batch_idx
            done_batches = len(st.session_state.polish_batch_results)

            cols_prog = st.columns(n_batches)
            for b, col in enumerate(cols_prog):
                batch_segs, _ = batches[b]
                dur = sum(s.get("end", 0) - s.get("start", 0) for s in batch_segs)
                dur_label = fmt_time(dur)
                with col:
                    if b in st.session_state.polish_batch_results:
                        st.markdown(f"✅ **{b + 1}**")
                    elif b == idx:
                        st.markdown(f"▶ **{b + 1}**")
                    else:
                        st.markdown(f"⬜ {b + 1}")
                    st.caption(dur_label)

            st.divider()
            batch_segs, prompt = batches[idx]
            batch_dur = sum(s.get("end", 0) - s.get("start", 0) for s in batch_segs)
            st.markdown(
                f"**Batch {idx + 1} / {n_batches}** — "
                f"{len(batch_segs)} segments · "
                f"{fmt_time(batch_dur)} of audio"
            )
            st.text_area(
                "Prompt to copy",
                value=prompt,
                height=280,
                label_visibility="collapsed",
                key=f"polish_prompt_{idx}",
            )

            st.markdown("**Paste the JSON result:**")
            pasted = st.text_area(
                "JSON result",
                value="",
                height=180,
                key=f"polish_paste_{idx}",
                placeholder='[{"index": 0, "text": "corrected text..."}, ...]',
                label_visibility="collapsed",
            )

            col_prev, col_validate, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("← Prev", disabled=idx == 0, use_container_width=True):
                    st.session_state.polish_batch_idx -= 1
                    st.rerun()
            with col_validate:
                if st.button(
                    "✅ Validate batch",
                    use_container_width=True,
                    type="primary",
                    disabled=not pasted.strip(),
                ):
                    try:
                        data = json.loads(pasted)
                        validated = polish_mod.polish_segments(data, mode="manual")
                        st.session_state.polish_batch_results[idx] = validated
                        st.success(
                            f"Batch {idx + 1} validated — {len(validated)} segments."
                        )
                        if idx < n_batches - 1:
                            st.session_state.polish_batch_idx += 1
                        st.rerun()
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")
                    except ValueError as e:
                        st.error(str(e))
            with col_next:
                if st.button(
                    "Next →", disabled=idx == n_batches - 1, use_container_width=True
                ):
                    st.session_state.polish_batch_idx += 1
                    st.rerun()

            if done_batches == n_batches:
                st.divider()
                st.success(
                    f"All {n_batches} batches validated ({len(simplified)} segments total)."
                )
                if st.button("💾 Save", use_container_width=True, type="primary"):
                    all_results = []
                    for b in range(n_batches):
                        all_results.extend(st.session_state.polish_batch_results[b])
                    polish_mod.save_polished(
                        audio_path, all_results, output_dir=output_dir
                    )
                    st.session_state.polished = polish_mod.load_polished(
                        audio_path, output_dir=output_dir
                    )
                    st.session_state.polish_batch_idx = 0
                    st.session_state.polish_batch_results = {}
                    st.success(f"Saved — {len(all_results)} segments.")
                    st.rerun()

    # ── Section 3: Import existing polished file ──
    with st.container(border=True):
        st.markdown("### 📂 Import Existing Polished File")
        st.caption(
            "Already have a polished JSON? Import it to skip the correction step."
        )

        uploaded_json = st.file_uploader(
            "Upload polished JSON",
            type=["json"],
            help="JSON array with 'speaker', 'start', 'end', 'text' fields per segment.",
            label_visibility="collapsed",
            key="polish_upload",
        )
        with st.expander("📋 Expected JSON format", expanded=False):
            st.code(
                '[{\n  "speaker": "Alice",\n  "start": 0.0,\n  "end": 5.2,\n  "text": "Corrected text."\n}, ...]',
                language="json",
            )
        if uploaded_json:
            if st.button("Import", use_container_width=True, key="polish_import_btn"):
                try:
                    data = json.loads(uploaded_json.read().decode("utf-8"))
                    if not isinstance(data, list) or not data:
                        st.error("Expected a non-empty JSON array.")
                    elif "text" not in data[0]:
                        st.error("Missing 'text' field in segments.")
                    else:
                        polish_mod.save_polished(
                            audio_path, data, output_dir=output_dir
                        )
                        st.session_state.polished = polish_mod.load_polished(
                            audio_path, output_dir=output_dir
                        )
                        st.success(f"Imported — {len(data)} segments.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # ── Section 4: Preview ──
    if polish_mod.polished_exists(audio_path, output_dir=output_dir):
        _render_polish_preview(audio_path, output_dir)


def _build_kwargs(mode: str, source_lang: str, context: str) -> dict:
    kwargs = dict(mode=mode, source_lang=source_lang, context=context)
    if mode == "api":
        kwargs["model"] = st.session_state.get(
            "polish_api_model", _PROVIDERS["Mistral"]["model"]
        )
        kwargs["api_base_url"] = st.session_state.get(
            "polish_api_base_url", _PROVIDERS["Mistral"]["url"]
        )
        api_key = st.session_state.get("polish_api_key_input", "").strip()
        if api_key:
            kwargs["api_key"] = api_key
    elif mode == "ollama":
        kwargs["model"] = st.session_state.get("polish_ollama_model", "qwen3:14b")
    return kwargs


def _render_polish_preview(audio_path, output_dir: str):
    polished = polish_mod.load_polished(audio_path, output_dir=output_dir)
    st.session_state.polished = polished

    with st.container(border=True):
        col_title, col_edit = st.columns([4, 1])
        with col_title:
            st.markdown("### 👁️ Preview")
        with col_edit:
            edit_mode = st.checkbox("Edit mode", key="polish_edit_mode", value=False)

        if not edit_mode:
            preview = polish_mod.polished_to_text(polished[:10])
            st.text_area(
                "First 10 segments",
                value=preview,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
            st.caption(f"{len(polished)} segments total")
        else:
            st.caption(
                f"{len(polished)} segments — expand a segment to edit, then save."
            )
            edited = []
            for i, seg in enumerate(polished):
                lbl = f"[{seg['start']:.1f}s] **{seg.get('speaker', '?')}** — {seg.get('text', '')[:50]}..."
                with st.expander(lbl, expanded=False):
                    new_text = st.text_area(
                        "Text",
                        value=seg.get("text", ""),
                        key=f"polish_edit_{i}",
                        height=80,
                    )
                    edited.append({**seg, "text": new_text})

            if st.button(
                "💾 Save edits",
                use_container_width=True,
                type="primary",
                key="polish_save_edits",
            ):
                polish_mod.save_polished(audio_path, edited, output_dir=output_dir)
                st.session_state.polished = edited
                st.success("Saved!")
                st.rerun()

        st.divider()
        stem = Path(audio_path).stem
        st.download_button(
            "📄 Export polished",
            data=polish_mod.polished_to_text(polished),
            file_name=f"{stem}.polished.txt",
            mime="text/plain",
            use_container_width=True,
            key="polish_download",
        )

        st.divider()
        if st.button("→ Go to Translate", use_container_width=True):
            st.session_state.requested_tab = "translate"
            st.rerun()
