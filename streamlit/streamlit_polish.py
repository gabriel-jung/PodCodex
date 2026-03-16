"""
podcodex.ui.streamlit_polish — Polish tab (source correction)
"""

import json
from pathlib import Path

import streamlit as st

from podcodex.core import AudioPaths, polish as polish_mod
from podcodex.core._utils import segments_to_text
from podcodex.core.polish import (
    load_polished_raw,
    load_polished_validated,
)
from utils import PROVIDERS, build_llm_kwargs, fmt_time, on_provider_change
from streamlit_editor import render_segment_editor


def _run_polish_button(
    btn_disabled, mode, source_lang, context, transcript, audio_path, output_dir
):
    """Render the 'Polish' action button and run the pipeline on click.

    Shared by API and Ollama modes — only the mode-specific settings differ.
    """
    if st.button(
        "✨ Polish",
        use_container_width=True,
        type="primary",
        disabled=btn_disabled,
        help="Already polished. Check 'Force' to re-run." if btn_disabled else None,
    ):
        kwargs = build_llm_kwargs(
            "polish", mode, source_lang=source_lang, context=context
        )
        with st.spinner(f"Processing {len(transcript)} segments..."):
            try:
                result = polish_mod.polish_segments(transcript, **kwargs)
                polish_mod.save_polished_raw(audio_path, result, output_dir=output_dir)
                st.session_state.polished = polish_mod.load_polished(
                    audio_path, output_dir=output_dir
                )
                st.success(f"Done — {len(result)} segments processed.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")


def render():
    st.header("Polish")
    st.caption("Correct transcription errors and proper nouns in the source language.")

    audio_path = st.session_state.get("audio_path")
    output_dir = st.session_state.get("output_dir", str(Path.cwd() / "Transcriptions"))

    if not audio_path:
        st.info(
            "No episode loaded. Load one from the sidebar or go to the **🎙️ Transcribe** tab."
        )
        return

    paths = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    # ── Episode header ──
    with st.container(border=True):
        st.markdown(f"**{Path(str(audio_path)).name}**")
        st.caption(str(output_dir))

    if not st.session_state.get("transcript"):
        with st.container(border=True):
            st.markdown("### 📄 Load Transcript")
            st.caption("No transcript in session. Upload a JSON transcript to proceed.")
            uploaded_json = st.file_uploader(
                "Upload transcript JSON",
                type=["json"],
                key="polish_transcript_upload",
                label_visibility="collapsed",
            )
            if uploaded_json:
                try:
                    data = json.loads(uploaded_json.read().decode("utf-8"))
                    segs = data["segments"] if isinstance(data, dict) else data
                    if not segs or "text" not in segs[0]:
                        st.error("Missing 'text' field in segments.")
                    else:
                        st.session_state.transcript = segs
                        st.success(f"Loaded {len(segs)} segments.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        return

    transcript = st.session_state.transcript

    # ── Import existing polished file ──
    already_polished = paths.has_polished()
    with st.expander(
        "📂 **Import existing polished file** — skip the correction step",
        expanded=not already_polished and not st.session_state.get("polished"),
    ):
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
                        polish_mod.save_polished_raw(
                            audio_path, data, output_dir=output_dir
                        )
                        st.session_state.polished = polish_mod.load_polished(
                            audio_path, output_dir=output_dir
                        )
                        st.success(f"Imported — {len(data)} segments.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # ── Section 1: Configuration ──
    with st.container(border=True):
        col_title, col_force = st.columns([4, 1])
        with col_title:
            st.markdown("### ⚙️ Step 1 — Configuration")
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
        already_done = paths.has_polished()
        btn_disabled = already_done and not force

        if mode == "api":
            st.markdown("### 🌐 Step 2 — API Polish")
            st.selectbox(
                "Provider",
                list(PROVIDERS.keys()),
                key="polish_api_provider",
                on_change=lambda: on_provider_change("polish"),
            )
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(
                    "Model",
                    value=st.session_state.get(
                        "polish_api_model", PROVIDERS["Mistral"]["model"]
                    ),
                    key="polish_api_model",
                )
            with col2:
                st.text_input(
                    "API base URL",
                    value=st.session_state.get(
                        "polish_api_base_url", PROVIDERS["Mistral"]["url"]
                    ),
                    key="polish_api_base_url",
                )
            st.text_input(
                "API key",
                type="password",
                key="polish_api_key_input",
                placeholder="Leave empty to use API_KEY from .env",
            )

            _run_polish_button(
                btn_disabled,
                mode,
                source_lang,
                context,
                transcript,
                audio_path,
                output_dir,
            )

        elif mode == "ollama":
            st.markdown("### 🖥️ Step 2 — Ollama Polish")
            st.text_input(
                "Ollama model",
                value="qwen3:14b",
                key="polish_ollama_model",
                help="Run `ollama list` to see available models.",
            )
            st.caption("⚠️ Reliable JSON output requires models ≥ 14B parameters.")

            _run_polish_button(
                btn_disabled,
                mode,
                source_lang,
                context,
                transcript,
                audio_path,
                output_dir,
            )

        elif mode == "manual":
            st.markdown("### ✍️ Step 2 — Manual Correction")
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

            batches = polish_mod.build_manual_prompts_batched(
                transcript,
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
                        validated = polish_mod.polish_segments(
                            data, mode="manual", original_segments=batch_segs
                        )
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
                    f"All {n_batches} batches validated ({len(transcript)} segments total)."
                )
                if st.button("💾 Save", use_container_width=True, type="primary"):
                    all_results = []
                    for b in range(n_batches):
                        all_results.extend(st.session_state.polish_batch_results[b])
                    polish_mod.save_polished_raw(
                        audio_path, all_results, output_dir=output_dir
                    )
                    st.session_state.polished = polish_mod.load_polished(
                        audio_path, output_dir=output_dir
                    )
                    st.session_state.polish_batch_idx = 0
                    st.session_state.polish_batch_results = {}
                    st.success(f"Saved — {len(all_results)} segments.")
                    st.rerun()

    # ── Section 3: Editor ──
    if paths.has_polished():
        p_key = f"editor_polished_{audio_path}"
        if p_key not in st.session_state:
            st.session_state[p_key] = polish_mod.load_polished(
                audio_path, output_dir=output_dir
            )
        polished = st.session_state[p_key]
        st.session_state.polished = polished
        with st.container(border=True):
            col_title, col_badge = st.columns([5, 1])
            with col_title:
                st.markdown("### ✏️ Step 3 — Review & Edit")
            with col_badge:
                _dirty = st.session_state.get(f"polish_{audio_path}_dirty", False)
                if paths.has_validated_polished() and not _dirty:
                    st.success("✅ Saved")
                elif paths.has_raw_polished() or _dirty:
                    st.warning("⚠️ Unsaved")

            has_raw = paths.polished_raw_exists()
            has_validated = paths.has_validated_polished()
            cols = st.columns(2)
            with cols[0]:
                if st.button(
                    "↩ Load original",
                    key="load_raw_polished",
                    use_container_width=True,
                    disabled=not has_raw,
                ):
                    st.session_state[p_key] = load_polished_raw(
                        audio_path, output_dir=output_dir
                    )
                    st.session_state[f"polish_{audio_path}_dirty"] = False
                    st.rerun()
            with cols[1]:
                if st.button(
                    "✏️ Load edits",
                    key="load_edited_polished",
                    use_container_width=True,
                    disabled=not has_validated,
                ):
                    st.session_state[p_key] = load_polished_validated(
                        audio_path, output_dir=output_dir
                    )
                    st.session_state[f"polish_{audio_path}_dirty"] = False
                    st.rerun()

            def _on_save(merged):
                polish_mod.save_polished(audio_path, merged, output_dir=output_dir)
                st.session_state[p_key] = merged
                st.session_state.polished = merged
                st.toast("Polished transcript saved!")

            render_segment_editor(
                polished,
                editor_key=f"polish_{audio_path}",
                on_save=_on_save,
                audio_path=audio_path,
                reference_segments=transcript,
                is_saved=paths.has_validated_polished(),
                export_fn=segments_to_text,
                export_filename=f"{Path(audio_path).stem}.polished.txt",
                next_tab="translate",
                next_tab_label="→ Go to Translate",
            )
