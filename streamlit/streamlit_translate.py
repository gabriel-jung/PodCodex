"""
podcodex.ui.streamlit_translate — Translate tab
"""

import json
from pathlib import Path

import streamlit as st

from podcodex.core import AudioPaths, transcribe
from podcodex.core import translate as translate_mod
from podcodex.core import polish as polish_mod
from podcodex.core._utils import segments_to_text
from podcodex.core.translate import (
    load_translation_raw,
    load_translation_validated,
)
from podcodex.core import validate_segments_json
from constants import DEFAULT_OLLAMA_MODEL, DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG
from utils import PROVIDERS, build_llm_kwargs, fmt_time, on_provider_change
from streamlit_editor import render_segment_editor


def _run_translate_button(
    btn_disabled,
    mode,
    source_lang,
    target_lang,
    context,
    segments,
    audio_path,
    output_dir,
):
    """Render the 'Translate' action button and run the pipeline on click.

    Shared by API and Ollama modes — only the mode-specific settings differ.
    """
    if st.button(
        "🌍 Translate",
        use_container_width=True,
        type="primary",
        disabled=btn_disabled,
        help="Already translated. Check 'Force' to re-run." if btn_disabled else None,
    ):
        kwargs = build_llm_kwargs(
            "trad",
            mode,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
        )
        with st.spinner(f"Processing {len(segments)} segments..."):
            try:
                result = translate_mod.translate_segments(segments, **kwargs)
                _save_translation(audio_path, output_dir, result, target_lang)
                # Load the new raw into the editor cache
                t_key = f"editor_translate_{audio_path}_{target_lang}"
                st.session_state[t_key] = result
                st.session_state[f"translate_{audio_path}_{target_lang}_source"] = "raw"
                st.session_state.pop(
                    f"translate_{audio_path}_{target_lang}_dirty", None
                )
                st.success(f"Done — {len(result)} segments translated.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")


def render():
    st.header("Translate")
    st.caption(
        "Translate a transcript to another language. Use the ✨ Polish tab first to correct the source."
    )

    # ── Import transcript (standalone mode) ──
    if not st.session_state.get("transcript"):
        with st.container(border=True):
            st.markdown("### 📥 Import Transcript")
            st.caption(
                "No transcript loaded. Upload a transcript JSON or complete the **Transcribe** step first."
            )
            col1, col2 = st.columns(2)
            with col1:
                uploaded_transcript = st.file_uploader(
                    "Upload transcript JSON",
                    type=["json"],
                    help="JSON array with 'speaker', 'start', 'end', 'text' fields.",
                    label_visibility="collapsed",
                )
                if uploaded_transcript:
                    try:
                        data = json.loads(uploaded_transcript.read().decode("utf-8"))
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
                        data = None
                    if data is not None:
                        err = validate_segments_json(data)
                        if err:
                            st.error(f"Format error — {err}")
                        else:
                            if not st.session_state.get("audio_path"):
                                out = Path(
                                    st.session_state.get(
                                        "output_dir", str(Path.cwd() / "Transcriptions")
                                    )
                                )
                                out.mkdir(parents=True, exist_ok=True)
                                st.session_state.output_dir = str(out)
                                stem = Path(uploaded_transcript.name).stem.replace(
                                    ".transcript", ""
                                )
                                dummy = out / f"{stem}.imported.json"
                                dummy.write_text(
                                    json.dumps(data, ensure_ascii=False),
                                    encoding="utf-8",
                                )
                                st.session_state.audio_path = dummy
                            st.session_state.transcript = data
                            st.success(f"Loaded — {len(data)} segments.")
                            st.session_state.requested_tab = "translate"
                            st.rerun()
            with col2:
                st.info(
                    "💡 Or go to the **🎙️ Transcribe** tab to generate a transcript."
                )

        if not st.session_state.get("transcript"):
            return

    audio_path = st.session_state.get("audio_path")
    output_dir = st.session_state.get("output_dir", str(Path.cwd() / "Transcriptions"))

    if not audio_path:
        st.warning("Session lost — please reload the page.")
        st.session_state.transcript = None
        st.rerun()

    nodiar = st.session_state.get("skip_diarization", False)
    paths = AudioPaths.from_audio(audio_path, output_dir=output_dir, nodiar=nodiar)

    # ── Episode header ──
    with st.container(border=True):
        st.markdown(f"**{Path(str(audio_path)).name}**")
        st.caption(str(output_dir))

    transcript = st.session_state.transcript

    # ── Import existing translation ──
    existing_langs = translate_mod.list_translations(
        audio_path, output_dir=output_dir, nodiar=nodiar
    )
    with st.expander(
        "📂 **Import existing translation** — skip the translation step",
        expanded=not existing_langs,
    ):
        import_lang = st.text_input(
            "Language",
            value=DEFAULT_TARGET_LANG,
            key="trad_import_lang",
            help="Language name used in the filename, e.g. 'English' → episode.english.json.",
        )
        uploaded_json = st.file_uploader(
            "Upload translation JSON",
            type=["json"],
            help="JSON array with 'speaker', 'start', 'end', 'text' fields (text = translation).",
            label_visibility="collapsed",
            key="trad_upload",
        )
        with st.expander("📋 Expected JSON format", expanded=False):
            st.code(
                '[{\n  "speaker": "Alice",\n  "start": 0.0,\n  "end": 5.2,\n  "text": "Translated text."\n}, ...]',
                language="json",
            )
        if uploaded_json:
            if st.button("Import", use_container_width=True, key="trad_import_btn"):
                try:
                    data = json.loads(uploaded_json.read().decode("utf-8"))
                    err = validate_segments_json(data)
                    if err:
                        st.error(f"Format error — {err}")
                    else:
                        translate_mod.save_translation_raw(
                            audio_path,
                            data,
                            import_lang,
                            output_dir=output_dir,
                            nodiar=nodiar,
                        )
                        t_key = f"editor_translate_{audio_path}_{import_lang}"
                        st.session_state[t_key] = data
                        st.session_state[
                            f"translate_{audio_path}_{import_lang}_source"
                        ] = "raw"
                        st.session_state.pop(
                            f"translate_{audio_path}_{import_lang}_dirty", None
                        )
                        _reload_translations(audio_path, output_dir)
                        st.success(f"Imported — {len(data)} segments.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # ── Section 1: Source selection ──
    with st.container(border=True):
        col_title, col_force = st.columns([4, 1])
        with col_title:
            st.markdown("### ⚙️ Step 1 — Configuration")
        with col_force:
            force = st.checkbox(
                "Force",
                key="force_translate",
                value=False,
                help="Re-run translation even if a translation file already exists.",
            )

        # Offer polished as source if available
        has_polished = paths.has_polished()
        if has_polished:
            source_choice = st.radio(
                "Source",
                ["polished", "raw"],
                format_func=lambda x: {
                    "polished": "✨ Polished transcript (recommended)",
                    "raw": "🎙️ Raw transcript",
                }[x],
                horizontal=True,
                key="trad_source_choice",
                help="Use the polished transcript if you've already corrected the source.",
            )
            if source_choice == "polished":
                segments = polish_mod.load_polished(
                    audio_path, output_dir=output_dir, nodiar=nodiar
                )
                st.caption(f"Using polished transcript — {len(segments)} segments")
            else:
                segments = transcribe.merge_consecutive_segments(transcript)
                st.caption(f"Using raw transcript — {len(segments)} segments")
        else:
            segments = transcribe.merge_consecutive_segments(transcript)
            n_orig = len(transcript)
            n_simplified = len(segments)
            if n_simplified < n_orig:
                st.caption(
                    f"**{n_orig}** segments → **{n_simplified}** (consecutive same-speaker segments consolidated)"
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
            key="trad_mode",
        )

        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.text_input(
                "Source language",
                value=DEFAULT_SOURCE_LANG,
                key="trad_source_lang",
                help="Full language name of the original podcast.",
            )
        with col2:
            target_lang = st.text_input(
                "Target language",
                value=DEFAULT_TARGET_LANG,
                key="trad_target_lang",
                help="Full language name to translate into.",
            )
        if "trad_context" not in st.session_state:
            show_name = st.session_state.get("show_name", "")
            if show_name:
                st.session_state.trad_context = f"Podcast: {show_name}"
        context = st.text_area(
            "Context",
            placeholder="e.g. French podcast about film music, hosted by Alice and Bob.",
            help="Optional hint for the LLM.",
            height=100,
            key="trad_context",
        )

    # ── Section 2: Mode-specific settings ──
    with st.container(border=True):
        already_done = paths.has_translation(target_lang)
        btn_disabled = already_done and not force
        if already_done and not force:
            st.info("Translation already exists. Check **Force** to redo it.")

        if mode == "api":
            st.markdown("### 🌐 Step 2 — API Translation")
            st.selectbox(
                "Provider",
                list(PROVIDERS.keys()),
                key="trad_api_provider",
                on_change=lambda: on_provider_change("trad"),
            )
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(
                    "Model",
                    value=st.session_state.get(
                        "trad_api_model", PROVIDERS["Mistral"]["model"]
                    ),
                    key="trad_api_model",
                )
            with col2:
                st.text_input(
                    "API base URL",
                    value=st.session_state.get(
                        "trad_api_base_url", PROVIDERS["Mistral"]["url"]
                    ),
                    key="trad_api_base_url",
                )
            st.text_input(
                "API key",
                type="password",
                key="trad_api_key_input",
                placeholder="Leave empty to use API_KEY from .env",
            )

            _run_translate_button(
                btn_disabled,
                mode,
                source_lang,
                target_lang,
                context,
                segments,
                audio_path,
                output_dir,
            )

        elif mode == "ollama":
            st.markdown("### 🖥️ Step 2 — Ollama Translation")
            st.text_input(
                "Ollama model",
                value=DEFAULT_OLLAMA_MODEL,
                key="trad_ollama_model",
                help="Run `ollama list` to see available models.",
            )
            st.caption("⚠️ Reliable JSON output requires models ≥ 14B parameters.")

            _run_translate_button(
                btn_disabled,
                mode,
                source_lang,
                target_lang,
                context,
                segments,
                audio_path,
                output_dir,
            )

        elif mode == "manual":
            st.markdown("### ✍️ Step 2 — Manual Translation")
            st.caption(
                "Copy each prompt into any LLM (ChatGPT, Claude, etc.), paste the JSON result back."
            )

            batch_minutes = st.slider(
                "Max duration per batch (minutes)",
                min_value=5,
                max_value=180,
                value=15,
                step=5,
                key="trad_batch_minutes",
                disabled=btn_disabled,
            )

            if not btn_disabled:
                batches = translate_mod.build_manual_prompts_batched(
                    segments,
                    batch_minutes=batch_minutes,
                    context=context,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                n_batches = len(batches)

                if "trad_batch_idx" not in st.session_state:
                    st.session_state.trad_batch_idx = 0
                if "trad_batch_results" not in st.session_state:
                    st.session_state.trad_batch_results = {}

                if st.session_state.get("trad_n_batches") != n_batches:
                    st.session_state.trad_batch_idx = 0
                    st.session_state.trad_batch_results = {}
                    st.session_state.trad_n_batches = n_batches

                idx = st.session_state.trad_batch_idx
                done_batches = len(st.session_state.trad_batch_results)

                cols_prog = st.columns(n_batches)
                for b, col in enumerate(cols_prog):
                    batch_segs, _ = batches[b]
                    dur = sum(s.get("end", 0) - s.get("start", 0) for s in batch_segs)
                    with col:
                        if b in st.session_state.trad_batch_results:
                            st.markdown(f"✅ **{b + 1}**")
                        elif b == idx:
                            st.markdown(f"▶ **{b + 1}**")
                        else:
                            st.markdown(f"⬜ {b + 1}")
                        st.caption(fmt_time(dur))

                st.divider()
                batch_segs, prompt = batches[idx]
                batch_dur = sum(s.get("end", 0) - s.get("start", 0) for s in batch_segs)
                st.markdown(
                    f"**Batch {idx + 1} / {n_batches}** — "
                    f"{len(batch_segs)} segments · {fmt_time(batch_dur)} of audio"
                )
                st.text_area(
                    "Prompt to copy",
                    value=prompt,
                    height=280,
                    label_visibility="collapsed",
                    key=f"trad_prompt_{idx}",
                )

                st.markdown("**Paste the JSON result:**")
                pasted = st.text_area(
                    "JSON result",
                    value="",
                    height=180,
                    key=f"trad_paste_{idx}",
                    placeholder='[{"index": 0, "text": "translated text..."}, ...]',
                    label_visibility="collapsed",
                )

                col_prev, col_validate, col_next = st.columns([1, 3, 1])
                with col_prev:
                    if st.button("← Prev", disabled=idx == 0, use_container_width=True):
                        st.session_state.trad_batch_idx -= 1
                        st.rerun()
                with col_validate:
                    batch_done = idx in st.session_state.trad_batch_results
                    if st.button(
                        "✅ Validated" if batch_done else "✅ Validate batch",
                        use_container_width=True,
                        type="secondary" if batch_done else "primary",
                        disabled=not pasted.strip(),
                    ):
                        try:
                            data = json.loads(pasted)
                            validated = translate_mod.translate_segments(
                                data, mode="manual", original_segments=batch_segs
                            )
                            st.session_state.trad_batch_results[idx] = validated
                            st.success(
                                f"Batch {idx + 1} validated — {len(validated)} segments."
                            )
                            if idx < n_batches - 1:
                                st.session_state.trad_batch_idx += 1
                            st.rerun()
                        except json.JSONDecodeError as e:
                            st.error(f"Invalid JSON: {e}")
                        except ValueError as e:
                            st.error(str(e))
                with col_next:
                    if st.button(
                        "Next →",
                        disabled=idx == n_batches - 1,
                        use_container_width=True,
                    ):
                        st.session_state.trad_batch_idx += 1
                        st.rerun()

                if done_batches == n_batches:
                    st.divider()
                    st.success(
                        f"All {n_batches} batches validated ({len(segments)} segments total)."
                    )
                    if st.button("💾 Save", use_container_width=True, type="primary"):
                        all_results = []
                        for b in range(n_batches):
                            all_results.extend(st.session_state.trad_batch_results[b])
                        _save_translation(
                            audio_path, output_dir, all_results, target_lang
                        )
                        t_key = f"editor_translate_{audio_path}_{target_lang}"
                        st.session_state[t_key] = all_results
                        st.session_state[
                            f"translate_{audio_path}_{target_lang}_source"
                        ] = "raw"
                        st.session_state.pop(
                            f"translate_{audio_path}_{target_lang}_dirty", None
                        )
                        st.session_state.trad_batch_idx = 0
                        st.session_state.trad_batch_results = {}
                        st.success(f"Saved — {len(all_results)} segments.")
                        st.rerun()

    # ── Section 3: Editor ──
    langs = translate_mod.list_translations(
        audio_path, output_dir=output_dir, nodiar=nodiar
    )
    if langs:
        _render_translation_editor(audio_path, output_dir, langs, segments)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _save_translation(
    audio_path, output_dir: str, segments: list[dict], target_lang: str
) -> None:
    """Save translation as raw and refresh the session-state translations cache."""
    _nd = st.session_state.get("skip_diarization", False)
    translate_mod.save_translation_raw(
        audio_path, segments, target_lang, output_dir=output_dir, nodiar=_nd
    )
    _reload_translations(audio_path, output_dir)


def _reload_translations(audio_path, output_dir: str) -> None:
    """Rescan disk for translations and update session state."""
    _nd = st.session_state.get("skip_diarization", False)
    langs = translate_mod.list_translations(
        audio_path, output_dir=output_dir, nodiar=_nd
    )
    st.session_state.translations = {
        lang: translate_mod.load_translation(
            audio_path, lang, output_dir=output_dir, nodiar=_nd
        )
        for lang in langs
    }
    st.session_state.translation = next(
        iter(st.session_state.translations.values()), None
    )


def _render_translation_editor(
    audio_path, output_dir: str, langs: list[str], source_segments: list[dict]
):
    """Render one tab per language with a segment editor, load/save buttons, and badges."""
    stem = Path(audio_path).stem
    _nd = st.session_state.get("skip_diarization", False)
    paths = AudioPaths.from_audio(audio_path, output_dir=output_dir, nodiar=_nd)
    with st.container(border=True):
        st.markdown("### ✏️ Step 3 — Review & Edit")
        tabs = st.tabs([lang.capitalize() for lang in langs])

        for tab, lang in zip(tabs, langs):
            t_key = f"editor_translate_{audio_path}_{lang}"
            source_key = f"translate_{audio_path}_{lang}_source"
            if t_key not in st.session_state:
                if paths.has_validated_translation(lang):
                    st.session_state[t_key] = load_translation_validated(
                        audio_path, lang, output_dir=output_dir, nodiar=_nd
                    )
                    st.session_state[source_key] = "edited"
                else:
                    st.session_state[t_key] = load_translation_raw(
                        audio_path, lang, output_dir=output_dir, nodiar=_nd
                    )
                    st.session_state[source_key] = "raw"
            translation = st.session_state[t_key]

            def _make_save(lang=lang, t_key=t_key, source_key=source_key):
                """Build a save callback bound to a specific language and cache key."""

                def _on_save(merged):
                    _nd2 = st.session_state.get("skip_diarization", False)
                    translate_mod.save_translation(
                        audio_path, merged, lang, output_dir=output_dir, nodiar=_nd2
                    )
                    st.session_state[t_key] = merged
                    st.session_state[source_key] = "edited"
                    # Update session-state cache directly instead of rescanning disk
                    st.session_state.translations[lang] = merged
                    st.session_state.translation = merged
                    st.toast(f"{lang.capitalize()} translation saved!")

                return _on_save

            with tab:
                col_title, col_badge = st.columns([5, 1])
                with col_title:
                    st.caption(f"{len(translation)} segments")
                with col_badge:
                    _dirty = st.session_state.get(
                        f"translate_{audio_path}_{lang}_dirty", False
                    )
                    _src = st.session_state.get(source_key, "")
                    _viewing_raw = _src == "raw"
                    if (
                        paths.has_validated_translation(lang)
                        and not _dirty
                        and not _viewing_raw
                    ):
                        st.success("✅ Saved")
                    elif _dirty or _viewing_raw or paths.has_raw_translation(lang):
                        st.warning("⚠️ Unsaved")
                if _viewing_raw:
                    if paths.has_validated_translation(lang):
                        st.caption("Viewing: **original** (you have saved edits)")
                    else:
                        st.caption("Viewing: **original** (not yet reviewed)")
                elif _src == "edited":
                    st.caption("Viewing: **saved edits**")

                # Warn if raw file is newer than validated (e.g. forced re-run)
                has_raw = paths.translation_raw_exists(lang)
                has_validated = paths.has_validated_translation(lang)
                if (
                    has_raw
                    and has_validated
                    and paths.translation_raw(lang).stat().st_mtime
                    > paths.translation(lang).stat().st_mtime
                ):
                    st.warning(
                        "The previous step was re-run after your last edits. "
                        "Click **↩ Load original** to see the new version, or keep your current edits."
                    )
                cols = st.columns(2)
                with cols[0]:
                    if st.button(
                        "↩ Load original",
                        key=f"load_raw_{lang}",
                        use_container_width=True,
                        disabled=not has_raw,
                    ):
                        st.session_state[t_key] = load_translation_raw(
                            audio_path, lang, output_dir=output_dir, nodiar=_nd
                        )
                        st.session_state[source_key] = "raw"
                        st.session_state[f"translate_{audio_path}_{lang}_dirty"] = False
                        st.rerun()
                with cols[1]:
                    if st.button(
                        "✏️ Load edits",
                        key=f"load_edited_{lang}",
                        use_container_width=True,
                        disabled=not has_validated,
                    ):
                        st.session_state[t_key] = load_translation_validated(
                            audio_path, lang, output_dir=output_dir, nodiar=_nd
                        )
                        st.session_state[source_key] = "edited"
                        st.session_state[f"translate_{audio_path}_{lang}_dirty"] = False
                        st.rerun()

                render_segment_editor(
                    translation,
                    editor_key=f"translate_{audio_path}_{lang}",
                    on_save=_make_save(),
                    audio_path=audio_path,
                    reference_segments=source_segments,
                    is_saved=paths.has_validated_translation(lang) and not _viewing_raw,
                    export_fn=segments_to_text,
                    export_filename=f"{stem}.{lang}.txt",
                    next_tab="synthesize" if lang == langs[-1] else None,
                    next_tab_label="→ Go to Synthesis",
                )
