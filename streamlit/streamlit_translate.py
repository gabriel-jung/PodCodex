"""
podcodex.ui.streamlit_translate — Translate tab
"""

import json
from pathlib import Path

import streamlit as st

from podcodex.core import transcribe
from podcodex.core import translate as translate_mod
from podcodex.core import polish as polish_mod
from podcodex.core.translate import (
    _translation_json,
    has_raw_translation,
    is_validated_translation,
)
from utils import fmt_time
from streamlit_editor import render_segment_editor

# OpenAI-compatible provider presets
_PROVIDERS = {
    "Mistral": {"url": "https://api.mistral.ai/v1", "model": "mistral-small-latest"},
    "OpenAI": {"url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "Custom": {"url": "", "model": ""},
}


def _on_provider_change():
    provider = st.session_state.get("trad_api_provider", "Mistral")
    preset = _PROVIDERS.get(provider, {})
    if preset["url"]:
        st.session_state["trad_api_base_url"] = preset["url"]
        st.session_state["trad_api_model"] = preset["model"]


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
                        err = _validate_segments(data)
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

    # ── Episode header ──
    with st.container(border=True):
        st.markdown(f"**{Path(str(audio_path)).name}**")
        st.caption(str(output_dir))

    transcript = st.session_state.transcript

    # ── Section 1: Source selection ──
    with st.container(border=True):
        st.markdown("### ⚙️ Configuration")

        # Offer polished as source if available
        has_polished = polish_mod.polished_exists(audio_path, output_dir=output_dir)
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
                segments = polish_mod.load_polished(audio_path, output_dir=output_dir)
                st.caption(f"Using polished transcript — {len(segments)} segments")
            else:
                segments = transcribe.simplify_transcript(transcript)
                st.caption(
                    f"Using raw transcript — {len(segments)} segments (after merging)"
                )
        else:
            segments = transcribe.simplify_transcript(transcript)
            n_orig = len(transcript)
            n_simplified = len(segments)
            if n_simplified < n_orig:
                st.caption(
                    f"**{n_orig}** segments → **{n_simplified}** after merging consecutive same-speaker segments"
                )

        col_force, _ = st.columns([1, 3])
        with col_force:
            force = st.checkbox(
                "Force",
                key="force_translate",
                value=False,
                help="Re-run translation even if a translation file already exists.",
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
                value="French",
                key="trad_source_lang",
                help="Full language name of the original podcast.",
            )
        with col2:
            target_lang = st.text_input(
                "Target language",
                value="English",
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
        already_done = translate_mod.translation_exists(
            audio_path, target_lang, output_dir=output_dir
        )
        btn_disabled = already_done and not force
        if already_done and not force:
            st.info("Translation already exists. Check **Force** to redo it.")

        if mode == "api":
            st.markdown("### 🌐 API Settings")
            st.selectbox(
                "Provider",
                list(_PROVIDERS.keys()),
                key="trad_api_provider",
                on_change=_on_provider_change,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(
                    "Model",
                    value=st.session_state.get(
                        "trad_api_model", _PROVIDERS["Mistral"]["model"]
                    ),
                    key="trad_api_model",
                )
            with col2:
                st.text_input(
                    "API base URL",
                    value=st.session_state.get(
                        "trad_api_base_url", _PROVIDERS["Mistral"]["url"]
                    ),
                    key="trad_api_base_url",
                )
            st.text_input(
                "API key",
                type="password",
                key="trad_api_key_input",
                placeholder="Leave empty to use API_KEY from .env",
            )

            if st.button(
                "🌍 Translate",
                use_container_width=True,
                type="primary",
                disabled=btn_disabled,
            ):
                kwargs = _build_kwargs(mode, source_lang, target_lang, context)
                with st.spinner(f"Processing {len(segments)} segments..."):
                    try:
                        result = translate_mod.translate_segments(segments, **kwargs)
                        _save_translation(audio_path, output_dir, result, target_lang)
                        st.success(f"Done — {len(result)} segments translated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        elif mode == "ollama":
            st.markdown("### 🖥️ Ollama Settings")
            st.text_input(
                "Ollama model",
                value="qwen3:14b",
                key="trad_ollama_model",
                help="Run `ollama list` to see available models.",
            )
            st.caption("⚠️ Reliable JSON output requires models ≥ 14B parameters.")

            if st.button(
                "🌍 Translate",
                use_container_width=True,
                type="primary",
                disabled=btn_disabled,
            ):
                kwargs = _build_kwargs(mode, source_lang, target_lang, context)
                with st.spinner(f"Processing {len(segments)} segments..."):
                    try:
                        result = translate_mod.translate_segments(segments, **kwargs)
                        _save_translation(audio_path, output_dir, result, target_lang)
                        st.success(f"Done — {len(result)} segments translated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        elif mode == "manual":
            st.markdown("### ✍️ Manual Translation")
            st.caption(
                "Copy each prompt into any LLM (ChatGPT, Claude, etc.), paste the JSON result back."
            )

            batch_minutes = st.slider(
                "Max duration per batch (minutes)",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key="trad_batch_minutes",
            )

            batches = translate_mod.build_manual_translate_prompts_batched(
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
                if st.button(
                    "✅ Validate batch",
                    use_container_width=True,
                    type="primary",
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
                    "Next →", disabled=idx == n_batches - 1, use_container_width=True
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
                    _save_translation(audio_path, output_dir, all_results, target_lang)
                    st.session_state.trad_batch_idx = 0
                    st.session_state.trad_batch_results = {}
                    st.success(f"Saved — {len(all_results)} segments.")
                    st.rerun()

    # ── Section 3: Import existing translation ──
    with st.container(border=True):
        st.markdown("### 📂 Import Existing Translation")
        st.caption("Already have a translated JSON? Import it directly.")

        import_lang = st.text_input(
            "Language",
            value="English",
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
                    err = _validate_segments(data)
                    if err:
                        st.error(f"Format error — {err}")
                    else:
                        translate_mod.save_translation_raw(
                            audio_path, data, import_lang, output_dir=output_dir
                        )
                        _reload_translations(audio_path, output_dir)
                        st.success(f"Imported — {len(data)} segments.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # ── Section 4: Editor ──
    langs = translate_mod.list_translations(audio_path, output_dir=output_dir)
    if langs:
        _render_translation_editor(audio_path, output_dir, langs, segments)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _reset_translation(audio_path, lang: str, output_dir: str) -> None:
    _translation_json(Path(audio_path), lang, output_dir).unlink(missing_ok=True)


def _validate_segments(data) -> str | None:
    if not isinstance(data, list):
        return f"Expected a JSON array, got {type(data).__name__}."
    if not data:
        return "The JSON array is empty."
    if not isinstance(data[0], dict):
        return f"Expected each element to be an object, got {type(data[0]).__name__}."
    if "text" not in data[0]:
        found = list(data[0].keys())
        return f"Missing 'text' field. Fields found: {found}."
    return None


def _build_kwargs(mode: str, source_lang: str, target_lang: str, context: str) -> dict:
    kwargs = dict(
        mode=mode, source_lang=source_lang, target_lang=target_lang, context=context
    )
    if mode == "api":
        kwargs["model"] = st.session_state.get(
            "trad_api_model", _PROVIDERS["Mistral"]["model"]
        )
        kwargs["api_base_url"] = st.session_state.get(
            "trad_api_base_url", _PROVIDERS["Mistral"]["url"]
        )
        api_key = st.session_state.get("trad_api_key_input", "").strip()
        if api_key:
            kwargs["api_key"] = api_key
    elif mode == "ollama":
        kwargs["model"] = st.session_state.get("trad_ollama_model", "qwen3:14b")
    return kwargs


def _save_translation(
    audio_path, output_dir: str, segments: list[dict], target_lang: str
) -> None:
    translate_mod.save_translation_raw(
        audio_path, segments, target_lang, output_dir=output_dir
    )
    _reload_translations(audio_path, output_dir)


def _reload_translations(audio_path, output_dir: str) -> None:
    langs = translate_mod.list_translations(audio_path, output_dir=output_dir)
    st.session_state.translations = {
        lang: translate_mod.load_translation(audio_path, lang, output_dir=output_dir)
        for lang in langs
    }
    st.session_state.translation = next(
        iter(st.session_state.translations.values()), None
    )


def _render_translation_editor(
    audio_path, output_dir: str, langs: list[str], source_segments: list[dict]
):
    stem = Path(audio_path).stem
    with st.container(border=True):
        st.markdown("### ✏️ Translations")
        tabs = st.tabs([lang.capitalize() for lang in langs])

        for tab, lang in zip(tabs, langs):
            translation = translate_mod.load_translation(
                audio_path, lang, output_dir=output_dir
            )

            def _make_save(lang=lang):
                def _on_save(merged):
                    translate_mod.save_translation(
                        audio_path, merged, lang, output_dir=output_dir
                    )
                    _reload_translations(audio_path, output_dir)
                    st.toast(f"{lang.capitalize()} translation saved!")

                return _on_save

            with tab:
                col_title, col_badge = st.columns([5, 1])
                with col_title:
                    st.caption(f"{len(translation)} segments")
                with col_badge:
                    if is_validated_translation(
                        audio_path, lang, output_dir=output_dir
                    ):
                        st.success("✅ Saved")
                    elif has_raw_translation(audio_path, lang, output_dir=output_dir):
                        st.warning("⚠️ Raw")
                render_segment_editor(
                    translation,
                    editor_key=f"translate_{audio_path}_{lang}",
                    on_save=_make_save(),
                    audio_path=audio_path,
                    reference_segments=source_segments,
                    is_saved=is_validated_translation(
                        audio_path, lang, output_dir=output_dir
                    ),
                    export_fn=translate_mod.translation_to_text,
                    export_filename=f"{stem}.{lang}.txt",
                    next_tab="synthesize" if lang == langs[-1] else None,
                    next_tab_label="→ Go to Synthesis",
                )
