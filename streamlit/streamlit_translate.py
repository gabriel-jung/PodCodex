"""
podcodex.ui.streamlit_translate — Translation tab
"""

import json
from pathlib import Path

import streamlit as st

from podcodex.core import transcribe, translate

# OpenAI-compatible provider presets
_PROVIDERS = {
    "Mistral": {"url": "https://api.mistral.ai/v1", "model": "mistral-small-latest"},
    "OpenAI": {"url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "Custom": {"url": "", "model": ""},
}


def _on_provider_change():
    provider = st.session_state.get("api_provider", "Mistral")
    preset = _PROVIDERS.get(provider, {})
    if preset["url"]:  # don't overwrite when Custom is selected
        st.session_state["api_base_url"] = preset["url"]
        st.session_state["api_model"] = preset["model"]


def render():
    st.header("Translation")
    st.caption(
        "Translate your transcript into a target language using an LLM or manual copy/paste."
    )

    # ── Import transcript (standalone mode) ──
    if not st.session_state.get("transcript"):
        with st.container(border=True):
            st.markdown("### 📥 Import Transcript")
            st.caption(
                "No transcript loaded. Upload an existing transcript JSON or complete the **Transcribe** step first."
            )

            col1, col2 = st.columns(2)
            with col1:
                uploaded_transcript = st.file_uploader(
                    "Upload transcript JSON",
                    type=["json"],
                    help="JSON array with 'speaker', 'start', 'end', 'text' fields per segment. Compatible with podcodex transcript.json format.",
                    label_visibility="collapsed",
                )
                if uploaded_transcript:
                    try:
                        data = json.loads(uploaded_transcript.read().decode("utf-8"))
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
                        data = None
                    if data is not None:
                        err = _validate_segments_json(data, required=("text",))
                        if err:
                            st.error(f"Format error — {err}")
                        else:
                            # Set a dummy audio_path and output_dir if not already set
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
                    "💡 Or go to the **🎙️ Transcribe** tab to generate a transcript from an audio file."
                )
                with st.expander("📋 Expected JSON format", expanded=False):
                    st.code(
                        "[{\n"
                        '  "speaker": "Alice",\n'
                        '  "start": 0.0,\n'
                        '  "end": 5.2,\n'
                        '  "text": "Original text."\n'
                        "}, ...]",
                        language="json",
                    )

        if not st.session_state.get("transcript"):
            return

    audio_path = st.session_state.get("audio_path")
    output_dir = st.session_state.get("output_dir", str(Path.cwd() / "Transcriptions"))

    if not audio_path:
        st.warning(
            "Session lost — please reload the page and re-import your transcript."
        )
        # Clear transcript so import section shows again
        st.session_state.transcript = None
        st.session_state.requested_tab = "translate"
        st.rerun()

    # Load from disk if not in session
    if not st.session_state.get("transcript") and st.session_state.get("audio_path"):
        if transcribe.processing_status(audio_path, output_dir=output_dir)["exported"]:
            st.session_state.transcript = transcribe.load_transcript(
                audio_path, output_dir=output_dir
            )

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
                key="force_translate",
                value=False,
                help="Re-run translation even if a translation file already exists.",
            )

        task = st.radio(
            "Task",
            options=["translate", "polish"],
            format_func=lambda x: {
                "translate": "🌍 Translate — correct + translate",
                "polish": "✨ Polish only — correct without translating",
            }[x],
            horizontal=True,
            help="'Translate' corrects the source text and adds a translation. 'Polish only' corrects the source text without translating.",
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
            help="'API' uses a remote LLM. 'Ollama' runs a model locally. 'Manual' lets you use any external tool.",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            source_lang = st.text_input(
                "Source language",
                value="French",
                help="Full language name of the original podcast (e.g. 'French', 'Spanish').",
            )
        with col2:
            target_lang = st.text_input(
                "Target language",
                value="English",
                disabled=task == "polish",
                help="Full language name to translate into. Not used in Polish only mode.",
            )
        with col3:
            context = st.text_input(
                "Context",
                placeholder="e.g. French podcast about film music",
                help="Optional hint for the LLM — improves translation quality for domain-specific vocabulary and proper nouns.",
            )

    # ── Section 2: Mode-specific settings ──
    with st.container(border=True):
        if mode == "api":
            col_title, _ = st.columns([4, 1])
            with col_title:
                st.markdown("### 🌐 API Settings")

            st.selectbox(
                "Provider",
                list(_PROVIDERS.keys()),
                key="api_provider",
                on_change=_on_provider_change,
                help="Select a provider to auto-fill the base URL and a suggested model. Choose Custom to enter values manually.",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(
                    "Model",
                    value=st.session_state.get(
                        "api_model", _PROVIDERS["Mistral"]["model"]
                    ),
                    key="api_model",
                    help="Model identifier for the selected provider.",
                )
            with col2:
                st.text_input(
                    "API base URL",
                    value=st.session_state.get(
                        "api_base_url", _PROVIDERS["Mistral"]["url"]
                    ),
                    key="api_base_url",
                    help="Base URL of the OpenAI-compatible API endpoint.",
                )
            st.text_input(
                "API key",
                type="password",
                key="api_key_input",
                placeholder="Leave empty to use API_KEY from .env",
                help="Optional override. If empty, API_KEY from your .env file will be used.",
            )

            already_done = translate.translation_exists(
                audio_path, output_dir=output_dir
            )
            btn_disabled = already_done and not force
            if already_done and not force:
                st.info("Translation already exists. Check **Force** to redo it.")

            action_label = "🚀 Translate" if task == "translate" else "✨ Polish"
            if st.button(
                action_label,
                use_container_width=True,
                type="primary",
                disabled=btn_disabled,
            ):
                kwargs = _build_translate_kwargs(
                    mode, source_lang, target_lang, context, task=task
                )
                action = "Translating" if task == "translate" else "Polishing"
                with st.spinner(f"{action} {len(simplified)} segments..."):
                    try:
                        translated = translate.translate_segments(simplified, **kwargs)
                        translate.save_translation(
                            audio_path, translated, output_dir=output_dir
                        )
                        st.session_state.translation = translated
                        st.success(f"Done — {len(translated)} segments processed.")
                        st.session_state.requested_tab = "translate"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        elif mode == "ollama":
            col_title, _ = st.columns([4, 1])
            with col_title:
                st.markdown("### 🖥️ Ollama Settings")

            st.text_input(
                "Ollama model",
                value="qwen3:14b",
                key="ollama_model",
                help="Name of the locally installed Ollama model. Run `ollama list` to see available models.",
            )
            st.caption("⚠️ Reliable JSON output requires models ≥ 14B parameters.")

            already_done = translate.translation_exists(
                audio_path, output_dir=output_dir
            )
            btn_disabled = already_done and not force
            if already_done and not force:
                st.info("Translation already exists. Check **Force** to redo it.")

            action_label = "🚀 Translate" if task == "translate" else "✨ Polish"
            if st.button(
                action_label,
                use_container_width=True,
                type="primary",
                disabled=btn_disabled,
            ):
                kwargs = _build_translate_kwargs(
                    mode, source_lang, target_lang, context, task=task
                )
                action = "Translating" if task == "translate" else "Polishing"
                with st.spinner(f"{action} {len(simplified)} segments..."):
                    try:
                        translated = translate.translate_segments(simplified, **kwargs)
                        translate.save_translation(
                            audio_path, translated, output_dir=output_dir
                        )
                        st.session_state.translation = translated
                        st.success(f"Done — {len(translated)} segments processed.")
                        st.session_state.requested_tab = "translate"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        elif mode == "manual":
            col_title, _ = st.columns([4, 1])
            with col_title:
                st.markdown("### ✍️ Manual Translation")
                st.caption(
                    "Copy each prompt into any LLM (ChatGPT, Claude, etc.), paste the JSON result back, then move to the next batch."
                )

            batch_minutes = st.slider(
                "Max duration per batch (minutes)",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                help="Each batch will cover at most this many minutes of audio. Larger batches = fewer copy/pastes but higher risk of truncation. 15 min works well with most models, 30 min for GPT-4o / Claude.",
            )

            batches = translate.build_manual_prompts_batched(
                simplified,
                batch_minutes=batch_minutes,
                context=context,
                source_lang=source_lang,
                target_lang=target_lang,
                task=task,
            )
            n_batches = len(batches)

            # Batch stepper state
            if "manual_batch_idx" not in st.session_state:
                st.session_state.manual_batch_idx = 0
            if "manual_batch_results" not in st.session_state:
                st.session_state.manual_batch_results = {}

            # Reset stepper if batching changed
            if st.session_state.get("manual_n_batches") != n_batches:
                st.session_state.manual_batch_idx = 0
                st.session_state.manual_batch_results = {}
                st.session_state.manual_n_batches = n_batches

            idx = st.session_state.manual_batch_idx
            done_batches = len(st.session_state.manual_batch_results)

            # Progress indicator — show duration of each batch
            cols_prog = st.columns(n_batches)
            for b, col in enumerate(cols_prog):
                batch_segs, _ = batches[b]
                dur = sum(s.get("end", 0) - s.get("start", 0) for s in batch_segs)
                dur_label = f"{int(dur // 60)}:{int(dur % 60):02d}"
                with col:
                    if b in st.session_state.manual_batch_results:
                        st.markdown(f"✅ **{b + 1}**")
                    elif b == idx:
                        st.markdown(f"▶ **{b + 1}**")
                    else:
                        st.markdown(f"⬜ {b + 1}")
                    st.caption(dur_label)

            st.divider()

            # Current batch info
            batch_segs, prompt = batches[idx]
            batch_dur = sum(s.get("end", 0) - s.get("start", 0) for s in batch_segs)
            st.markdown(
                f"**Batch {idx + 1} / {n_batches}** — "
                f"{len(batch_segs)} segments · "
                f"{int(batch_dur // 60)}:{int(batch_dur % 60):02d} of audio"
            )
            st.text_area(
                "Prompt to copy",
                value=prompt,
                height=280,
                label_visibility="collapsed",
                help="Copy this into your LLM, paste the JSON result below.",
            )

            st.markdown("**Paste the JSON result:**")
            pasted = st.text_area(
                "JSON result",
                value="",
                height=180,
                key=f"manual_paste_{idx}",
                placeholder='[{"index": 0, "text": "...", "text_trad": "..."}, ...]',
                label_visibility="collapsed",
            )

            col_prev, col_validate, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("← Prev", disabled=idx == 0, use_container_width=True):
                    st.session_state.manual_batch_idx -= 1
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
                        validated = translate.translate_segments(
                            data, mode="manual", task=task
                        )
                        st.session_state.manual_batch_results[idx] = validated
                        st.success(
                            f"Batch {idx + 1} validated — {len(validated)} segments."
                        )
                        # Auto-advance to next batch
                        if idx < n_batches - 1:
                            st.session_state.manual_batch_idx += 1
                        st.rerun()
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")
                    except ValueError as e:
                        st.error(str(e))
            with col_next:
                if st.button(
                    "Next →", disabled=idx == n_batches - 1, use_container_width=True
                ):
                    st.session_state.manual_batch_idx += 1
                    st.rerun()

            # Save all when all batches done
            if done_batches == n_batches:
                st.divider()
                st.success(
                    f"All {n_batches} batches validated ({len(simplified)} segments total)."
                )
                if st.button(
                    "💾 Save translation", use_container_width=True, type="primary"
                ):
                    all_results = []
                    for b in range(n_batches):
                        all_results.extend(st.session_state.manual_batch_results[b])
                    translate.save_translation(
                        audio_path, all_results, output_dir=output_dir
                    )
                    st.session_state.translation = all_results
                    st.session_state.manual_batch_idx = 0
                    st.session_state.manual_batch_results = {}
                    st.success(f"Saved — {len(all_results)} segments.")
                    st.session_state.requested_tab = "translate"
                    st.rerun()

    # ── Section 3: Import existing translation ──
    with st.container(border=True):
        st.markdown("### 📂 Import Existing Translation")
        st.caption(
            "Already have a translated JSON file? Import it directly to skip the translation step."
        )

        uploaded_json = st.file_uploader(
            "Upload translation JSON",
            type=["json"],
            help="Must be a JSON array with 'speaker', 'text', and 'text_trad' fields per segment.",
            label_visibility="collapsed",
        )
        with st.expander("📋 Expected JSON format", expanded=False):
            st.code(
                "[{\n"
                '  "speaker": "Alice",\n'
                '  "start": 0.0,\n'
                '  "end": 5.2,\n'
                '  "text": "Original text (corrected).",\n'
                '  "text_trad": "Translated text."\n'
                "}, ...]",
                language="json",
            )
        if uploaded_json:
            if st.button("Import", use_container_width=True):
                try:
                    data = json.loads(uploaded_json.read().decode("utf-8"))
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                    data = None
                if data is not None:
                    err = _validate_segments_json(data, required=("text",))
                    if err:
                        st.error(f"Format error — {err}")
                    else:
                        try:
                            translate.save_translation(
                                audio_path, data, output_dir=output_dir
                            )
                            st.session_state.translation = data
                            st.success(f"Imported — {len(data)} segments.")
                            st.session_state.requested_tab = "translate"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Import failed: {e}")

    # ── Section 4: Preview ──
    if translate.translation_exists(audio_path, output_dir=output_dir):
        _render_translation_preview(audio_path, output_dir)


def _validate_segments_json(data, required: tuple[str, ...]) -> str | None:
    """
    Return a human-readable error string if data doesn't look like a valid
    segments array, or None if it passes basic validation.
    """
    if not isinstance(data, list):
        if isinstance(data, dict):
            keys = list(data.keys())[:6]
            hint = (
                " Looks like a raw Whisper output — use the 🎙️ Transcribe tab to export it first."
                if "segments" in data or "text" in data
                else ""
            )
            return f"Expected a JSON array but got an object with keys {keys}.{hint}"
        return f"Expected a JSON array, got {type(data).__name__}."
    if not data:
        return "The JSON array is empty."
    if not isinstance(data[0], dict):
        return f"Expected each element to be an object, got {type(data[0]).__name__}."
    missing = [f for f in required if f not in data[0]]
    if missing:
        found = list(data[0].keys())
        return (
            f"Missing required field(s) {missing} in the first segment. "
            f"Fields found: {found}. Check the format hint above."
        )
    return None


def _build_translate_kwargs(
    mode, source_lang, target_lang, context, task="translate"
) -> dict:
    kwargs = dict(
        mode=mode,
        task=task,
        source_lang=source_lang,
        target_lang=target_lang,
        context=context,
    )
    if mode == "api":
        kwargs["model"] = st.session_state.get(
            "api_model", _PROVIDERS["Mistral"]["model"]
        )
        kwargs["api_base_url"] = st.session_state.get(
            "api_base_url", _PROVIDERS["Mistral"]["url"]
        )
        api_key = st.session_state.get("api_key_input", "").strip()
        if api_key:
            kwargs["api_key"] = api_key
    elif mode == "ollama":
        kwargs["model"] = st.session_state.get("ollama_model", "qwen3:14b")
    return kwargs


def _render_translation_preview(audio_path: Path, output_dir: str):
    translation = translate.load_translation(audio_path, output_dir=output_dir)
    st.session_state.translation = translation

    with st.container(border=True):
        col_title, col_edit = st.columns([4, 1])
        with col_title:
            st.markdown("### 👁️ Translation Preview")
        with col_edit:
            edit_mode = st.checkbox(
                "Edit mode",
                key="translation_edit_mode",
                value=False,
                help="Enable editing of translated segments directly.",
            )

        if not edit_mode:
            lang = st.radio(
                "Display",
                ["both", "source", "trad"],
                format_func=lambda x: {
                    "both": "Both languages",
                    "source": "Source only",
                    "trad": "Translation only",
                }[x],
                horizontal=True,
                label_visibility="collapsed",
                help="Choose which text to display in the preview.",
            )
            preview = translate.translation_to_text(translation[:10], lang=lang)
            st.text_area(
                "First 10 segments",
                value=preview,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
            st.caption(f"{len(translation)} segments total")

        else:
            st.caption(
                f"{len(translation)} segments — expand a segment to edit, then save."
            )
            edited = []
            for i, seg in enumerate(translation):
                label = f"[{seg['start']:.1f}s] **{seg.get('speaker', '?')}** — {seg.get('text', '')[:50]}..."
                with st.expander(label, expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_text = st.text_area(
                            "Source",
                            value=seg.get("text", ""),
                            key=f"trad_src_{i}",
                            height=80,
                            help="Source text in the translation file — edit if needed.",
                        )
                    with col2:
                        new_trad = st.text_area(
                            "Translation",
                            value=seg.get("text_trad", ""),
                            key=f"trad_edit_{i}",
                            height=80,
                            help="Translated text — edit if needed.",
                        )
                    edited.append({**seg, "text": new_text, "text_trad": new_trad})

            if st.button("💾 Save edits", use_container_width=True, type="primary"):
                translate.save_translation(audio_path, edited, output_dir=output_dir)
                st.session_state.translation = edited
                st.success("Translation saved!")
                st.session_state.requested_tab = "translate"
                st.rerun()

        if st.button("→ Go to Synthesis", use_container_width=True):
            st.session_state.requested_tab = "synthesize"
            st.rerun()
