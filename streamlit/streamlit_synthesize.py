"""
podcodex.ui.streamlit_synthesize — Synthesis tab
"""

import json
from pathlib import Path

import streamlit as st

from podcodex.core import synthesize


@st.cache_resource
def _load_tts_model(model_size: str):
    return synthesize.load_tts_model(model_size=model_size)


def render():
    st.header("Synthesis")
    st.caption("Clone speaker voices and synthesize a translated podcast episode.")

    # ── Import (standalone mode) ──
    if not st.session_state.get("translation"):
        with st.container(border=True):
            st.markdown("### 📥 Import")
            st.caption(
                "No data loaded. Upload a transcript or translation JSON, and optionally the original audio for voice sample extraction."
            )

            col1, col2 = st.columns(2)
            with col1:
                uploaded_json = st.file_uploader(
                    "Transcript or translation JSON",
                    type=["json"],
                    help="JSON array with 'speaker', 'start', 'end', 'text' fields. May also contain 'text_trad' if already translated. Compatible with podcodex transcript.json and translated.json formats.",
                    label_visibility="collapsed",
                )
            with col2:
                uploaded_audio = st.file_uploader(
                    "Original audio file (for voice samples)",
                    type=["mp3", "wav", "m4a", "ogg", "flac"],
                    help="The original audio file used to extract voice samples for cloning. Required if you want to clone the speakers' voices.",
                    label_visibility="collapsed",
                )

            with st.expander("📋 Expected JSON format", expanded=False):
                st.code(
                    "[{\n"
                    '  "speaker": "Alice",\n'
                    '  "start": 0.0,\n'
                    '  "end": 5.2,\n'
                    '  "text": "Original text.",\n'
                    '  "text_trad": "Translated text."  ← optional\n'
                    "}, ...]",
                    language="json",
                )
                st.caption(
                    "`text_trad` is required for synthesis. "
                    "If absent, select `text` as the field to synthesize in Configuration."
                )

            if uploaded_json:
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
                        out = Path(
                            st.session_state.get(
                                "output_dir", str(Path.cwd() / "Transcriptions")
                            )
                        )
                        stem = (
                            Path(uploaded_json.name)
                            .stem.replace(".translated", "")
                            .replace(".transcript", "")
                        )
                        episode_dir = out / stem
                        episode_dir.mkdir(parents=True, exist_ok=True)

                        if (
                            uploaded_audio
                            and st.session_state.get("audio_filename")
                            != uploaded_audio.name
                        ):
                            audio_dest = episode_dir / uploaded_audio.name
                            audio_dest.write_bytes(uploaded_audio.read())
                            st.session_state.audio_path = audio_dest
                            st.session_state.audio_filename = uploaded_audio.name
                        elif not st.session_state.get("audio_path"):
                            dummy = episode_dir / f"{stem}.imported.json"
                            dummy.write_text(
                                json.dumps(data, ensure_ascii=False), encoding="utf-8"
                            )
                            st.session_state.audio_path = dummy

                        st.session_state.output_dir = str(episode_dir)
                        st.session_state.translation = data
                        st.success(
                            f"Loaded — {len(data)} segments."
                            + (
                                " Audio ready for voice extraction."
                                if uploaded_audio
                                else " No audio uploaded — voice extraction will be skipped."
                            )
                        )
                        st.session_state.requested_tab = "synthesize"
                        st.rerun()

            st.info(
                "💡 Or complete the **🌍 Translate** tab first — data will be passed automatically."
            )

        if not st.session_state.get("translation"):
            return

    audio_path = st.session_state.get("audio_path")
    output_dir = st.session_state.get("output_dir", str(Path.cwd() / "Transcriptions"))

    if not audio_path:
        st.warning("Session lost — please reload the page and re-import your file.")
        st.session_state.translation = None
        st.session_state.requested_tab = "synthesize"
        st.rerun()
        return

    translation = st.session_state.translation

    # ── Section 1: Configuration ──
    with st.container(border=True):
        st.markdown("### ⚙️ Configuration")

        sample_seg = translation[0] if translation else {}
        available_fields = [k for k in ["text", "text_trad"] if k in sample_seg]
        extra_fields = [
            k
            for k in sample_seg
            if k not in ["speaker", "start", "end", "text", "text_trad"]
        ]
        available_fields += extra_fields

        col1, col2 = st.columns(2)
        with col1:
            text_field = st.selectbox(
                "Text field to synthesize",
                options=available_fields,
                index=available_fields.index("text_trad")
                if "text_trad" in available_fields
                else 0,
                help="Choose which text field to use for synthesis. 'text_trad' for translated text, 'text' for source language.",
            )
        with col2:
            language = st.text_input(
                "Synthesis language",
                value="English" if text_field == "text_trad" else "French",
                help="Full language name matching the text field you selected. Used by the TTS model for pronunciation.",
            )

        col3, col4 = st.columns(2)
        with col3:
            model_size = st.selectbox(
                "TTS model size",
                ["1.7B", "0.6B"],
                index=0,
                help="Larger model (1.7B) gives better quality but requires more VRAM (~8GB). Smaller (0.6B) runs on ~4GB.",
            )
        with col4:
            max_chunk_duration = st.number_input(
                "Max chunk duration (s)",
                min_value=5.0,
                max_value=60.0,
                value=20.0,
                step=5.0,
                help="Source segments shorter than this are synthesized in one TTS call. Longer segments are split at sentence boundaries into ceil(duration / max) chunks. Reduce for very long segments if you notice quality drift.",
            )

        st.caption(
            f"**{len(translation)}** segments · synthesizing field **`{text_field}`**"
        )

    # ── Section 2: Voice samples ──
    with st.container(border=True):
        st.markdown("### 🎤 Step 1 — Voice Samples")
        st.caption("Assign a voice reference clip to each speaker for cloning.")

        speaker_map = _load_speaker_map(output_dir)  # {SPEAKER_XX: human_name}
        _vs = st.session_state.get("voice_samples")
        if _vs and speaker_map:
            # If session-state samples use SPEAKER_XX keys but translation uses
            # human names (speaker_map was applied during export), remap the keys
            # so each speaker is matched to their own samples in the pool.
            translation_speakers = {
                seg.get("speaker") for seg in translation if seg.get("speaker")
            }
            if not (set(_vs.keys()) & translation_speakers):
                _vs = {speaker_map.get(k, k): v for k, v in _vs.items()}
                st.session_state.voice_samples = _vs
        voice_samples = _vs or _load_voice_samples_from_disk(
            audio_path, output_dir, translation
        )
        samples_exist = bool(voice_samples)

        has_real_audio = audio_path and Path(audio_path).suffix in {
            ".mp3",
            ".wav",
            ".m4a",
            ".ogg",
            ".flac",
        }

        # Voice mapping is always shown — custom uploads are available even without audio
        _render_voice_mapping(voice_samples, translation, output_dir)

        # Extraction from audio — secondary action, collapsed when samples already exist
        with st.expander(
            "🔧 Extract from audio file",
            expanded=bool(has_real_audio and not samples_exist),
        ):
            if not has_real_audio:
                st.info(
                    "Upload the original audio in the **Import** section above "
                    "to enable automatic voice sample extraction."
                )
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_duration = st.number_input(
                        "Min duration (s)",
                        value=3.0,
                        min_value=0.0,
                        help="Minimum clip duration to consider as a voice sample.",
                    )
                with col2:
                    max_duration = st.number_input(
                        "Max duration (s)",
                        value=0.0,
                        min_value=0.0,
                        help="Maximum clip duration. Set to 0 for no limit.",
                    )
                with col3:
                    top_k = st.number_input(
                        "Candidates per speaker",
                        value=3,
                        min_value=1,
                        max_value=10,
                        help="Number of audio clips to extract per speaker.",
                    )

                col_btn, col_force = st.columns([4, 1])
                with col_force:
                    force_samples = st.checkbox(
                        "Force",
                        key="force_samples",
                        value=False,
                        help="Re-extract voice samples even if they already exist.",
                    )
                if samples_exist and not force_samples:
                    st.info("Samples already extracted. Check **Force** to redo.")
                if st.button(
                    "Extract voice samples",
                    use_container_width=True,
                    disabled=samples_exist and not force_samples,
                ):
                    with st.spinner("Extracting voice samples..."):
                        voice_samples = synthesize.extract_voice_samples(
                            Path(audio_path),
                            translation,
                            output_dir=output_dir,
                            min_duration=min_duration or None,
                            max_duration=max_duration or None,
                            top_k=top_k,
                        )
                        st.session_state.voice_samples = voice_samples
                    st.session_state.requested_tab = "synthesize"
                    st.rerun()

    # ── Section 3: Generate segments ──
    voice_samples = voice_samples if "voice_samples" in dir() else {}
    voice_samples_for_gen = (
        st.session_state.get("voice_samples_resolved") or voice_samples
    )

    if voice_samples_for_gen:
        with st.container(border=True):
            col_title, col_force = st.columns([4, 1])
            with col_title:
                st.markdown("### 🔊 Step 2 — Generate Segments")
                st.caption(
                    "Synthesize each segment using the selected voice references."
                )
            with col_force:
                force_generate = st.checkbox(
                    "Force",
                    key="force_generate",
                    value=False,
                    help="Regenerate all segments even if they already exist.",
                )

            generated = st.session_state.get("generated")
            if not generated:
                generated = _load_generated_from_disk(output_dir, translation)
                if generated:
                    st.session_state.generated = generated

            already_generated = bool(generated)

            if already_generated and not force_generate:
                st.info(
                    f"{len(generated)} segments already generated. Check **Force** to redo all."
                )

            if st.button(
                "🚀 Generate all segments",
                use_container_width=True,
                type="primary",
                disabled=already_generated and not force_generate,
            ):
                model = _load_tts_model(model_size)
                clone_prompts = synthesize.build_clone_prompts(
                    model,
                    voice_samples_for_gen,
                    sample_index=st.session_state.get("sample_index", 0),
                )
                generated = []
                progress = st.progress(0, text="Starting...")
                segments_dir = Path(output_dir) / "tts_segments"
                segments_dir.mkdir(parents=True, exist_ok=True)

                chunk_status = st.empty()
                for i, seg in enumerate(translation):
                    seg_to_synth = {**seg, "text_trad": seg.get(text_field, "")}
                    output_path = (
                        segments_dir / f"{i:04d}_{seg.get('speaker', 'UNK')}.wav"
                    )

                    def make_on_chunk(seg_i, n_seg, speaker):
                        def on_chunk(chunk_i, n_chunks):
                            base = seg_i / n_seg
                            chunk_frac = chunk_i / n_chunks / n_seg
                            progress.progress(
                                min(base + chunk_frac, 1.0),
                                text=f"Segment {seg_i + 1}/{n_seg} · {speaker} · chunk {chunk_i}/{n_chunks}",
                            )
                            if n_chunks > 1:
                                chunk_status.caption(
                                    f"↳ chunk {chunk_i}/{n_chunks} · "
                                    f"{seg_to_synth['text_trad'][:60]}…"
                                )

                        return on_chunk

                    result = synthesize.generate_segment(
                        model,
                        seg_to_synth,
                        clone_prompts,
                        output_path,
                        language=language,
                        max_chunk_duration=max_chunk_duration,
                        on_chunk=make_on_chunk(
                            i, len(translation), seg.get("speaker", "?")
                        ),
                    )
                    if result:
                        generated.append(result)
                    chunk_status.empty()

                st.session_state.generated = generated
                st.success(f"Generated {len(generated)} segments.")
                st.session_state.requested_tab = "synthesize"
                st.rerun()

    # ── Section 4: Segment review ──
    if st.session_state.get("generated"):
        generated = st.session_state.generated
        with st.container(border=True):
            st.markdown("### ✏️ Step 3 — Review & Edit Segments")
            st.caption(
                "Listen to each segment, edit text if needed, and regenerate individually."
            )

            all_samples = st.session_state.get(
                "voice_samples_resolved"
            ) or st.session_state.get("voice_samples", {})
            pool = _build_voice_pool(all_samples)
            pool_labels = [e["label"] for e in pool]

            for i, seg in enumerate(generated):
                label = f"[{seg['start']:.1f}s] **{seg.get('speaker', '?')}** — {seg.get('text_trad', '')[:60]}..."
                with st.expander(label, expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Source text")
                        st.write(seg.get("text", ""))
                    with col2:
                        new_text = st.text_area(
                            "Synthesized text",
                            value=seg.get("text_trad", ""),
                            key=f"edit_trad_{i}",
                            height=80,
                            help="Edit the text and click Regenerate to re-synthesize this segment.",
                            label_visibility="collapsed",
                        )

                    col_voice, col_intonation = st.columns(2)
                    with col_voice:
                        speaker = seg.get("speaker", "")
                        own_label = next(
                            (e["label"] for e in pool if e["speaker"] == speaker),
                            pool_labels[0] if pool_labels else None,
                        )
                        saved_voice = seg.get("_voice_override", own_label)
                        default_voice_idx = (
                            pool_labels.index(saved_voice)
                            if saved_voice in pool_labels
                            else 0
                        )
                        voice_override = st.selectbox(
                            "Voice",
                            options=pool_labels,
                            index=default_voice_idx,
                            key=f"voice_override_{i}",
                            help="Override the voice used for this segment.",
                        )
                    with col_intonation:
                        intonation = st.text_input(
                            "Instruct",
                            value=seg.get("_intonation", ""),
                            key=f"intonation_{i}",
                            placeholder="e.g. Whisper, conspiratorial tone…",
                            help="Style instruction passed to Qwen3-TTS via the `instruct` parameter.",
                        )

                    st.audio(str(seg["audio_file"]))

                    if st.button(
                        "🔄 Regenerate",
                        key=f"regen_{i}",
                        help="Re-synthesize this segment with the current settings.",
                    ):
                        seg_updated = {
                            **seg,
                            "text_trad": new_text,
                            "_voice_override": voice_override,
                            "_intonation": intonation,
                        }

                        chosen_entry = next(
                            (e for e in pool if e["label"] == voice_override), None
                        )
                        model = _load_tts_model(model_size)
                        if chosen_entry:
                            single_sample = {
                                speaker: [
                                    {
                                        "file": chosen_entry["file"],
                                        "duration": chosen_entry["duration"],
                                        "text": chosen_entry["text"],
                                    }
                                ]
                            }
                            clone_prompts = synthesize.build_clone_prompts(
                                model, single_sample, sample_index=0
                            )
                            if chosen_entry["speaker"] != speaker:
                                clone_prompts[speaker] = clone_prompts.pop(
                                    chosen_entry["speaker"]
                                )
                        else:
                            single_sample = (
                                {speaker: all_samples[speaker]}
                                if speaker in all_samples
                                else all_samples
                            )
                            clone_prompts = synthesize.build_clone_prompts(
                                model,
                                single_sample,
                                sample_index=st.session_state.get("sample_index", 0),
                            )

                        progress_bar = st.progress(0, text="Starting…")

                        def on_chunk(chunk_i, n_chunks, _pb=progress_bar):
                            _pb.progress(
                                chunk_i / n_chunks,
                                text=f"Chunk {chunk_i}/{n_chunks}",
                            )

                        result = synthesize.generate_segment(
                            model,
                            seg_updated,
                            clone_prompts,
                            Path(seg["audio_file"]),
                            language=language,
                            instruct=intonation.strip() or None,
                            max_chunk_duration=max_chunk_duration,
                            on_chunk=on_chunk,
                        )
                        progress_bar.empty()
                        if result:
                            st.session_state.generated[i] = result
                        st.session_state.requested_tab = "synthesize"
                        st.rerun()

        # ── Section 5: Assemble ──
        with st.container(border=True):
            st.markdown("### 🎬 Step 4 — Assemble Episode")
            st.caption("Merge all synthesized segments into a final audio file.")

            col1, col2 = st.columns(2)
            with col1:
                strategy = st.radio(
                    "Assembly strategy",
                    options=["original_timing", "silence"],
                    format_func=lambda x: {
                        "original_timing": "Original timing — preserve original gaps between segments",
                        "silence": "Fixed silence — insert a fixed pause between segments",
                    }[x],
                )
            with col2:
                silence_duration = st.number_input(
                    "Silence duration (s)",
                    value=0.5,
                    min_value=0.0,
                    disabled=strategy != "silence",
                )

            audio_path_obj = Path(audio_path)
            episode_path = Path(output_dir) / f"{audio_path_obj.stem}.synthesized.wav"

            # Detect if any segment is newer than the assembled file
            needs_reassemble = False
            if episode_path.exists():
                episode_mtime = episode_path.stat().st_mtime
                newest_seg = max(
                    (
                        Path(seg["audio_file"]).stat().st_mtime
                        for seg in generated
                        if Path(seg["audio_file"]).exists()
                    ),
                    default=0,
                )
                needs_reassemble = newest_seg > episode_mtime

            if episode_path.exists():
                if needs_reassemble:
                    st.warning(
                        "⚠️ Some segments have changed since the last assembly — reassembly recommended."
                    )
                else:
                    st.info(f"Episode already assembled: **{episode_path.name}**")
                    _render_download(episode_path)

            if st.button(
                "🎙️ Assemble episode",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner("Assembling episode..."):
                    episode_path = synthesize.assemble_episode(
                        generated,
                        audio_path,
                        output_dir=output_dir,
                        strategy=strategy,
                        silence_duration=silence_duration,
                    )
                st.session_state.generated = generated
                st.session_state.requested_tab = "synthesize"
                st.rerun()


def _render_download(episode_path: Path):
    st.audio(str(episode_path))
    with open(episode_path, "rb") as f:
        st.download_button(
            label="⬇️ Download episode",
            data=f,
            file_name=episode_path.name,
            mime="audio/wav",
            use_container_width=True,
        )


def _render_voice_mapping(
    voice_samples: dict[str, list[dict]],
    translation: list,
    output_dir: str,
):
    st.markdown("**🗂️ Voice assignment — map each speaker to a reference clip:**")
    st.caption(
        "Each speaker is matched to their own samples by default. "
        "Reassign freely or upload a custom WAV."
    )

    pool = _build_voice_pool(voice_samples)

    speakers = list(
        dict.fromkeys(
            seg.get("speaker", "") for seg in translation if seg.get("speaker")
        )
    )

    resolved_mapping: dict[str, dict] = {}

    for speaker in speakers:
        own_in_pool = [e for e in pool if e["speaker"] == speaker]
        has_any_pool = bool(pool)

        # ── Custom upload ──────────────────────────────────────────────────
        # Prominent (flat) when the speaker has no samples yet; collapsed otherwise.
        if own_in_pool:
            with st.expander(f"📎 Custom upload for {speaker}", expanded=False):
                uploaded = st.file_uploader(
                    "Drop a WAV/MP3 here",
                    type=["wav", "mp3", "ogg", "flac", "m4a"],
                    key=f"voice_upload_{speaker}",
                    label_visibility="collapsed",
                )
        else:
            col_no_sample, col_upload = st.columns([2, 5])
            with col_no_sample:
                st.markdown(f"**{speaker}**")
                st.caption("No sample yet")
            with col_upload:
                uploaded = st.file_uploader(
                    f"Upload a voice sample for {speaker}",
                    type=["wav", "mp3", "ogg", "flac", "m4a"],
                    key=f"voice_upload_{speaker}",
                    label_visibility="collapsed",
                )

        # ── Register custom upload in pool ─────────────────────────────────
        if uploaded:
            custom_dir = Path(output_dir) / "voice_samples"
            custom_dir.mkdir(parents=True, exist_ok=True)
            dest = custom_dir / f"{speaker}_custom_{uploaded.name}"
            if not dest.exists():
                dest.write_bytes(uploaded.read())
                voice_samples.setdefault(speaker, []).insert(
                    0,
                    {"file": dest, "duration": _wav_duration(dest), "text": ""},
                )
                st.session_state.voice_samples = voice_samples
            custom_entry = {
                "label": f"🎙️ Custom: {uploaded.name}",
                "file": dest,
                "duration": _wav_duration(dest),
                "text": "",
                "speaker": speaker,
                "idx": -1,
            }
            speaker_pool = [custom_entry] + pool
            natural_default = custom_entry["label"]
        else:
            speaker_pool = pool
            natural_default = (
                own_in_pool[0]["label"]
                if own_in_pool
                else (pool[0]["label"] if has_any_pool else "")
            )

        # ── No sample at all — skip selectbox ─────────────────────────────
        pool_labels = [e["label"] for e in speaker_pool]
        if not pool_labels:
            st.divider()
            continue

        # ── Selectbox + player ─────────────────────────────────────────────
        col_label, col_select, col_player = st.columns([2, 3, 2])

        current_value = st.session_state.get(f"voice_select_{speaker}")
        if current_value and current_value in pool_labels:
            default_idx = pool_labels.index(current_value)
        elif natural_default in pool_labels:
            default_idx = pool_labels.index(natural_default)
        else:
            default_idx = 0

        with col_label:
            st.markdown(f"**{speaker}**")
        with col_select:
            chosen_label = st.selectbox(
                "Voice reference",
                options=pool_labels,
                index=default_idx,
                key=f"voice_select_{speaker}",
                label_visibility="collapsed",
            )
        chosen_entry = next(e for e in speaker_pool if e["label"] == chosen_label)

        with col_player:
            st.audio(str(chosen_entry["file"]))
            cross = (
                f" · from **{chosen_entry['speaker']}**"
                if chosen_entry["speaker"] != speaker
                else ""
            )
            st.caption(f"{chosen_entry['duration']:.1f}s{cross}")

        own_candidates = [
            e
            for e in speaker_pool
            if e["speaker"] == speaker and e["label"] != chosen_label
        ]
        if own_candidates:
            with st.expander(
                f"Other candidates ({len(own_candidates)})", expanded=False
            ):
                for entry in own_candidates:
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.caption(f"#{entry['idx'] + 1} · {entry['duration']:.1f}s")
                    with c2:
                        st.audio(str(entry["file"]))

        resolved_mapping[speaker] = {**chosen_entry}
        st.divider()

    st.session_state.voice_mapping = resolved_mapping
    st.session_state.voice_samples_resolved = {
        spk: [{"file": e["file"], "duration": e["duration"], "text": e["text"]}]
        for spk, e in resolved_mapping.items()
    }
    st.session_state.sample_index = {spk: 0 for spk in resolved_mapping}


def _build_voice_pool(voice_samples: dict[str, list[dict]]) -> list[dict]:
    pool = []
    for speaker, samples in voice_samples.items():
        for i, s in enumerate(samples):
            pool.append(
                {
                    "label": f"{speaker} #{i + 1} ({s['duration']:.1f}s)",
                    "file": s["file"],
                    "duration": s["duration"],
                    "text": s.get("text", ""),
                    "speaker": speaker,
                    "idx": i,
                }
            )
    return pool


def _load_speaker_map(output_dir: str) -> dict[str, str]:
    """Return {SPEAKER_XX: human_name} from the first *.speaker_map.json in output_dir."""
    for p in Path(output_dir).glob("*.speaker_map.json"):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_voice_samples_from_disk(
    audio_path, output_dir: str, translation: list
) -> dict:
    samples_dir = Path(output_dir) / "voice_samples"
    if not samples_dir.exists():
        return {}
    seen: set[str] = set()
    speakers: list[str] = []
    for seg in translation:
        sp = seg.get("speaker", "")
        if sp and sp not in seen:
            seen.add(sp)
            speakers.append(sp)

    # speaker_map lets us fall back to SPEAKER_XX-named files when the transcript
    # uses human names (i.e. the map was applied during export).
    speaker_map = _load_speaker_map(output_dir)  # {SPEAKER_XX: human_name}
    inv_map = {v: k for k, v in speaker_map.items()}  # {human_name: SPEAKER_XX}

    result = {}
    for speaker in speakers:
        files = sorted(
            f for f in samples_dir.glob(f"{speaker}_*.wav") if "_custom_" not in f.name
        )
        if not files:
            # Fallback: samples may be stored under the original SPEAKER_XX label
            speaker_id = inv_map.get(speaker)
            if speaker_id:
                files = sorted(
                    f
                    for f in samples_dir.glob(f"{speaker_id}_*.wav")
                    if "_custom_" not in f.name
                )
        if files:
            result[speaker] = [
                {"file": f, "duration": _wav_duration(f), "text": ""} for f in files
            ]
    return result


def _load_generated_from_disk(output_dir: str, translation: list) -> list:
    segments_dir = Path(output_dir) / "tts_segments"
    if not segments_dir.exists():
        return []
    result = []
    for i, seg in enumerate(translation):
        speaker = seg.get("speaker", "UNK")
        wav_path = segments_dir / f"{i:04d}_{speaker}.wav"
        if not wav_path.exists():
            return []
        try:
            import soundfile as sf

            info = sf.info(str(wav_path))
            result.append(
                {**seg, "audio_file": wav_path, "sample_rate": info.samplerate}
            )
        except Exception:
            return []
    return result


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


def _wav_duration(path: Path) -> float:
    try:
        import soundfile as sf

        return sf.info(str(path)).duration
    except Exception:
        return 0.0
