"""
podcodex.ui.streamlit_transcribe — Transcription tab
"""

import json
import os
from pathlib import Path

import streamlit as st

from podcodex.core import AudioPaths, transcribe
from podcodex.core._utils import group_by_speaker, segments_to_text
from podcodex.core.transcribe import (
    save_transcript,
    load_transcript_raw,
    load_transcript_validated,
)
from podcodex.core.synthesize import is_hallucination
from constants import (
    AUDIO_EXTENSIONS,
    WHISPER_MODELS,
    WHISPER_MODELS_LIST,
    DEFAULT_LANGUAGE_CODE,
)
from utils import fmt_time
from streamlit_editor import render_segment_editor


def _render_transcript_import(
    audio_path: str, output_dir: str, *, wrap_expander: bool = False
) -> None:
    """Render the transcript JSON upload UI. Saves to .transcript.raw.json on import."""
    uploaded_json = st.file_uploader(
        "Upload transcript JSON",
        type=["json"],
        help='JSON array or {"segments": [...]}. Each segment needs at least a "text" field.',
        key="transcribe_upload",
    )
    with st.expander("📋 Expected JSON format", expanded=False):
        st.code(
            '[{\n  "speaker": "Alice",\n  "start": 0.0,\n  "end": 5.2,\n  "text": "Hello world."\n}, ...]',
            language="json",
        )
    if uploaded_json:
        if st.button("Import", use_container_width=True, key="transcribe_import_btn"):
            try:
                raw = json.loads(uploaded_json.read().decode("utf-8"))
                data = raw.get("segments", raw) if isinstance(raw, dict) else raw
                if not isinstance(data, list) or not data:
                    st.error('Expected a JSON array or {"segments": [...]}.')
                elif "text" not in data[0]:
                    st.error("Missing 'text' field in segments.")
                else:
                    speakers = sorted({s.get("speaker", "") for s in data} - {""})
                    meta = {
                        "episode": Path(audio_path).stem,
                        "speakers": speakers,
                        "duration": round(
                            max((s.get("end", 0) for s in data), default=0.0), 3
                        ),
                        "word_count": sum(len(s.get("text", "").split()) for s in data),
                    }
                    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
                    p.transcript_raw.parent.mkdir(parents=True, exist_ok=True)
                    payload = {"meta": meta, "segments": data}
                    p.transcript_raw.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    from podcodex.core._utils import write_json

                    write_json(p.transcript, payload)
                    t_key = f"editor_transcript_{audio_path}"
                    st.session_state[t_key] = data
                    st.session_state[f"transcript_{audio_path}_source"] = "raw"
                    st.session_state.pop(f"transcript_{audio_path}_dirty", None)
                    st.session_state.transcript = data
                    st.success(f"Imported — {len(data)} segments.")
                    st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")


def _find_rss_episode():
    """Find the RSS episode matching the current episode stem. Returns (rss_ep, show_folder) or (None, "")."""
    episode_stem = st.session_state.get("episode_stem", "")
    show_folder = st.session_state.get("show_folder", "")
    if not episode_stem or not show_folder:
        return None, show_folder
    try:
        from podcodex.ingest.rss import episode_stem as rss_stem, load_feed_cache

        feed = st.session_state.get("_rss_feed") or load_feed_cache(Path(show_folder))
        if feed:
            for ep in feed:
                if rss_stem(ep) == episode_stem and ep.audio_url:
                    return ep, show_folder
    except ImportError:
        pass
    return None, show_folder


def _render_audio_container(output_dir: str) -> None:
    """Unified audio container: RSS download, upload, and player.

    Adapts to whether audio already exists on disk.
    """
    audio_path = st.session_state.get("audio_path", "")
    audio_exists = audio_path and Path(audio_path).is_file()
    rss_ep, show_folder = _find_rss_episode()

    with st.container(border=True):
        st.markdown("### 🎧 Audio")
        if audio_exists:
            st.audio(str(audio_path))
            st.caption(Path(audio_path).name)
            show_sources = st.checkbox(
                "Change audio source",
                value=False,
                key="show_audio_sources",
            )
        else:
            st.caption("No audio file yet.")
            show_sources = True

        if show_sources:
            # RSS download
            if rss_ep:
                from podcodex.ingest.rss import download_audio

                col_label, col_btn = st.columns([4, 2])
                with col_label:
                    st.markdown(f"📡 **{rss_ep.title}**")
                with col_btn:
                    if st.button(
                        "Download from RSS",
                        use_container_width=True,
                        type="primary",
                        key="transcribe_rss_download",
                    ):
                        with st.spinner("Downloading…"):
                            try:
                                dl_path = download_audio(rss_ep, Path(show_folder))
                                if dl_path:
                                    st.session_state.audio_path = str(dl_path)
                                st.toast("Audio downloaded.", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Download failed: {e}")

            # File upload
            audio_file = st.file_uploader(
                "Upload audio file",
                type=[ext.lstrip(".") for ext in AUDIO_EXTENSIONS],
                key="transcribe_audio_upload",
            )
            if audio_file:
                dest = Path(show_folder or output_dir) / audio_file.name
                if st.button(
                    "Save audio file",
                    use_container_width=True,
                    type="primary",
                    key="transcribe_audio_save",
                ):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(audio_file.read())
                    st.session_state.audio_path = str(dest)
                    st.toast(f"Saved {dest.name}.", icon="✅")
                    st.rerun()


def render() -> None:
    st.header("Transcription")
    st.caption("Transcribe, diarize and prepare your podcast episode for translation.")

    # Episodes opened without audio (transcript-only)
    if st.session_state.get("transcript_only"):
        audio_path = st.session_state.get("audio_path", "")
        output_dir = st.session_state.get("output_dir", "")
        audio_exists = audio_path and Path(audio_path).is_file()

        # Unified audio container: download / upload / player
        _render_audio_container(output_dir)

        # Import transcript
        has_transcript = bool(st.session_state.get("transcript"))
        with st.expander(
            "📄 **Import existing transcript**",
            expanded=not has_transcript,
        ):
            st.caption(
                "Already have a transcript? Import it and skip audio processing."
            )
            _render_transcript_import(audio_path, output_dir)

        # Editor
        if has_transcript:
            _render_transcript_editor(audio_path, output_dir)

        # Pipeline steps (only available when audio exists)
        if audio_exists:
            if st.checkbox(
                "Show audio processing steps",
                value=False,
                key="show_pipeline_steps_to",
            ):
                # Switch to full pipeline mode
                st.session_state.transcript_only = False
                st.rerun()
        return

    # ── Section 1: Audio & Config ──
    if st.session_state.get("audio_path"):
        output_dir = st.session_state.get("output_dir", "")
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{Path(str(st.session_state.audio_path)).name}**")
                st.caption(str(output_dir))
            with col2:
                language = st.text_input(
                    "Language",
                    value=st.session_state.get(
                        "transcribe_language", DEFAULT_LANGUAGE_CODE
                    ),
                    key="transcribe_language",
                    help="ISO 639-1 language code (e.g. 'fr', 'en').",
                )
    else:
        st.info(
            "No audio file loaded. Select an episode from the sidebar, "
            "or use **📁 Upload a single file** in the sidebar to upload one."
        )

    # ── Audio range selector ──
    # Shown after upload, before any processing. Trimming replaces audio_path
    # in session so the entire pipeline works on the selected slice.
    if st.session_state.get("audio_path"):
        _render_audio_trim(output_dir)

    if not st.session_state.get("audio_path"):
        return

    audio_path = st.session_state.audio_path
    output_dir = st.session_state.output_dir

    if Path(audio_path).is_file():
        st.audio(str(audio_path))

    # ── Diarization toggle ──
    _has_hf_token = bool(os.environ.get("HF_TOKEN"))
    # Default to skip when no HF key is available
    if "skip_diarization" not in st.session_state:
        st.session_state["skip_diarization"] = not _has_hf_token
    skip_diarization = st.checkbox(
        "Skip diarization (no speaker detection)",
        key="skip_diarization",
        help="Produce a transcript without speaker labels. "
        "Useful when you don't have a HuggingFace API key for pyannote."
        + ("" if _has_hf_token else " **No HF_TOKEN detected — defaulting to skip.**"),
    )

    # ── Status bar ──
    status = transcribe.processing_status(audio_path, output_dir=output_dir)
    _render_status(status, skip_diarization=skip_diarization)

    st.divider()

    # ── Import existing transcript ──
    already_exported = status["exported"]
    has_transcript = already_exported or st.session_state.get("transcript")
    with st.expander(
        "📂 **Import existing transcript** — skip transcription and diarization",
        expanded=not has_transcript,
    ):
        _render_transcript_import(audio_path, output_dir)

    # ── Transcript editor (shown first when transcript exists) ──
    # Auto-load transcript into session if exported but not yet loaded
    if already_exported and not st.session_state.get("transcript"):
        st.session_state.transcript = transcribe.load_transcript(
            audio_path, output_dir=output_dir
        )

    if st.session_state.get("transcript"):
        _render_transcript_editor(audio_path, output_dir)

    # ── Audio processing steps (collapsed when transcript exists) ──
    show_pipeline = st.checkbox(
        "Show audio processing steps",
        value=not has_transcript,
        key="show_pipeline_steps",
    )
    if not show_pipeline:
        return

    # ── Step 1: Transcription ──
    with st.container(border=True):
        col_title, col_force = st.columns([4, 1])
        with col_title:
            st.markdown("### 🎙️ Step 1 — Transcription")
            st.caption("WhisperX transcription with phonetic alignment.")
        with col_force:
            force_transcribe = st.checkbox(
                "Force",
                key="force_transcribe",
                value=False,
                help="Re-run transcription even if output already exists.",
            )

        model_size = st.selectbox(
            "Whisper model",
            WHISPER_MODELS_LIST,
            index=0,
            help="\n".join(f"**{k}**: {v}" for k, v in WHISPER_MODELS.items()),
        )

        if st.button(
            "Run transcription",
            use_container_width=True,
            type="primary",
            disabled=status["transcribed"] and not force_transcribe,
            help="Already transcribed. Check 'Force' to re-run."
            if status["transcribed"] and not force_transcribe
            else None,
        ):
            with st.spinner("Transcribing... this may take a few minutes."):
                transcribe.transcribe_file(
                    audio_path,
                    model_size=model_size,
                    language=language,
                    output_dir=output_dir,
                    force=force_transcribe,
                )
            st.session_state.requested_tab = "transcribe"
            st.rerun()

        if status["transcribed"]:
            with st.expander("📄 Inspect transcription results", expanded=False):
                result = transcribe.load_segments(audio_path, output_dir=output_dir)
                st.caption(
                    f"Language: **{result['language']}** · Duration: **{result['duration']:.1f}s** · **{result['num_segments']}** segments"
                )
                for seg in result["segments"][:20]:
                    st.markdown(
                        f"`{seg['start']:.2f}s → {seg['end']:.2f}s` {seg.get('text', '')}"
                    )
                if result["num_segments"] > 20:
                    st.caption(
                        f"... {result['num_segments'] - 20} more segments not shown"
                    )

    # ── Steps 2–3: Diarization & Speaker Map (skipped when skip_diarization) ──
    if not status["transcribed"]:
        return  # Nothing more to show until transcription is done

    if skip_diarization:
        st.info(
            "⏭️ Diarization skipped — all segments will be attributed to a single narrator."
        )
    else:
        with st.container(border=True):
            col_title, col_force = st.columns([4, 1])
            with col_title:
                st.markdown("### 👥 Step 2 — Diarization & Speaker Assignment")
                st.caption(
                    "Pyannote speaker diarization + WhisperX segment assignment."
                )
            with col_force:
                force_diarize = st.checkbox(
                    "Force",
                    key="force_diarize",
                    value=False,
                    help="Re-run diarization and speaker assignment even if output already exists.",
                )

            col1, col2 = st.columns(2)
            with col1:
                min_speakers = st.number_input(
                    "Min speakers",
                    min_value=1,
                    value=2,
                    disabled=status["assigned"] and not force_diarize,
                    help="Minimum number of speakers expected in the episode. Set to 1 if unsure.",
                )
            with col2:
                max_speakers = st.number_input(
                    "Max speakers",
                    min_value=1,
                    value=4,
                    disabled=status["assigned"] and not force_diarize,
                    help="Maximum number of speakers expected. Keeping this close to the actual number improves accuracy.",
                )

            _diar_disabled = not status["transcribed"] or (
                status["assigned"] and not force_diarize
            )
            _diar_help = (
                "Run transcription first."
                if not status["transcribed"]
                else "Already done. Check 'Force' to re-run."
                if status["assigned"] and not force_diarize
                else None
            )
            if st.button(
                "Run diarization & assign",
                use_container_width=True,
                type="primary",
                disabled=_diar_disabled,
                help=_diar_help,
            ):
                with st.spinner("Diarizing..."):
                    transcribe.diarize_file(
                        audio_path,
                        output_dir=output_dir,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        force=force_diarize,
                    )
                with st.spinner("Assigning speakers to segments..."):
                    transcribe.assign_speakers(
                        audio_path, output_dir=output_dir, force=force_diarize
                    )
                st.session_state.requested_tab = "transcribe"
                st.rerun()

            if status["assigned"]:
                with st.expander("🔗 Inspect diarization results", expanded=False):
                    segs = transcribe.load_diarized_segments(
                        audio_path, output_dir=output_dir
                    )
                    diar = transcribe.load_diarization(
                        audio_path, output_dir=output_dir
                    )
                    st.caption(
                        f"**{diar['num_speakers']}** speakers detected · **{len(segs)}** segments"
                    )
                    for seg in segs[:20]:
                        st.markdown(
                            f"`{seg['start']:.2f}s → {seg['end']:.2f}s` **{seg.get('speaker', '?')}** {seg.get('text', '')}"
                        )
                    if len(segs) > 20:
                        st.caption(f"... {len(segs) - 20} more segments not shown")

        with st.container(border=True):
            col_title, col_force = st.columns([4, 1])
            with col_title:
                st.markdown("### 🏷️ Step 3 — Name Speakers")
                st.caption(
                    "Listen to each speaker's segments and enter their name. "
                    "Use 🗑️ to flag all segments for a speaker for removal."
                )
            with col_force:
                force_map = st.checkbox(
                    "Force",
                    key="force_speaker_map",
                    value=False,
                    help="Re-save the speaker map even if one already exists.",
                    disabled=not status["assigned"],
                )
            if status["assigned"]:
                _render_speaker_map(audio_path, output_dir, force=force_map)
            else:
                st.info("Run diarization & speaker assignment first.")

    # ── Export ──
    # Progressive disclosure: show export only when prerequisites are met
    _export_ready = status["transcribed"] if skip_diarization else status["mapped"]
    if not _export_ready and not status["exported"]:
        return  # Don't show export until prerequisites are done

    with st.container(border=True):
        col_title, col_force = st.columns([4, 1])
        with col_title:
            step_label = "Step 2" if skip_diarization else "Step 4"
            st.markdown(f"### 📝 {step_label} — Export Transcript")
            if skip_diarization:
                st.caption(
                    "Generate the final JSON transcript (single narrator, no speaker detection)."
                )
            else:
                st.caption(
                    "Generate the final JSON transcript with resolved speaker names. Requires the speaker map to be saved first."
                )
        with col_force:
            force_export = st.checkbox(
                "Force",
                key="force_export",
                value=False,
                help="Re-run export even if transcript already exists.",
            )

        _export_disabled = status["exported"] and not force_export
        _export_help = (
            "Already exported. Check 'Force' to re-run." if _export_disabled else None
        )
        if st.button(
            "Export transcript",
            use_container_width=True,
            type="primary",
            disabled=_export_disabled,
            help=_export_help,
        ):
            with st.spinner("Exporting..."):
                transcript = transcribe.export_transcript(
                    audio_path, output_dir=output_dir, diarized=not skip_diarization
                )
                # Load the new export into the editor cache
                t_key = f"editor_transcript_{audio_path}"
                st.session_state[t_key] = transcript
                st.session_state[f"transcript_{audio_path}_source"] = "raw"
                st.session_state.pop(f"transcript_{audio_path}_dirty", None)
                st.session_state.transcript = transcript
            st.session_state.requested_tab = "transcribe"
            st.rerun()


def _render_audio_trim(output_dir: str) -> None:
    """
    Optional audio range selector. Shown once the file is uploaded.

    If the user sets a start/end range and clicks "Trim & use this range",
    ffmpeg cuts the clip, saves it as {stem}.trim_{s}_{e}.wav, and replaces
    audio_path in session state — so transcription, diarization, and synthesis
    all work on the trimmed slice.

    A "Reset to full file" button restores the original.
    """
    audio_path = Path(st.session_state.audio_path)
    trim_applied = st.session_state.get("trim_applied")

    with st.expander("✂️ **Trim audio** (optional):", expanded=False):
        col_title, col_reset = st.columns([4, 1])
        with col_title:
            if trim_applied:
                st.caption(
                    f"Working on trimmed clip: **{audio_path.name}** "
                    f"({fmt_time(trim_applied['start'])} → {fmt_time(trim_applied['end'])})"
                )
            else:
                st.caption(
                    "Optionally restrict processing to a portion of the file. "
                    "Useful for long episodes — trim once, then transcribe the slice."
                )
        with col_reset:
            if trim_applied:
                if st.button("↩ Reset", help="Restore the original full-length file."):
                    st.session_state.audio_path = Path(trim_applied["original"])
                    st.session_state.output_dir = trim_applied.get(
                        "original_output_dir", output_dir
                    )
                    st.session_state.pop("trim_applied")
                    st.rerun()

        duration = transcribe.audio_duration(audio_path)
        if duration:
            st.audio(str(audio_path))
            st.caption(f"Duration: **{fmt_time(duration)}**")

            if not trim_applied:
                col_s, col_e, col_btn = st.columns([2, 2, 1])
                with col_s:
                    start_m = st.number_input(
                        "Start min",
                        min_value=0,
                        max_value=int(duration // 60),
                        value=0,
                        key="trim_start_m",
                        label_visibility="collapsed",
                    )
                    start_s = st.number_input(
                        "Start sec",
                        min_value=0,
                        max_value=59,
                        value=0,
                        key="trim_start_s",
                        label_visibility="collapsed",
                    )
                    st.caption("Start (min / sec)")
                with col_e:
                    end_m = st.number_input(
                        "End min",
                        min_value=0,
                        max_value=int(duration // 60),
                        value=0,
                        key="trim_end_m",
                        label_visibility="collapsed",
                    )
                    end_s = st.number_input(
                        "End sec",
                        min_value=0,
                        max_value=59,
                        value=0,
                        key="trim_end_s",
                        label_visibility="collapsed",
                    )
                    st.caption("End (min / sec) — leave at 0:00 for full file")
                with col_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    do_trim = st.button(
                        "✂️ Trim", use_container_width=True, type="primary"
                    )

                t_start = start_m * 60 + start_s
                t_end = end_m * 60 + end_s

                if t_end > t_start:
                    st.caption(
                        f"▶ Selected: **{fmt_time(t_start)}** → **{fmt_time(t_end)}** "
                        f"({fmt_time(t_end - t_start)})"
                    )

                if do_trim:
                    if t_end <= t_start:
                        st.warning("End must be greater than Start.")
                    else:
                        with st.spinner("Trimming..."):
                            trimmed = transcribe.trim_audio(
                                audio_path, t_start, t_end, output_dir
                            )
                        st.session_state.trim_applied = {
                            "original": str(audio_path),
                            "original_output_dir": output_dir,
                            "start": t_start,
                            "end": t_end,
                        }
                        st.session_state.audio_path = trimmed
                        st.session_state.audio_filename = trimmed.name
                        # output_dir = the trim subfolder, so all pipeline outputs land there
                        st.session_state.output_dir = str(trimmed.parent)
                        st.rerun()
        else:
            st.warning("Could not read audio duration — ffprobe may not be available.")


def _render_status(status: dict, skip_diarization: bool = False) -> None:
    """Show the pipeline status bar, hiding diarization steps when skip_diarization."""
    labels = {
        "transcribed": "Transcribed",
        "diarized": "Diarized",
        "assigned": "Assigned",
        "mapped": "Speaker map",
        "exported": "Exported",
    }
    skip_keys = {"diarized", "assigned", "mapped"} if skip_diarization else set()
    visible = {k: v for k, v in status.items() if k not in skip_keys}
    cols = st.columns(len(visible))
    for col, (key, done) in zip(cols, visible.items()):
        with col:
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {labels[key]}")


def _render_speaker_map(audio_path: Path, output_dir: str, force: bool = False) -> None:
    """Render the speaker naming UI: text inputs, audio previews, and save button."""
    diarization = transcribe.load_diarization(audio_path, output_dir=output_dir)
    existing_map = transcribe.load_speaker_map(audio_path, output_dir=output_dir)
    speaker_ids = diarization["speakers_found"]
    episode_stem = Path(audio_path).stem  # namespace all widget keys by episode

    all_segs = _load_diarized_segments_cached(str(audio_path), output_dir)
    segs_by_speaker = group_by_speaker(all_segs)

    def _check_hallucination(seg):
        """Return (is_hallucinated, text) for a segment."""
        text = str(seg.get("text", "")).strip()
        return is_hallucination(text), text

    new_map = {}
    for speaker_id in speaker_ids:
        all_speaker_segs = sorted(
            segs_by_speaker.get(speaker_id, []),
            key=lambda s: float(s["end"]) - float(s["start"]),
            reverse=True,
        )
        clean = [s for s in all_speaker_segs if not _check_hallucination(s)[0]]
        flagged_segs = [s for s in all_speaker_segs if _check_hallucination(s)[0]]
        top_segs = (clean + flagged_segs)[:3]
        n_flagged = len(flagged_segs)

        with st.container(border=True):
            # ── Three-column row: ID | name input | remove button ──
            # col_del is rendered before col_name so the button can set the text_input's
            # session state key before the widget is instantiated (Streamlit restriction).
            col_id, col_name, col_del = st.columns([2, 4, 1])
            with col_id:
                flag_mark = f" ⚠️ {n_flagged}" if n_flagged else ""
                st.markdown(
                    f"**{speaker_id}**{flag_mark}  \n`{len(all_speaker_segs)} segs`"
                )
            with col_del:
                if st.button(
                    "🗑️",
                    key=f"remove_{episode_stem}_{speaker_id}",
                    help="Fill [remove] in name field",
                ):
                    st.session_state[f"speaker_{episode_stem}_{speaker_id}"] = (
                        "[remove]"
                    )
                    st.rerun()
            with col_name:
                name = st.text_input(
                    "Name",
                    value=existing_map.get(speaker_id, ""),
                    key=f"speaker_{episode_stem}_{speaker_id}",
                    placeholder="Enter name…",
                    label_visibility="collapsed",
                )
            new_map[speaker_id] = (
                st.session_state.get(f"speaker_{episode_stem}_{speaker_id}", name)
                or speaker_id
            )

            # ── Audio previews — collapsed ──
            # Only one speaker's audio is active at a time to avoid Streamlit media file limits.
            top_seg_ids = {id(s) for s in top_segs}
            remaining_segs = [s for s in all_speaker_segs if id(s) not in top_seg_ids]
            n_clean = len(clean)
            label = f"🎧 {speaker_id}"
            if n_clean < len(top_segs):
                label += f" — ⚠️ {len(top_segs) - n_clean} suspect"
            active_audio_key = f"_active_speaker_audio_{episode_stem}"
            with st.expander(label, expanded=False):
                active = st.session_state.get(active_audio_key)
                is_active = active in (
                    f"top_{episode_stem}_{speaker_id}",
                    f"all_{episode_stem}_{speaker_id}",
                )
                show_audio = active == f"all_{episode_stem}_{speaker_id}"

                if not is_active:
                    if st.button(
                        "🔊 Preview",
                        key=f"btn_top_{episode_stem}_{speaker_id}",
                        use_container_width=True,
                    ):
                        st.session_state[active_audio_key] = (
                            f"top_{episode_stem}_{speaker_id}"
                        )
                        st.rerun()
                else:
                    if st.button(
                        "✖ Unload",
                        key=f"btn_top_{episode_stem}_{speaker_id}",
                        use_container_width=True,
                    ):
                        st.session_state.pop(active_audio_key, None)
                        st.rerun()

                cols = st.columns(len(top_segs)) if top_segs else []
                for i, (col, seg) in enumerate(zip(cols, top_segs)):
                    dur = float(seg["end"]) - float(seg["start"])
                    is_hallucinated, text = _check_hallucination(seg)
                    with col:
                        if is_active:
                            st.audio(
                                audio_path,
                                start_time=float(seg["start"]),
                                end_time=float(seg["end"]),
                            )
                        st.caption(
                            f"#{i + 1} · {dur:.1f}s · `{fmt_time(seg['start'])}→{fmt_time(seg['end'])}`"
                        )
                        if not text:
                            st.caption("⚠️ _(no text)_")
                        elif is_hallucinated:
                            st.caption(f"⚠️ _{text[:70]}_ ← hallucination")
                        else:
                            st.caption(f"_{text[:70]}{'…' if len(text) > 70 else ''}_")

                if remaining_segs and is_active:
                    if not show_audio:
                        if st.button(
                            f"Show all segments ({len(remaining_segs)} more)",
                            key=f"btn_load_{episode_stem}_{speaker_id}",
                            use_container_width=True,
                        ):
                            st.session_state[active_audio_key] = (
                                f"all_{episode_stem}_{speaker_id}"
                            )
                            st.rerun()
                    else:
                        st.markdown(f"**Remaining {len(remaining_segs)} segments:**")
                        for j, seg in enumerate(remaining_segs[:30]):
                            dur = float(seg["end"]) - float(seg["start"])
                            is_hallucinated, text = _check_hallucination(seg)
                            flags = "⚠️ " if is_hallucinated else ""
                            col1, col2, col3 = st.columns([3, 4, 2])
                            with col1:
                                st.caption(
                                    f"`{fmt_time(seg['start'])}→{fmt_time(seg['end'])}` ({dur:.1f}s) {flags}"
                                )
                            with col2:
                                if not text:
                                    st.caption("_(no text)_")
                                elif is_hallucinated:
                                    st.caption(f"⚠️ _{text[:80]}_ ← hallucination")
                                else:
                                    st.caption(
                                        f"_{text[:80]}{'…' if len(text) > 80 else ''}_"
                                    )
                            with col3:
                                st.audio(
                                    audio_path,
                                    start_time=float(seg["start"]),
                                    end_time=float(seg["end"]),
                                )

    unnamed = [
        sp
        for sp in speaker_ids
        if not new_map.get(sp, "").strip() or new_map.get(sp) == sp
    ]
    if unnamed:
        st.caption(f"⚠️ {len(unnamed)} speaker(s) not yet named: {', '.join(unnamed)}")
    already_saved = transcribe.processing_status(audio_path, output_dir=output_dir)[
        "mapped"
    ]
    _map_disabled = bool(unnamed) or (already_saved and not force)
    _map_help = (
        f"Name all speakers first ({len(unnamed)} remaining)."
        if unnamed
        else "Already saved. Check 'Force' to re-save."
        if already_saved and not force
        else None
    )
    if st.button(
        "💾 Save speaker map",
        use_container_width=True,
        type="primary",
        disabled=_map_disabled,
        help=_map_help,
    ):
        transcribe.save_speaker_map(audio_path, new_map, output_dir=output_dir)
        st.success("Speaker map saved! You can now export the transcript.")
        st.session_state.requested_tab = "transcribe"
        st.rerun()


@st.cache_data(show_spinner=False)
def _load_diarized_segments_cached(audio_path: str, output_dir: str) -> list:
    """Cached wrapper for ``transcribe.load_diarized_segments``."""
    return transcribe.load_diarized_segments(Path(audio_path), output_dir=output_dir)


def _render_transcript_editor(audio_path: Path | str, output_dir: str) -> None:
    """Render the transcript editor with load original/edits buttons and segment editor."""
    audio_path = Path(audio_path)
    t_key = f"editor_transcript_{audio_path}"
    source_key = f"transcript_{audio_path}_source"
    if t_key not in st.session_state:
        paths_init = AudioPaths.from_audio(audio_path, output_dir=output_dir)
        if paths_init.transcript.exists():
            st.session_state[t_key] = load_transcript_validated(
                audio_path, output_dir=output_dir
            )
            st.session_state[source_key] = "edited"
        else:
            st.session_state[t_key] = load_transcript_raw(
                audio_path, output_dir=output_dir
            )
            st.session_state[source_key] = "raw"
    transcript = st.session_state[t_key]

    paths = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    _dirty = st.session_state.get(f"transcript_{audio_path}_dirty", False)
    _src = st.session_state.get(source_key, "")
    _viewing_raw = _src == "raw"

    with st.container(border=True):
        col_title, col_badge = st.columns([5, 1])
        with col_title:
            st.markdown("### ✏️ Review & Edit Transcript")
            st.caption(
                "Correct transcription errors directly. Changes are saved "
                "to the transcript file and will be used for translation."
            )
        with col_badge:
            if paths.transcript.exists() and not _dirty and not _viewing_raw:
                st.success("✅ Saved")
            elif (
                _dirty
                or _viewing_raw
                or (paths.transcript_raw.exists() and not paths.transcript.exists())
            ):
                st.warning("⚠️ Unsaved")

        # Show which version is currently loaded
        if _viewing_raw:
            if paths.transcript.exists():
                st.caption("Viewing: **original** (you have saved edits)")
            else:
                st.caption("Viewing: **original** (not yet reviewed)")
        elif _src == "edited":
            st.caption("Viewing: **saved edits**")

        # Warn if raw file is newer than validated (e.g. forced re-export)
        if (
            paths.transcript_raw.exists()
            and paths.transcript.exists()
            and paths.transcript_raw.stat().st_mtime > paths.transcript.stat().st_mtime
        ):
            st.warning(
                "The previous step was re-run after your last edits. "
                "Click **↩ Load original** to see the new version, or keep your current edits."
            )

        has_raw = paths.transcript_raw.exists()
        has_validated = paths.transcript.exists()
        if has_raw or has_validated:
            cols = st.columns(2)
            with cols[0]:
                if st.button(
                    "↩ Load original", use_container_width=True, disabled=not has_raw
                ):
                    st.session_state[t_key] = load_transcript_raw(
                        audio_path, output_dir=output_dir
                    )
                    st.session_state[source_key] = "raw"
                    st.session_state[f"transcript_{audio_path}_dirty"] = False
                    st.rerun()
            with cols[1]:
                if st.button(
                    "✏️ Load edits", use_container_width=True, disabled=not has_validated
                ):
                    st.session_state[t_key] = load_transcript_validated(
                        audio_path, output_dir=output_dir
                    )
                    st.session_state[source_key] = "edited"
                    st.session_state[f"transcript_{audio_path}_dirty"] = False
                    st.rerun()

        # ── Summary ──
        if transcript:
            speakers = sorted(
                {s.get("speaker", "") for s in transcript} - {"", "[BREAK]"}
            )
            total_chars = sum(len(s.get("text", "")) for s in transcript)
            total_words = sum(len(s.get("text", "").split()) for s in transcript)
            first_start = transcript[0].get("start")
            last_end = transcript[-1].get("end")
            if first_start is not None and last_end is not None:
                duration_str = fmt_time(last_end - first_start)
            elif Path(audio_path).is_file():
                try:
                    dur = transcribe.audio_duration(audio_path)
                    duration_str = fmt_time(dur) if dur else None
                except Exception:
                    duration_str = None
            else:
                duration_str = None
            parts = [f"**{len(transcript)}** segments"]
            if speakers:
                parts.append(f"**{len(speakers)}** speakers ({', '.join(speakers)})")
            parts.append(f"**{total_words:,}** words · **{total_chars:,}** chars")
            if duration_str:
                parts.append(f"**{duration_str}**")
            st.caption(" · ".join(parts))

        skip_diar = st.session_state.get("skip_diarization", False)
        if not transcript or not transcript[0].get("speaker"):
            if not skip_diar and not st.session_state.get("transcript_only"):
                st.warning(
                    "Transcript has no speaker info — save the speaker map and re-export first."
                )
                return

        def _on_save(merged):
            save_transcript(audio_path, merged, output_dir=output_dir)
            st.session_state[t_key] = merged
            st.session_state[source_key] = "edited"
            st.session_state.transcript = merged
            st.toast("Transcript saved!")

        has_timestamps = any(seg.get("start") is not None for seg in transcript)
        render_segment_editor(
            transcript,
            editor_key=f"transcript_{audio_path}",
            on_save=_on_save,
            audio_path=audio_path if Path(audio_path).is_file() else None,
            show_timestamps=has_timestamps,
            show_delete=True,
            show_flags=True,
            show_speaker=not skip_diar,
            diarized=not skip_diar,
            is_saved=paths.transcript.exists()
            and st.session_state.get(source_key, "") != "raw",
            export_fn=segments_to_text,
            export_filename=f"{audio_path.stem}_transcript.txt",
            next_tab="polish",
            next_tab_label="→ Go to Polish",
        )
