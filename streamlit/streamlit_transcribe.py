"""
podcodex.ui.streamlit_transcribe — Transcription tab
"""

from pathlib import Path

import streamlit as st

from podcodex.core import transcribe
from podcodex.core.transcribe import (
    save_transcript,
    has_raw_transcript,
    is_validated_transcript,
    transcript_raw_exists,
    load_transcript_raw,
    load_transcript_validated,
)
from podcodex.core.synthesize import is_hallucination
from streamlit_editor import render_segment_editor, audio_slice_bytes


def render():
    st.header("Transcription")
    st.caption("Transcribe, diarize and prepare your podcast episode for translation.")

    # ── Section 1: Audio & Config ──
    if st.session_state.get("audio_path"):
        # Episode loaded from sidebar — just show language param
        audio_path_loaded = st.session_state.audio_path
        output_dir = st.session_state.get("output_dir", "")
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{Path(str(audio_path_loaded)).name}**")
                st.caption(str(output_dir))
            with col2:
                language = st.text_input(
                    "Language",
                    value=st.session_state.get("transcribe_language", "fr"),
                    key="transcribe_language",
                    help="ISO 639-1 language code (e.g. 'fr', 'en').",
                )
    else:
        with st.container(border=True):
            st.markdown("### 📁 Audio File")

            uploaded = st.file_uploader(
                "Upload audio file",
                type=["mp3", "wav", "m4a", "ogg", "flac"],
                help="Supported formats: MP3, WAV, M4A, OGG, FLAC. The file will be saved to the output directory.",
            )

            default_output = st.session_state.get("show_folder") or str(
                Path.cwd() / "Transcriptions"
            )
            col1, col2 = st.columns(2)
            with col1:
                output_dir = st.text_input(
                    "Output directory",
                    value=default_output,
                    help="Absolute path where all outputs will be saved. A subfolder named after the episode will be created automatically.",
                )
                output_dir = str(Path(output_dir).resolve())
                st.session_state.output_dir = output_dir
                st.session_state.base_output_dir = output_dir
            with col2:
                language = st.text_input(
                    "Audio language",
                    value="fr",
                    key="transcribe_language",
                    help="ISO 639-1 language code of the podcast (e.g. 'fr' for French, 'en' for English). Used by WhisperX for transcription.",
                )

            # Save uploaded file into base_output_dir/{stem}/
            if uploaded and st.session_state.get("audio_filename") != uploaded.name:
                stem = Path(uploaded.name).stem
                base_output = st.session_state.get(
                    "base_output_dir", str(Path(output_dir).resolve())
                )
                save_dir = Path(base_output)
                save_dir.mkdir(parents=True, exist_ok=True)
                audio_dest = save_dir / uploaded.name
                audio_dest.write_bytes(uploaded.read())
                ep_output_dir = save_dir / stem
                ep_output_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.audio_path = audio_dest
                st.session_state.audio_filename = uploaded.name
                st.session_state.output_dir = str(ep_output_dir)
                st.session_state.base_output_dir = base_output
                # Reset any previous trim when a new file is loaded
                st.session_state.pop("trim_applied", None)
                st.session_state.requested_tab = "transcribe"
                st.rerun()

    # ── Audio range selector ──
    # Shown after upload, before any processing. Trimming replaces audio_path
    # in session so the entire pipeline works on the selected slice.
    if st.session_state.get("audio_path"):
        _render_audio_trim(output_dir)

    if not st.session_state.get("audio_path"):
        return

    audio_path = st.session_state.audio_path
    output_dir = st.session_state.output_dir

    st.audio(str(audio_path))

    # ── Status bar ──
    status = transcribe.processing_status(audio_path, output_dir=output_dir)
    _render_status(status)

    st.divider()

    # ── Section 2: Transcription ──
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
            ["large-v3", "medium", "small"],
            index=0,
            help="Larger models are more accurate but slower and require more VRAM. 'large-v3' requires ~10GB VRAM, 'medium' ~5GB, 'small' ~2GB.",
        )

        if st.button(
            "Run transcription",
            use_container_width=True,
            type="primary",
            disabled=status["transcribed"] and not force_transcribe,
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
                result = transcribe.load_transcription(
                    audio_path, output_dir=output_dir
                )
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

    # ── Section 3: Diarization ──
    with st.container(border=True):
        col_title, col_force = st.columns([4, 1])
        with col_title:
            st.markdown("### 👥 Step 2 — Diarization & Speaker Assignment")
            st.caption("Pyannote speaker diarization + WhisperX segment assignment.")
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

        if st.button(
            "Run diarization & assign",
            use_container_width=True,
            type="primary",
            disabled=not status["transcribed"]
            or (status["assigned"] and not force_diarize),
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
                transcribe.assign_speakers_to_file(
                    audio_path, output_dir=output_dir, force=force_diarize
                )
            st.session_state.requested_tab = "transcribe"
            st.rerun()

        if status["assigned"]:
            with st.expander("🔗 Inspect diarization results", expanded=False):
                segs = transcribe.load_diarized_segments(
                    audio_path, output_dir=output_dir
                )
                diar = transcribe.load_diarization(audio_path, output_dir=output_dir)
                st.caption(
                    f"**{diar['num_speakers']}** speakers detected · **{len(segs)}** segments"
                )
                for seg in segs[:20]:
                    st.markdown(
                        f"`{seg['start']:.2f}s → {seg['end']:.2f}s` **{seg.get('speaker', '?')}** {seg.get('text', '')}"
                    )
                if len(segs) > 20:
                    st.caption(f"... {len(segs) - 20} more segments not shown")

    # ── Section 4: Speaker map ──
    if status["assigned"]:
        with st.container(border=True):
            col_title, col_force = st.columns([4, 1])
            with col_title:
                st.markdown("### 🏷️ Step 4 — Name Speakers")
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
                )
            _render_speaker_map(audio_path, output_dir, force=force_map)

    # ── Section 5: Export ──
    with st.container(border=True):
        col_title, col_force = st.columns([4, 1])
        with col_title:
            st.markdown("### 📝 Step 5 — Export Transcript")
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

        if st.button(
            "Export transcript",
            use_container_width=True,
            type="primary",
            disabled=not status["mapped"] or (status["exported"] and not force_export),
        ):
            with st.spinner("Exporting..."):
                transcript = transcribe.export_transcript(
                    audio_path, output_dir=output_dir
                )
                st.session_state.transcript = transcript
            st.session_state.requested_tab = "transcribe"
            st.rerun()

        if not status["mapped"] and not status["exported"]:
            st.info("Save the speaker map above before exporting.")

    # ── Section 7: Transcript editor ──
    if st.session_state.get("transcript"):
        with st.container(border=True):
            col_title, col_badge = st.columns([5, 1])
            with col_title:
                st.markdown("### ✏️ Step 6 — Review & Edit Transcript")
                st.caption(
                    "Correct transcription errors directly. Changes are saved to the transcript file and will be used for translation."
                )
            with col_badge:
                _dirty = st.session_state.get(f"transcript_{audio_path}_dirty", False)
                if (
                    is_validated_transcript(audio_path, output_dir=output_dir)
                    and not _dirty
                ):
                    st.success("✅ Saved")
                elif has_raw_transcript(audio_path, output_dir=output_dir) or _dirty:
                    st.warning("⚠️ Unsaved")
            _render_transcript_editor(audio_path, output_dir)


def _render_audio_trim(output_dir: str):
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
                    f"({_fmt_mmss(trim_applied['start'])} → {_fmt_mmss(trim_applied['end'])})"
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

        duration = _audio_duration_tc(audio_path)
        if duration:
            st.audio(str(audio_path))
            st.caption(f"Duration: **{_fmt_mmss(duration)}**")

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
                        f"▶ Selected: **{_fmt_mmss(t_start)}** → **{_fmt_mmss(t_end)}** "
                        f"({_fmt_mmss(t_end - t_start)})"
                    )

                if do_trim:
                    if t_end <= t_start:
                        st.warning("End must be greater than Start.")
                    else:
                        with st.spinner("Trimming..."):
                            trimmed = _trim_audio(
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


def _audio_duration_tc(path: Path) -> float | None:
    """Return duration in seconds, trying soundfile then ffprobe."""
    try:
        import soundfile as sf

        return sf.info(str(path)).duration
    except Exception:
        pass
    try:
        import subprocess
        import json

        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None


def _fmt_mmss(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _trim_audio(audio_path: Path, start: float, end: float, output_dir: str) -> Path:
    """
    Cut audio_path to [start, end].

    The trim subfolder is created as a sibling of output_dir (not inside it),
    so the structure is:

        Transcriptions/
        ├── Angelo Badalamenti/          ← output_dir (original episode)
        │   └── Angelo Badalamenti.mp3
        └── Angelo Badalamenti_trim_10m00s_20m00s/   ← trim sibling
            └── Angelo Badalamenti.mp3  ← same stem → clean output filenames

    This avoids the stem collision that would occur if the subfolder were
    created inside output_dir.
    """
    import subprocess

    def _mmss(s: float) -> str:
        return f"{int(s) // 60}m{int(s) % 60:02d}s"

    # Place trim dir next to output_dir, not inside it
    parent = Path(output_dir).parent
    trim_dir = parent / f"{audio_path.stem}_trim_{_mmss(start)}_{_mmss(end)}"
    trim_dir.mkdir(parents=True, exist_ok=True)
    dest = trim_dir / audio_path.name  # keeps original filename & stem
    if dest.exists():
        return dest
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-to",
            str(end),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest


def _render_status(status: dict):
    labels = {
        "transcribed": "Transcribed",
        "diarized": "Diarized",
        "assigned": "Assigned",
        "mapped": "Speaker map",
        "exported": "Exported",
    }
    cols = st.columns(len(status))
    for col, (key, done) in zip(cols, status.items()):
        with col:
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {labels[key]}")


def _render_speaker_map(audio_path: Path, output_dir: str, force: bool = False):
    diarization = transcribe.load_diarization(audio_path, output_dir=output_dir)
    existing_map = transcribe.load_speaker_map(audio_path, output_dir=output_dir)
    speaker_ids = diarization["speakers_found"]
    ep = Path(audio_path).stem  # namespace all keys by episode

    all_segs = _load_diarized_segments_cached(str(audio_path), output_dir)
    segs_by_speaker: dict[str, list[dict]] = {}
    for seg in all_segs:
        sp = seg.get("speaker", "UNKNOWN")
        segs_by_speaker.setdefault(sp, []).append(seg)

    def _seg_flags(seg):
        text = str(seg.get("text", "")).strip()
        return is_hallucination(text), text

    new_map = {}
    for speaker_id in speaker_ids:
        all_speaker_segs = sorted(
            segs_by_speaker.get(speaker_id, []),
            key=lambda s: float(s["end"]) - float(s["start"]),
            reverse=True,
        )
        clean = [s for s in all_speaker_segs if not _seg_flags(s)[0]]
        flagged_segs = [s for s in all_speaker_segs if _seg_flags(s)[0]]
        top_segs = (clean + flagged_segs)[:3]
        n_flagged = len(flagged_segs)

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
                "🗑️", key=f"remove_{ep}_{speaker_id}", help="Fill [remove] in name field"
            ):
                st.session_state[f"speaker_{ep}_{speaker_id}"] = "[remove]"
                st.rerun()
        with col_name:
            name = st.text_input(
                "Name",
                value=existing_map.get(speaker_id, ""),
                key=f"speaker_{ep}_{speaker_id}",
                placeholder="Enter name…",
                label_visibility="collapsed",
            )
        new_map[speaker_id] = (
            st.session_state.get(f"speaker_{ep}_{speaker_id}", name) or speaker_id
        )

        # ── Audio previews — collapsed ──
        # Only one speaker's audio is active at a time to avoid Streamlit media file limits.
        top_seg_ids = {id(s) for s in top_segs}
        remaining_segs = [s for s in all_speaker_segs if id(s) not in top_seg_ids]
        n_clean = len(clean)
        label = f"🎧 {speaker_id}"
        if n_clean < len(top_segs):
            label += f" — ⚠️ {len(top_segs) - n_clean} suspect"
        with st.expander(label, expanded=False):
            active = st.session_state.get("_active_speaker_audio")
            show_top_audio = active == f"top_{ep}_{speaker_id}"
            show_audio = active == f"all_{ep}_{speaker_id}"

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if not show_top_audio:
                    if st.button(
                        "🔊 Load previews",
                        key=f"btn_top_{ep}_{speaker_id}",
                        use_container_width=True,
                    ):
                        st.session_state["_active_speaker_audio"] = (
                            f"top_{ep}_{speaker_id}"
                        )
                        st.rerun()
                else:
                    if st.button(
                        "✖ Unload",
                        key=f"btn_top_{ep}_{speaker_id}",
                        use_container_width=True,
                    ):
                        st.session_state.pop("_active_speaker_audio", None)
                        st.rerun()
            with col_btn2:
                if remaining_segs:
                    if not show_audio:
                        if st.button(
                            "🔊 Load all",
                            key=f"btn_load_{ep}_{speaker_id}",
                            use_container_width=True,
                        ):
                            st.session_state["_active_speaker_audio"] = (
                                f"all_{ep}_{speaker_id}"
                            )
                            st.rerun()
                    else:
                        if st.button(
                            "✖ Unload all",
                            key=f"btn_load_{ep}_{speaker_id}",
                            use_container_width=True,
                        ):
                            st.session_state.pop("_active_speaker_audio", None)
                            st.rerun()

            cols = st.columns(len(top_segs)) if top_segs else []
            for i, (col, seg) in enumerate(zip(cols, top_segs)):
                dur = float(seg["end"]) - float(seg["start"])
                hallu, text = _seg_flags(seg)
                with col:
                    if show_top_audio or show_audio:
                        try:
                            st.audio(
                                audio_slice_bytes(
                                    str(audio_path),
                                    float(seg["start"]),
                                    float(seg["end"]),
                                ),
                                format="audio/wav",
                            )
                        except Exception:
                            st.caption("Preview unavailable")
                    st.caption(
                        f"#{i + 1} · {dur:.1f}s · `{seg['start']:.1f}→{seg['end']:.1f}s`"
                    )
                    if not text:
                        st.caption("⚠️ _(no text)_")
                    elif hallu:
                        st.caption(f"⚠️ _{text[:70]}_ ← hallucination")
                    else:
                        st.caption(f"_{text[:70]}{'…' if len(text) > 70 else ''}_")

            if remaining_segs and show_audio:
                st.markdown(f"**Remaining {len(remaining_segs)} segments:**")
                for j, seg in enumerate(remaining_segs[:30]):
                    dur = float(seg["end"]) - float(seg["start"])
                    hallu, text = _seg_flags(seg)
                    flags = "⚠️ " if hallu else ""
                    col1, col2, col3 = st.columns([3, 4, 2])
                    with col1:
                        st.caption(
                            f"`{seg['start']:.1f}→{seg['end']:.1f}s` ({dur:.1f}s) {flags}"
                        )
                    with col2:
                        if not text:
                            st.caption("_(no text)_")
                        elif hallu:
                            st.caption(f"⚠️ _{text[:80]}_ ← hallucination")
                        else:
                            st.caption(f"_{text[:80]}{'…' if len(text) > 80 else ''}_")
                    with col3:
                        if show_audio:
                            try:
                                st.audio(
                                    audio_slice_bytes(
                                        str(audio_path),
                                        float(seg["start"]),
                                        float(seg["end"]),
                                    ),
                                    format="audio/wav",
                                )
                            except Exception:
                                st.caption("—")

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
    if st.button(
        "💾 Save speaker map",
        use_container_width=True,
        type="primary",
        disabled=bool(unnamed) or (already_saved and not force),
    ):
        transcribe.save_speaker_map(audio_path, new_map, output_dir=output_dir)
        st.success("Speaker map saved! You can now export the transcript.")
        st.session_state.requested_tab = "transcribe"
        st.rerun()


def _reset_transcript(audio_path, output_dir: str) -> None:
    from podcodex.core.transcribe import _EpisodePaths

    p = _EpisodePaths.from_audio(Path(audio_path), output_dir=output_dir)
    p.transcript.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _load_diarized_segments_cached(audio_path: str, output_dir: str) -> list:
    return transcribe.load_diarized_segments(Path(audio_path), output_dir=output_dir)


def _render_validate_status(
    audio_path,
    output_dir: str,
    raw_check,
    validated_check,
    promote_fn,
    reset_fn,
    label: str,
    key_prefix: str,
) -> None:
    """Shared promote/validate status bar for transcript, polished, and translations."""
    is_raw_only = raw_check(audio_path, output_dir=output_dir)
    is_validated = validated_check(audio_path, output_dir=output_dir)
    if is_raw_only:
        col_msg, col_btn = st.columns([4, 1])
        with col_msg:
            st.warning(
                f"⚠️ This {label} is unvalidated (raw pipeline output). "
                "Review it below, then promote when ready."
            )
        with col_btn:
            if st.button(
                "Save",
                key=f"promote_{key_prefix}",
                type="primary",
                use_container_width=True,
                help=f"Save as validated — copies {label}.raw.json → {label}.json",
            ):
                promote_fn(audio_path, output_dir=output_dir)
                st.toast(f"{label.capitalize()} saved as validated.")
                st.rerun()
    elif is_validated:
        col_badge, col_reset = st.columns([4, 1])
        with col_badge:
            st.success(f"✓ Validated {label}")
        with col_reset:
            if st.button(
                "Reset to raw",
                key=f"reset_{key_prefix}",
                use_container_width=True,
                help=f"Delete validated {label} — raw file is kept",
            ):
                reset_fn(audio_path, output_dir=output_dir)
                st.rerun()


def _render_transcript_editor(audio_path, output_dir: str):
    audio_path = Path(audio_path)
    t_key = f"editor_transcript_{audio_path}"
    if t_key not in st.session_state:
        st.session_state[t_key] = transcribe.load_transcript(
            audio_path, output_dir=output_dir
        )
    transcript = st.session_state[t_key]

    has_raw = transcript_raw_exists(audio_path, output_dir=output_dir)
    has_validated = is_validated_transcript(audio_path, output_dir=output_dir)
    if has_raw or has_validated:
        cols = st.columns(2)
        with cols[0]:
            if st.button(
                "↩ Load original", use_container_width=True, disabled=not has_raw
            ):
                st.session_state[t_key] = load_transcript_raw(
                    audio_path, output_dir=output_dir
                )
                st.session_state[f"transcript_{audio_path}_dirty"] = False
                st.rerun()
        with cols[1]:
            if st.button(
                "✏️ Load edits", use_container_width=True, disabled=not has_validated
            ):
                st.session_state[t_key] = load_transcript_validated(
                    audio_path, output_dir=output_dir
                )
                st.session_state[f"transcript_{audio_path}_dirty"] = False
                st.rerun()

    if not transcript or not transcript[0].get("speaker"):
        st.warning(
            "Transcript has no speaker info — save the speaker map and re-export first."
        )
        return

    def _on_save(merged):
        save_transcript(audio_path, merged, output_dir=output_dir)
        st.session_state[t_key] = merged
        st.session_state.transcript = merged
        st.toast("Transcript saved!")

    render_segment_editor(
        transcript,
        editor_key=f"transcript_{audio_path}",
        on_save=_on_save,
        audio_path=audio_path,
        show_timestamps=True,
        show_delete=True,
        show_flags=True,
        is_saved=is_validated_transcript(audio_path, output_dir=output_dir),
        export_fn=transcribe.transcript_to_text,
        export_filename=f"{audio_path.stem}_transcript.txt",
        next_tab="polish",
        next_tab_label="→ Go to Polish",
    )
