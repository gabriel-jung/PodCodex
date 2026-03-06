"""
podcodex.ui.streamlit_transcribe — Transcription tab
"""

import json
from pathlib import Path

import streamlit as st

from podcodex.core import transcribe
from podcodex.core.synthesize import is_hallucination


def render():
    st.header("Transcription")
    st.caption("Transcribe, diarize and prepare your podcast episode for translation.")

    # ── Section 1: Audio & Config ──
    with st.container(border=True):
        st.markdown("### 📁 Audio File")

        uploaded = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            help="Supported formats: MP3, WAV, M4A, OGG, FLAC. The file will be saved to the output directory.",
        )

        col1, col2 = st.columns(2)
        with col1:
            default_output = str(Path.cwd() / "Transcriptions")
            if st.session_state.get("audio_path"):
                output_dir = st.session_state.get("output_dir", default_output)
                st.text_input(
                    "Output directory",
                    value=output_dir,
                    disabled=True,
                    help="Output directory is locked once an audio file is loaded. Reload the page to change it.",
                )
            else:
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
                help="ISO 639-1 language code of the podcast (e.g. 'fr' for French, 'en' for English). Used by WhisperX for transcription.",
            )

        # Save uploaded file into base_output_dir/{stem}/
        if uploaded and st.session_state.get("audio_filename") != uploaded.name:
            stem = Path(uploaded.name).stem
            # Always use the base output dir (not the previous episode subdir)
            base_output = st.session_state.get(
                "base_output_dir", str(Path(output_dir).resolve())
            )
            episode_dir = Path(base_output) / stem
            episode_dir.mkdir(parents=True, exist_ok=True)
            audio_dest = episode_dir / uploaded.name
            audio_dest.write_bytes(uploaded.read())
            st.session_state.audio_path = audio_dest
            st.session_state.audio_filename = uploaded.name
            st.session_state.output_dir = str(episode_dir)
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
            st.markdown("### 🏷️ Step 4 — Name Speakers")
            st.caption(
                "Listen to each speaker's segments and enter their name. "
                "Voice sample extraction happens in the Synthesis tab."
            )
            _render_speaker_map(audio_path, output_dir)

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
            st.markdown("### ✏️ Step 6 — Review & Edit Transcript")
            st.caption(
                "Correct transcription errors directly. Changes are saved to the transcript file and will be used for translation."
            )
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


def _render_speaker_map(audio_path: Path, output_dir: str):
    diarization = transcribe.load_diarization(audio_path, output_dir=output_dir)
    existing_map = transcribe.load_speaker_map(audio_path, output_dir=output_dir)
    speaker_ids = diarization["speakers_found"]

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
        st.markdown(f"---\n**{speaker_id}**")

        all_speaker_segs = sorted(
            segs_by_speaker.get(speaker_id, []),
            key=lambda s: float(s["end"]) - float(s["start"]),
            reverse=True,
        )
        clean = [s for s in all_speaker_segs if not _seg_flags(s)[0]]
        flagged = [s for s in all_speaker_segs if _seg_flags(s)[0]]
        top_segs = (clean + flagged)[:3]

        # Name input — always visible, no audio loading
        name = st.text_input(
            "Speaker name",
            value=existing_map.get(speaker_id, ""),
            key=f"speaker_{speaker_id}",
            placeholder="Enter name...",
            help=f"Name to use for {speaker_id} in the transcript and synthesis.",
            label_visibility="collapsed",
        )
        new_map[speaker_id] = name or speaker_id

        # Audio previews — collapsed by default, load on demand
        n_clean = len(clean)
        label = f"🎧 Listen to {speaker_id}"
        if n_clean < len(top_segs):
            label += f" — ⚠️ {len(top_segs) - n_clean} suspect segment(s)"
        with st.expander(label, expanded=False):
            cols = st.columns(len(top_segs)) if top_segs else []
            for i, (col, seg) in enumerate(zip(cols, top_segs)):
                dur = float(seg["end"]) - float(seg["start"])
                hallu, text = _seg_flags(seg)
                with col:
                    try:
                        st.audio(
                            _audio_slice_bytes(
                                str(audio_path), float(seg["start"]), float(seg["end"])
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

            if all_speaker_segs:
                load_key = f"load_all_audio_{speaker_id}"
                show_audio = st.session_state.get(load_key, False)
                col_hdr, col_btn = st.columns([3, 2])
                with col_hdr:
                    st.markdown("**Browse all segments:**")
                with col_btn:
                    if not show_audio:
                        if st.button(
                            "🔊 Load audio",
                            key=f"btn_load_{speaker_id}",
                            use_container_width=True,
                        ):
                            st.session_state[load_key] = True
                            st.rerun()
                for j, seg in enumerate(all_speaker_segs[:30]):
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
                                    _audio_slice_bytes(
                                        str(audio_path),
                                        float(seg["start"]),
                                        float(seg["end"]),
                                    ),
                                    format="audio/wav",
                                )
                            except Exception:
                                st.caption("—")

    if st.button("💾 Save speaker map", use_container_width=True, type="primary"):
        transcribe.save_speaker_map(audio_path, new_map, output_dir=output_dir)
        st.success("Speaker map saved! You can now export the transcript.")
        st.session_state.requested_tab = "transcribe"
        st.rerun()


@st.cache_data(show_spinner=False)
def _load_diarized_segments_cached(audio_path: str, output_dir: str) -> list:
    return transcribe.load_diarized_segments(Path(audio_path), output_dir=output_dir)


@st.cache_data(show_spinner=False)
def _audio_slice_bytes(audio_path: str, start: float, end: float) -> bytes:
    """Read a [start, end] slice from audio_path and return it as WAV bytes.

    Result is cached by Streamlit so repeated rerenders don't re-read the file.
    Note: audio_path must be a str (not Path) for cache key hashing.
    """
    import io
    import soundfile as sf

    info = sf.info(audio_path)
    sr = info.samplerate
    s_idx = int(start * sr)
    e_idx = int(end * sr)
    audio, _ = sf.read(
        audio_path, start=s_idx, stop=e_idx, dtype="float32", always_2d=False
    )
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _segment_density_warning(text: str, dur: float) -> str | None:
    """Return a warning message if chars/sec ratio looks suspicious, else None."""
    chars = len(text.strip())
    if dur < 0.5:
        return None
    density = chars / dur
    if density < 2.0:
        return f"Low speech density ({density:.1f} chars/s over {dur:.1f}s) — may be music, noise, or a subtitle artifact."
    return None


_EDITOR_PAGE_SIZE = 20
_UNKNOWN_SPEAKERS = {"UNKNOWN", "UNK", "None", "none"}


def _render_transcript_editor(audio_path: Path, output_dir: str):
    # Cache transcript in session_state — only reload from disk after save
    t_key = f"editor_transcript_{audio_path}"
    if t_key not in st.session_state:
        st.session_state[t_key] = transcribe.load_transcript(
            audio_path, output_dir=output_dir
        )
    transcript = st.session_state[t_key]

    if not transcript or not transcript[0].get("speaker"):
        st.warning(
            "Transcript has no speaker info — save the speaker map and re-export first."
        )
        return

    del_key = f"deleted_segs_{audio_path}"
    if del_key not in st.session_state:
        st.session_state[del_key] = set()
    deleted: set[int] = st.session_state[del_key]

    page_key = f"editor_page_{audio_path}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    filter_key = f"editor_filter_{audio_path}"

    def _is_flagged(i):
        seg = transcript[i]
        speaker = seg.get("speaker", "")
        text = seg.get("text", "")
        dur = float(seg["end"]) - float(seg["start"])
        return (not speaker or speaker in _UNKNOWN_SPEAKERS) or bool(
            _segment_density_warning(text, dur)
        )

    all_active = [i for i in range(len(transcript)) if i not in deleted]
    flagged = [i for i in all_active if _is_flagged(i)]

    col_cap, col_filter, col_prev, col_next = st.columns([3, 2, 1, 1])
    with col_filter:
        show_flagged_only = st.toggle(
            f"⚠️ Flagged only ({len(flagged)})", key=filter_key
        )
        if (
            show_flagged_only
            and st.session_state.get(f"{filter_key}_prev") != show_flagged_only
        ):
            st.session_state[page_key] = 0
    st.session_state[f"{filter_key}_prev"] = show_flagged_only

    active_indices = flagged if show_flagged_only else all_active
    n_pages = max(1, (len(active_indices) + _EDITOR_PAGE_SIZE - 1) // _EDITOR_PAGE_SIZE)
    page = min(st.session_state[page_key], n_pages - 1)

    with col_cap:
        st.caption(
            f"{len(transcript)} segments · {len(deleted)} deleted · page {page + 1}/{n_pages}"
        )
    with col_prev:
        if st.button("← Prev", disabled=page == 0, use_container_width=True):
            st.session_state[page_key] = page - 1
            st.rerun()
    with col_next:
        if st.button("Next →", disabled=page == n_pages - 1, use_container_width=True):
            st.session_state[page_key] = page + 1
            st.rerun()

    page_indices = active_indices[
        page * _EDITOR_PAGE_SIZE : (page + 1) * _EDITOR_PAGE_SIZE
    ]

    edited_on_page = {}
    for i in page_indices:
        seg = transcript[i]
        speaker = seg.get("speaker", "")
        text = seg.get("text", "")
        start = float(seg["start"])
        end = float(seg["end"])
        dur = end - start

        is_unknown = not speaker or speaker in _UNKNOWN_SPEAKERS
        density_warn = _segment_density_warning(text, dur)
        speaker_label = f"⚠️ {speaker or 'None'}" if is_unknown else speaker
        density_flag = " 🟡" if density_warn else ""

        with st.expander(
            f"[{start:.1f}s → {end:.1f}s] **{speaker_label}**{density_flag} — {text[:60]}{'...' if len(text) > 60 else ''}",
            expanded=False,
        ):
            if density_warn:
                st.warning(density_warn)

            seg_audio_key = f"editor_audio_{audio_path}_{i}"
            if st.session_state.get(seg_audio_key):
                try:
                    st.audio(
                        _audio_slice_bytes(str(audio_path), start, end),
                        format="audio/wav",
                    )
                except Exception:
                    st.caption("Preview unavailable")
            else:
                if st.button(
                    "🔊 Load audio", key=f"load_audio_{i}", use_container_width=True
                ):
                    st.session_state[seg_audio_key] = True
                    st.rerun()

            new_text = st.text_area(
                "Text",
                value=text,
                key=f"edit_seg_{i}",
                label_visibility="collapsed",
                height=80,
            )
            col_s, col_e, col_del = st.columns([2, 2, 1])
            with col_s:
                new_start = st.number_input(
                    "Start (s)",
                    value=start,
                    min_value=0.0,
                    step=0.1,
                    format="%.1f",
                    key=f"start_{i}",
                )
            with col_e:
                new_end = st.number_input(
                    "End (s)",
                    value=end,
                    min_value=0.0,
                    step=0.1,
                    format="%.1f",
                    key=f"end_{i}",
                )
            with col_del:
                if st.button("🗑️ Delete", key=f"del_{i}", use_container_width=True):
                    st.session_state[del_key] = st.session_state[del_key] | {i}
                    st.rerun()

            edited_on_page[i] = {
                **seg,
                "text": new_text,
                "start": new_start,
                "end": new_end,
            }

    # ── Delete all flagged ──
    if flagged:
        st.divider()
        confirm_key = f"confirm_delete_flagged_{audio_path}"
        if not st.session_state.get(confirm_key):
            if st.button(
                f"🗑️ Delete all flagged segments ({len(flagged)})",
                use_container_width=True,
            ):
                st.session_state[confirm_key] = True
                st.rerun()
        else:
            st.warning(
                f"Delete all **{len(flagged)}** flagged segments? This cannot be undone until you save."
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button(
                    "✅ Yes, delete all", use_container_width=True, type="primary"
                ):
                    st.session_state[del_key] = st.session_state[del_key] | set(flagged)
                    st.session_state.pop(confirm_key, None)
                    st.session_state[page_key] = 0
                    st.rerun()
            with col_no:
                if st.button("✖ Cancel", use_container_width=True):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save edits", use_container_width=True, type="primary"):
            # Collect edits from all pages via session state widget keys,
            # falling back to the cached transcript for unvisited pages.
            merged = []
            for i in range(len(transcript)):
                if i in deleted:
                    continue
                seg = transcript[i]
                merged.append(
                    {
                        **seg,
                        "text": st.session_state.get(
                            f"edit_seg_{i}", seg.get("text", "")
                        ),
                        "start": st.session_state.get(
                            f"start_{i}", float(seg["start"])
                        ),
                        "end": st.session_state.get(f"end_{i}", float(seg["end"])),
                    }
                )
            p = transcribe._EpisodePaths.from_audio(audio_path, output_dir=output_dir)
            full = transcribe.load_transcript_full(audio_path, output_dir=output_dir)
            full["segments"] = merged
            p.transcript.write_text(
                json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            st.session_state[t_key] = merged
            st.session_state.transcript = merged
            st.session_state.pop(del_key, None)
            st.session_state[page_key] = 0
            st.success("Transcript saved!")
    with col2:
        txt = transcribe.transcript_to_text(
            [
                edited_on_page.get(i, transcript[i])
                for i in range(len(transcript))
                if i not in deleted
            ]
        )
        audio_stem = Path(audio_path).stem
        st.download_button(
            "📄 Export as text",
            data=txt,
            file_name=f"{audio_stem}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col3:
        if st.button("→ Go to Translation", use_container_width=True):
            st.session_state.requested_tab = "translate"
            st.rerun()
