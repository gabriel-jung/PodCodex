"""
podcodex.ui.streamlit_transcribe — Transcription tab
"""

import json
from pathlib import Path

import streamlit as st

from podcodex.core import transcribe, synthesize


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
            st.markdown("### 🏷️ Step 3 — Speaker Names & Voice Samples")
            st.caption("Name each speaker and extract audio samples for voice cloning.")

            col1, col2 = st.columns(2)
            with col1:
                top_k = st.number_input(
                    "Voice candidates per speaker",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Number of audio clips to extract per speaker, sorted by duration. You will be able to listen to verify the correct speaker.",
                )
            with col2:
                min_duration = st.number_input(
                    "Min sample duration (s)",
                    min_value=0.0,
                    value=3.0,
                    help="Minimum duration in seconds for a segment to be considered as a voice sample candidate. Longer clips generally give better voice cloning results.",
                )

            if st.button("Extract voice samples", use_container_width=True):
                with st.spinner("Extracting voice samples..."):
                    segments = transcribe.load_diarized_segments(
                        audio_path, output_dir=output_dir
                    )
                    voice_samples = synthesize.extract_voice_samples(
                        audio_path,
                        segments,
                        output_dir=output_dir,
                        min_duration=min_duration or None,
                        top_k=top_k,
                    )
                    st.session_state.voice_samples = voice_samples
                st.session_state.requested_tab = "transcribe"
                st.rerun()

            _render_speaker_map(audio_path, output_dir)

    # ── Section 5: Export ──
    with st.container(border=True):
        col_title, col_force = st.columns([4, 1])
        with col_title:
            st.markdown("### 📝 Step 4 — Export Transcript")
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

    # ── Section 6: Transcript editor ──
    if status["exported"]:
        with st.container(border=True):
            st.markdown("### ✏️ Step 5 — Review & Edit Transcript")
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

    voice_samples = st.session_state.get(
        "voice_samples"
    ) or _load_voice_samples_from_disk(audio_path, output_dir, speaker_ids)

    st.markdown("**Name each speaker:**")

    new_map = {}
    for speaker_id in speaker_ids:
        st.markdown(f"---\n**{speaker_id}**")
        samples = voice_samples.get(speaker_id, [])

        col_input, *col_audios = st.columns([2] + [1] * min(len(samples), 3))

        with col_input:
            name = st.text_input(
                "Speaker name",
                value=existing_map.get(speaker_id, ""),
                key=f"speaker_{speaker_id}",
                placeholder="Enter name...",
                help=f"Name to use for {speaker_id} in the transcript and synthesis.",
                label_visibility="collapsed",
            )
            new_map[speaker_id] = name or speaker_id

        if samples:
            for i, (col, sample) in enumerate(zip(col_audios, samples)):
                with col:
                    st.audio(str(sample["file"]))
                    st.caption(f"#{i + 1} · {sample['duration']:.1f}s")
        else:
            with col_input:
                st.caption(
                    "No voice samples yet — click 'Extract voice samples' above."
                )

    if st.button("💾 Save speaker map", use_container_width=True, type="primary"):
        transcribe.save_speaker_map(audio_path, new_map, output_dir=output_dir)
        st.success("Speaker map saved! You can now export the transcript.")
        st.session_state.requested_tab = "transcribe"
        st.rerun()


def _load_voice_samples_from_disk(
    audio_path: Path, output_dir: str, speaker_ids: list
) -> dict:
    samples_dir = Path(output_dir) / "voice_samples"
    result = {}
    for speaker_id in speaker_ids:
        files = (
            sorted(samples_dir.glob(f"{speaker_id}_*.wav"))
            if samples_dir.exists()
            else []
        )
        if files:
            result[speaker_id] = [
                {"file": f, "duration": _wav_duration(f), "text": ""} for f in files
            ]
    return result


def _wav_duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.duration
    except Exception:
        return 0.0


def _render_transcript_editor(audio_path: Path, output_dir: str):
    transcript = transcribe.load_transcript(audio_path, output_dir=output_dir)
    st.session_state.transcript = transcript

    if not transcript or not transcript[0].get("speaker"):
        st.warning(
            "Transcript has no speaker info — please save the speaker map and re-export before editing."
        )
        return

    st.caption(
        f"{len(transcript)} segments — expand a segment to edit its text, then save all changes."
    )

    edited = []
    for i, seg in enumerate(transcript):
        with st.expander(
            f"[{seg['start']:.1f}s] **{seg['speaker']}** — {seg['text'][:60]}{'...' if len(seg['text']) > 60 else ''}",
            expanded=False,
        ):
            new_text = st.text_area(
                "Text",
                value=seg["text"],
                key=f"edit_seg_{i}",
                label_visibility="collapsed",
                height=80,
            )
            edited.append({**seg, "text": new_text})

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save edits", use_container_width=True, type="primary"):
            p = transcribe._EpisodePaths.from_audio(audio_path, output_dir=output_dir)
            p.transcript.write_text(
                json.dumps(edited, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            st.session_state.transcript = edited
            st.success("Transcript saved!")
    with col2:
        if st.button("→ Go to Translation", use_container_width=True):
            st.session_state.requested_tab = "translate"
            st.rerun()
