"""
podcodex.ui.streamlit_editor — Shared paginated segment editor.

Used by Transcribe, Polish and Translate tabs.
"""

import io
from pathlib import Path

import streamlit as st

from podcodex.core._utils import UNKNOWN_SPEAKERS
from podcodex.core.transcribe import REMOVE_SPEAKERS, is_segment_flagged
from constants import PAGE_SIZES, DEFAULT_PAGE_SIZE, AUDIO_PADDING


def render_segment_editor(
    segments: list[dict],
    editor_key: str,
    on_save,
    *,
    audio_path=None,
    reference_segments: list[dict] | None = None,
    show_timestamps: bool = False,
    show_delete: bool = False,
    show_flags: bool = False,
    show_speaker: bool = True,
    is_saved: bool = False,
    export_fn=None,
    export_filename: str | None = None,
    next_tab: str | None = None,
    next_tab_label: str | None = None,
):
    """
    Render a paginated, filterable segment editor.

    Parameters
    ----------
    segments           Segment list to edit (dicts with 'text', 'speaker', 'start', 'end').
    editor_key         Unique prefix for all session-state and widget keys.
    on_save            Callable(list[dict]) called with merged edited segments on save.
    audio_path         If provided, enables per-segment audio preview.
    reference_segments If provided, shows each reference segment's text alongside.
    show_timestamps    Enable start/end time editing.
    show_delete        Enable per-segment delete and bulk-delete of flagged segments.
    show_flags         Enable hallucination-based flagging and the "Flagged only" filter.
    export_fn          Callable(list[dict]) -> str for the text export download button.
    export_filename    Filename for the download button (e.g. "episode.polished.txt").
    next_tab           Session-state tab key to navigate to on "go to next" click.
    next_tab_label     Label for the navigation button (e.g. "→ Go to Translate").
    """
    if audio_path is not None:
        audio_path = Path(audio_path)

    # ── Session-state keys ──
    deleted_key = f"{editor_key}_deleted"
    page_key = f"{editor_key}_page"
    flag_filter_key = f"{editor_key}_filter_flagged"
    speaker_filter_key = f"{editor_key}_filter_speaker"
    dirty_key = f"{editor_key}_dirty"

    def _mark_dirty():
        st.session_state[dirty_key] = True

    if deleted_key not in st.session_state:
        st.session_state[deleted_key] = set()
    deleted: set[int] = st.session_state[deleted_key]
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    # ── Index sets ──
    all_active = [i for i in range(len(segments)) if i not in deleted]
    # [remove] speakers are always flagged regardless of show_flags
    remove_flagged = {
        i for i in all_active if segments[i].get("speaker", "") in REMOVE_SPEAKERS
    }
    if show_flags:

        def _live_seg(i):
            seg = segments[i]
            return {
                **seg,
                "start": st.session_state.get(
                    f"{editor_key}_start_{i}", seg.get("start", 0)
                ),
                "end": st.session_state.get(f"{editor_key}_end_{i}", seg.get("end", 0)),
            }

        flagged = [i for i in all_active if is_segment_flagged(_live_seg(i))]
    else:
        flagged = sorted(remove_flagged)
    all_speakers = sorted({segments[i].get("speaker", "") for i in all_active} - {""})

    # ── Filter / navigation bar ──
    show_flag_toggle = show_flags or bool(remove_flagged)
    col_widths = [3]
    if show_flag_toggle:
        col_widths.append(2)
    col_widths += [3, 1, 1, 1]
    bar = st.columns(col_widths)
    col_idx = 0

    caption_col = bar[col_idx]
    col_idx += 1

    if show_flag_toggle:
        with bar[col_idx]:
            col_idx += 1
            show_flagged_only = st.toggle(
                f"⚠️ Flagged only ({len(flagged)})", key=flag_filter_key
            )
            if (
                show_flagged_only
                and st.session_state.get(f"{flag_filter_key}_prev") != show_flagged_only
            ):
                st.session_state[page_key] = 0
        st.session_state[f"{flag_filter_key}_prev"] = show_flagged_only
    else:
        show_flagged_only = False

    with bar[col_idx]:
        col_idx += 1
        prev_sf = st.session_state.get(f"{speaker_filter_key}_prev", [])
        selected_speakers = st.multiselect(
            "Speaker",
            options=all_speakers,
            default=[],
            key=speaker_filter_key,
            label_visibility="collapsed",
            placeholder="All speakers",
        )
        if selected_speakers != prev_sf:
            st.session_state[page_key] = 0
        st.session_state[f"{speaker_filter_key}_prev"] = selected_speakers

    active_indices = flagged if show_flagged_only else all_active
    if selected_speakers:
        active_indices = [
            i
            for i in active_indices
            if segments[i].get("speaker", "") in selected_speakers
        ]

    page_size_key = f"{editor_key}_page_size"
    with bar[col_idx]:
        col_idx += 1
        page_size = st.selectbox(
            "Per page",
            options=PAGE_SIZES,
            index=PAGE_SIZES.index(
                st.session_state.get(page_size_key, DEFAULT_PAGE_SIZE)
            ),
            key=page_size_key,
            label_visibility="collapsed",
        )

    n_pages = max(1, (len(active_indices) + page_size - 1) // page_size)
    page = min(st.session_state[page_key], n_pages - 1)

    with caption_col:
        st.caption(
            f"{len(segments)} segments · {len(deleted)} deleted · page {page + 1}/{n_pages}"
        )
    with bar[col_idx]:
        col_idx += 1
        if st.button(
            "← Prev",
            disabled=page == 0,
            use_container_width=True,
            key=f"{editor_key}_prev",
        ):
            st.session_state[page_key] = page - 1
            st.rerun()
    with bar[col_idx]:
        if st.button(
            "Next →",
            disabled=page == n_pages - 1,
            use_container_width=True,
            key=f"{editor_key}_next",
        ):
            st.session_state[page_key] = page + 1
            st.rerun()

    # ── Segments ──
    page_indices = active_indices[page * page_size : (page + 1) * page_size]

    for i in page_indices:
        seg = segments[i]
        speaker = seg.get("speaker", "")
        text = seg.get("text", "")
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))

        live_start = st.session_state.get(f"{editor_key}_start_{i}", start)
        live_end = st.session_state.get(f"{editor_key}_end_{i}", end)
        live_seg = {**seg, "start": live_start, "end": live_end}
        density_warn = _density_warning(live_seg) if show_flags else None
        # Use current selectbox value for the title so it updates before save
        display_speaker = (
            st.session_state.get(f"{editor_key}_speaker_{i}", speaker)
            if show_speaker
            else speaker
        )
        is_unknown = show_flags and (
            not display_speaker or display_speaker in UNKNOWN_SPEAKERS
        )
        is_remove = display_speaker in REMOVE_SPEAKERS
        speaker_label = (
            f"⚠️ {display_speaker or 'None'}"
            if (is_unknown or is_remove)
            else display_speaker
        ) or "?"
        density_flag = " 🟡" if density_warn else ""
        ref_seg = (
            reference_segments[i]
            if reference_segments and i < len(reference_segments)
            else None
        )

        display_text = st.session_state.get(f"{editor_key}_text_{i}", text)
        with st.expander(
            f"[{start:.1f}s → {end:.1f}s] **{speaker_label}**{density_flag} — {display_text[:60]}{'...' if len(display_text) > 60 else ''}",
            expanded=False,
        ):
            if density_warn:
                st.warning(density_warn)

            if audio_path:
                audio_key = f"{editor_key}_audio_{i}"
                if st.session_state.get(audio_key):
                    try:
                        st.audio(
                            audio_slice_bytes(str(audio_path), start, end),
                            format="audio/wav",
                        )
                    except Exception:
                        st.caption("Preview unavailable")
                else:
                    if st.button(
                        "🔊 Load audio",
                        key=f"{editor_key}_load_audio_{i}",
                        use_container_width=True,
                    ):
                        st.session_state[audio_key] = True
                        st.rerun()

            if ref_seg is not None:
                st.caption(f"**Original:** {ref_seg.get('text', '')}")

            if show_speaker:
                options = (
                    all_speakers
                    if speaker in all_speakers
                    else ([speaker] + all_speakers if speaker else all_speakers)
                )
                st.selectbox(
                    "Speaker",
                    options=options,
                    index=options.index(speaker) if speaker in options else 0,
                    key=f"{editor_key}_speaker_{i}",
                    on_change=_mark_dirty,
                )

            st.text_area(
                "Text",
                value=text,
                key=f"{editor_key}_text_{i}",
                label_visibility="collapsed",
                height=80,
                on_change=_mark_dirty,
            )

            if show_timestamps or show_delete:
                row_widths = ([2, 2] if show_timestamps else []) + (
                    [1] if show_delete else []
                )
                row = st.columns(row_widths)
                row_idx = 0
                if show_timestamps:
                    with row[row_idx]:
                        st.number_input(
                            "Start (s)",
                            value=start,
                            min_value=0.0,
                            step=0.1,
                            format="%.1f",
                            key=f"{editor_key}_start_{i}",
                            on_change=_mark_dirty,
                        )
                    row_idx += 1
                    with row[row_idx]:
                        st.number_input(
                            "End (s)",
                            value=end,
                            min_value=0.0,
                            step=0.1,
                            format="%.1f",
                            key=f"{editor_key}_end_{i}",
                            on_change=_mark_dirty,
                        )
                    row_idx += 1
                if show_delete:
                    with row[row_idx]:
                        if st.button(
                            "🗑️ Delete",
                            key=f"{editor_key}_del_{i}",
                            use_container_width=True,
                        ):
                            st.session_state[deleted_key] = st.session_state[
                                deleted_key
                            ] | {i}
                            st.session_state[dirty_key] = True
                            st.rerun()

    # ── Bulk delete flagged ──
    if show_delete and flagged:
        st.divider()
        confirm_key = f"{editor_key}_confirm_del_flagged"
        if not st.session_state.get(confirm_key):
            if st.button(
                f"🗑️ Delete all flagged segments ({len(flagged)})",
                use_container_width=True,
                key=f"{editor_key}_del_all_btn",
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
                    "✅ Yes, delete all",
                    use_container_width=True,
                    type="primary",
                    key=f"{editor_key}_del_yes",
                ):
                    st.session_state[deleted_key] = st.session_state[deleted_key] | set(
                        flagged
                    )
                    st.session_state[dirty_key] = True
                    st.session_state.pop(confirm_key, None)
                    st.session_state[page_key] = 0
                    st.rerun()
            with col_no:
                if st.button(
                    "✖ Cancel", use_container_width=True, key=f"{editor_key}_del_no"
                ):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()

    # ── Bottom action bar ──
    st.divider()
    col_save, col_export, col_nav = st.columns(3)

    with col_save:
        if st.button(
            "💾 Save edits",
            use_container_width=True,
            type="secondary"
            if (is_saved and not st.session_state.get(dirty_key, False))
            else "primary",
            key=f"{editor_key}_save",
        ):
            merged = []
            for i in range(len(segments)):
                if i in deleted:
                    continue
                seg = segments[i]
                updated = {
                    **seg,
                    "text": st.session_state.get(
                        f"{editor_key}_text_{i}", seg.get("text", "")
                    ),
                }
                if show_speaker:
                    updated["speaker"] = st.session_state.get(
                        f"{editor_key}_speaker_{i}", seg.get("speaker", "")
                    )
                if show_timestamps:
                    updated["start"] = st.session_state.get(
                        f"{editor_key}_start_{i}", float(seg.get("start", 0))
                    )
                    updated["end"] = st.session_state.get(
                        f"{editor_key}_end_{i}", float(seg.get("end", 0))
                    )
                merged.append(updated)
            on_save(merged)
            st.session_state[dirty_key] = False
            st.session_state.pop(deleted_key, None)
            st.session_state[page_key] = 0
            st.rerun()

    with col_export:
        if export_fn and export_filename:
            export_segs = [
                {
                    **segments[i],
                    "text": st.session_state.get(
                        f"{editor_key}_text_{i}", segments[i].get("text", "")
                    ),
                }
                for i in range(len(segments))
                if i not in deleted
            ]
            st.download_button(
                "📄 Export as text",
                data=export_fn(export_segs),
                file_name=export_filename,
                mime="text/plain",
                use_container_width=True,
                key=f"{editor_key}_download",
            )

    with col_nav:
        if next_tab and next_tab_label:
            if st.button(
                next_tab_label, use_container_width=True, key=f"{editor_key}_nav"
            ):
                st.session_state.requested_tab = next_tab
                st.rerun()


# ── Shared helpers ──


def _density_warning(seg: dict) -> str | None:
    """Human-readable warning if the segment has suspiciously low speech density."""
    from podcodex.core.transcribe import segment_speech_density

    density = segment_speech_density(seg)
    if density is not None and density < 2.0:
        dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
        return (
            f"Low speech density ({density:.1f} chars/s over {dur:.1f}s)"
            " — may be music, noise, or a subtitle artifact."
        )
    return None


@st.cache_data(show_spinner=False)
def audio_slice_bytes(
    audio_path: str, start: float, end: float, pad: float = AUDIO_PADDING
) -> bytes:
    """Read a [start, end] slice from audio_path and return WAV bytes. Cached.

    pad: seconds of context added before and after the segment (default 0.3s).
    """
    import soundfile as sf

    info = sf.info(audio_path)
    sr = info.samplerate
    total_frames = info.frames
    audio, _ = sf.read(
        audio_path,
        start=max(0, int((start - pad) * sr)),
        stop=min(total_frames, int((end + pad) * sr)),
        dtype="float32",
        always_2d=False,
    )
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()
