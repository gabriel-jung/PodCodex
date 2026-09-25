"""Transcript export formats (text, SRT, WebVTT) and subtitle import.

``segments_to_*`` write, ``srt_to_segments`` / ``vtt_to_segments`` read
back, with the cue clean-up subtitle files need (duplicate and rolling
YouTube cues, HTML entities, speaker label detection). Split out of
``core/_utils``.
"""

from __future__ import annotations

import re
from collections.abc import Container


from podcodex.core._utils import (
    NARRATOR_SPEAKER,
    is_unattributed,
)


def segments_to_text(
    segments: list[dict], text_field: str = "text", declared: Container[str] = ()
) -> str:
    """Format segments as plain readable text.

    Args:
        segments   : list of segment dicts with speaker, start, end, and text fields
        text_field : which field to use for the text content (default "text")
    """
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        if is_unattributed(speaker, declared):
            speaker = ""
        start = seg.get("start")
        end = seg.get("end")
        if start is not None and end is not None:
            header = f"[{start:.3f}s - {end:.3f}s] {speaker}".rstrip()
        else:
            header = speaker
        text = seg.get(text_field) or "[empty]"
        # An untimed, unattributed segment has no header at all; emitting an
        # empty one would open the block with a blank line.
        lines.append(f"{header}\n{text}" if header else text)
    return "\n\n".join(lines)


def segments_to_srt(
    segments: list[dict], text_field: str = "text", declared: Container[str] = ()
) -> str:
    """Format segments as SRT subtitles.

    Args:
        segments   : list of segment dicts with speaker, start, end, and text fields
        text_field : which field to use for the text content (default "text")
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "")
        text = seg.get(text_field) or "[empty]"
        prefix = f"{speaker}: " if not is_unattributed(speaker, declared) else ""
        lines.append(str(i))
        lines.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        lines.append(f"{prefix}{text}")
        lines.append("")
    return "\n".join(lines)


def _fmt_ts(seconds: float, ms_sep: str) -> str:
    """``HH:MM:SS<sep>mmm``, rounded to the millisecond once.

    Deriving each field from the float truncated: 1.001 s is 1.00099... in
    binary and came out as ``,000``.
    """
    total_ms = max(0, round(seconds * 1000))
    s, ms = divmod(total_ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{ms_sep}{ms:03d}"


def _srt_ts(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    return _fmt_ts(seconds, ",")


def segments_to_vtt(
    segments: list[dict], text_field: str = "text", declared: Container[str] = ()
) -> str:
    """Format segments as WebVTT subtitles.

    Args:
        segments   : list of segment dicts with speaker, start, end, and text fields
        text_field : which field to use for the text content (default "text")
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "")
        text = seg.get(text_field) or "[empty]"
        prefix = f"<v {speaker}>" if not is_unattributed(speaker, declared) else ""
        lines.append(f"{_vtt_ts(start)} --> {_vtt_ts(end)}")
        lines.append(f"{prefix}{text}")
        lines.append("")
    return "\n".join(lines)


def _vtt_ts(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    return _fmt_ts(seconds, ".")


# ── Subtitle parsing (inverse of segments_to_srt / segments_to_vtt) ────


def _parse_ts(ts: str) -> float:
    """Parse an SRT or VTT timestamp (``[HH:]MM:SS[,.]mmm``) to seconds."""
    parts = ts.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


_VTT_SPEAKER_RE = re.compile(r"<v\s+([^>]+)>")


# Shortest repeated text the rolling-subtitle collapse will strip. A rolling
# cue repeats a whole caption line; a short repeat ("Okay.") is as likely to
# be a speaker saying it again.
_MIN_ROLLING_OVERLAP = 10


def _rolling_overlap(prev: str, cur: str) -> int:
    """Length of the longest suffix of *prev* that starts *cur*, or 0.

    Counted only when it is at least ``_MIN_ROLLING_OVERLAP`` characters and
    falls on word boundaries at both ends; otherwise stripping it would cut
    a word, or drop a short phrase that was genuinely said twice.
    """
    for length in range(len(prev), _MIN_ROLLING_OVERLAP - 1, -1):
        if not cur.startswith(prev[-length:]):
            continue
        starts_on_word = length == len(prev) or prev[-length - 1].isspace()
        ends_on_word = length == len(cur) or cur[length].isspace()
        if starts_on_word and ends_on_word:
            return length
    return 0


def _merge_parsed_cues(cues: list[dict]) -> list[dict]:
    """Deduplicate subtitle cues, preserving original timing.

    YouTube auto-generated subtitles often produce overlapping cues with
    repeated text.  This pass deduplicates consecutive identical lines and
    cleans HTML entities, but does NOT merge distinct cues — the original
    subtitle timing is kept as-is.
    """
    if not cues:
        return []

    import html

    # Decode HTML entities once (a hand-rolled chain decoded "&amp;lt;" twice,
    # to "<") and turn non-breaking spaces into plain ones.
    for cue in cues:
        text = html.unescape(cue["text"]).replace("\xa0", " ")
        cue["text"] = re.sub(r"  +", " ", text).strip()

    # Deduplicate consecutive identical text
    deduped: list[dict] = [cues[0]]
    for cue in cues[1:]:
        prev = deduped[-1]
        if cue["text"] == prev["text"] and cue["speaker"] == prev["speaker"]:
            # Extend end time of previous cue
            prev["end"] = max(prev["end"], cue["end"])
        else:
            deduped.append(cue)

    # Detect and collapse rolling/progressive subtitles.
    # YouTube auto-generated VTTs display text progressively: each cue shows
    # the previously completed line plus new words.  Between the rolling cues
    # there are brief "flash" cues (< 0.05s) that just repeat completed text.
    # We strip the flash cues, detect rolling overlap, and extract only the
    # new text from each cue.
    if len(deduped) >= 4:
        # Remove near-zero-duration "flash" cues
        no_flash: list[dict] = []
        for cue in deduped:
            if cue["end"] - cue["start"] >= 0.05:
                no_flash.append(cue)

        # Detect rolling pattern: suffix of cue[i] == prefix of cue[i+1]
        if len(no_flash) >= 4:
            overlap_count = 0
            for i in range(len(no_flash) - 1):
                t1, t2 = no_flash[i]["text"], no_flash[i + 1]["text"]
                # Check if any suffix of t1 (>10 chars) is a prefix of t2
                min_overlap = min(10, len(t1) // 2)
                for length in range(len(t1), min_overlap - 1, -1):
                    if t2.startswith(t1[-length:]):
                        overlap_count += 1
                        break

            if overlap_count > len(no_flash) * 0.3:
                collapsed: list[dict] = []
                for i, cue in enumerate(no_flash):
                    if i == 0:
                        collapsed.append(cue)
                        continue
                    best_overlap = _rolling_overlap(
                        no_flash[i - 1]["text"], cue["text"]
                    )
                    cur_text = cue["text"]
                    if best_overlap > 0:
                        new_text = cur_text[best_overlap:].strip()
                        if new_text:
                            collapsed.append(
                                {
                                    **cue,
                                    "text": new_text,
                                }
                            )
                    else:
                        collapsed.append(cue)
                deduped = collapsed

    return deduped


def srt_to_segments(srt_text: str) -> list[dict]:
    """Parse SRT subtitle text into segment dicts.

    Returns a list of ``{"speaker": str, "text": str, "start": float,
    "end": float}`` dicts.  Speaker is extracted from a ``Speaker: ``
    prefix if present.

    Args:
        srt_text: Full SRT file content.
    """
    parsed: list[tuple[float, float, str]] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Find the timestamp line (skip the index line)
        ts_line = None
        text_start = 0
        for idx, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                text_start = idx + 1
                break
        if ts_line is None:
            continue
        parts = ts_line.split("-->")
        if len(parts) != 2:
            continue
        start = _parse_ts(parts[0])
        end = _parse_ts(parts[1])
        text = " ".join(lines[text_start:]).strip()
        if text:
            parsed.append((start, end, text))

    labels = _srt_speaker_labels(text for _, _, text in parsed)
    cues: list[dict] = []
    for start, end, text in parsed:
        speaker = ""
        prefix, sep, rest = text.partition(": ")
        if sep and prefix in labels and rest:
            speaker, text = prefix, rest
        cues.append(
            {
                "speaker": speaker or NARRATOR_SPEAKER,
                "text": text,
                "start": start,
                "end": end,
            }
        )

    return _merge_parsed_cues(cues)


_SRT_LABEL_RE = re.compile(r"^[^\s.,!?:;\"'()][^.,!?:;\"()]{0,38}$")


def _srt_speaker_labels(texts) -> set[str]:
    """``Label: text`` prefixes in an SRT file that are speaker labels.

    SRT has no speaker field, so a leading ``Name: `` is the convention, but
    ordinary text uses colons too ("Note: this changed", "So here's the
    thing: ..."). A candidate is short (at most three words, no sentence
    punctuation). It counts as a speaker when the file is a labelled
    transcript (most cues carry a candidate, as ``segments_to_srt`` writes
    them), when it recurs the way a speaker does, or when it is written in
    capitals the way a lone label usually is (``JOHN: ``, ``SPEAKER_01: ``).
    """
    counts: dict[str, int] = {}
    total = 0
    for text in texts:
        total += 1
        prefix, sep, rest = text.partition(": ")
        if sep and rest and len(prefix.split()) <= 3 and _SRT_LABEL_RE.match(prefix):
            counts[prefix] = counts.get(prefix, 0) + 1
    labelled_file = sum(counts.values()) >= max(2, total / 2)
    return {
        label
        for label, n in counts.items()
        if labelled_file
        or n >= 2
        or (label.upper() == label and any(c.isalpha() for c in label))
    }


def vtt_to_segments(vtt_text: str) -> list[dict]:
    """Parse WebVTT subtitle text into segment dicts.

    Handles YouTube's auto-generated format with overlapping/duplicate cues
    and ``<v SpeakerName>`` voice tags.

    Returns a list of ``{"speaker": str, "text": str, "start": float,
    "end": float}`` dicts.

    Args:
        vtt_text: Full WebVTT file content.
    """
    cues: list[dict] = []
    blocks = re.split(r"\n\s*\n", vtt_text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        # Find timestamp line
        ts_line = None
        text_start = 0
        for idx, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                text_start = idx + 1
                break
        if ts_line is None:
            continue
        # Strip position/alignment metadata after timestamp
        ts_part = ts_line.split("-->")
        if len(ts_part) != 2:
            continue
        start = _parse_ts(ts_part[0].split()[0] if ts_part[0].strip() else "0")
        end_raw = ts_part[1].strip().split()
        end = _parse_ts(end_raw[0]) if end_raw else start

        text = " ".join(lines[text_start:]).strip()
        if not text:
            continue
        # Extract speaker from <v SpeakerName> tags
        speaker = ""
        m = _VTT_SPEAKER_RE.search(text)
        if m:
            speaker = m.group(1).strip()
            text = _VTT_SPEAKER_RE.sub("", text).strip()
        # Strip remaining HTML-like tags
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            cues.append(
                {
                    "speaker": speaker or NARRATOR_SPEAKER,
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )

    return _merge_parsed_cues(cues)
