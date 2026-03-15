"""
podcodex.core._utils — Lightweight shared utilities with no heavy dependencies.

Functions here must NOT import torch, pandas, or other heavy libraries.
"""


def simplify_transcript(segments: list[dict], max_gap: float = 10.0) -> list[dict]:
    """
    Merge consecutive segments from the same speaker into single entries.
    Segments are only merged if the gap between them is <= max_gap seconds,
    preventing merges across music breaks or long silences.

    Args:
        segments: raw diarized segments
        max_gap: maximum silence gap (seconds) to merge across (default: 10s)

    Returns:
        List of simplified segments [{speaker, start, end, text}]
    """
    result = []
    for seg in segments:
        speaker = seg.get("speaker_name") or seg.get("speaker") or "UNKNOWN"
        entry = {
            "speaker": speaker,
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": str(seg.get("text", "")).strip(),
        }
        if (
            result
            and result[-1]["speaker"] == entry["speaker"]
            and entry["start"] - result[-1]["end"] <= max_gap
        ):
            result[-1]["end"] = entry["end"]
            result[-1]["text"] += " " + entry["text"]
        else:
            if result and entry["start"] - result[-1]["end"] > max_gap:
                result.append(
                    {
                        "speaker": "[BREAK]",
                        "start": result[-1]["end"],
                        "end": entry["start"],
                        "text": "",
                    }
                )
            result.append(entry)
    return result
