"""Core transcription, correcting, translation, and synthesis pipeline."""

from __future__ import annotations

import importlib as _importlib
import types as _types


# synthesize has heavy deps (soundfile, numpy) from the pipeline extra —
# import lazily so the API can start without [pipeline] installed.
def __getattr__(name: str) -> _types.ModuleType:
    if name == "synthesize":
        return _importlib.import_module(".synthesize", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_segments_json(data, required: tuple[str, ...] = ("text",)) -> str | None:
    """Validate that data looks like a transcript segments array.

    Returns a human-readable error string, or None if valid.
    """
    if not isinstance(data, list):
        if isinstance(data, dict):
            keys = list(data.keys())[:6]
            hint = (
                " Looks like a raw Whisper output — extract the 'segments' key first."
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
            f"Fields found: {found}."
        )
    return None
