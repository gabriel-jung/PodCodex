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
