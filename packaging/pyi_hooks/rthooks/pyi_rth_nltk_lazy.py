"""Runtime hook: register nltk's bundled data directory without importing nltk.

Replaces the auto-injected ``pyi_rth_nltk`` from hooks-contrib, which is::

    import nltk
    nltk.data.path.insert(0, os.path.join(sys._MEIPASS, "nltk_data"))

That bare ``import nltk`` costs ~1.6 s and ran at *every* launch of the
sidecar, before the entry script — measured as the single largest item in
the startup budget. Nothing in PodCodex imports nltk directly; it arrives
as a transitive dependency of whisperx, whose ``alignment.py`` needs it
during transcription alignment, minutes into a session if at all.

So the path insertion is deferred to the moment nltk is actually imported.
The end state is identical: the same entry, at the same position, before
anything reads ``nltk.data.path``.

Runtime hooks run before the entry script, but the frozen ``sys.path`` is
already set up, so ``podcodex`` is importable here. It falls back to the
original eager behaviour if that import fails, because a broken nltk data
path fails far away from here, inside whisperx alignment.
"""

import os
import sys


def _register_bundled_nltk_data() -> None:
    import nltk

    nltk.data.path.insert(0, os.path.join(sys._MEIPASS, "nltk_data"))


try:
    from podcodex.bootstrap import defer_until_imported
except Exception:
    _register_bundled_nltk_data()
else:
    defer_until_imported("nltk", _register_bundled_nltk_data)
