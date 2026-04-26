"""PyInstaller runtime hook: tolerate Anaconda/conda-flavored sys.version strings.

cloudpickle (pulled in by joblib via sklearn via nltk via the auto-injected
``pyi_rth_nltk`` hook) calls ``platform.python_implementation()`` at module
import time. That walks ``platform._sys_version`` whose regex insists on the
canonical ``"3.12.11 (main, ...)"`` shape. Anaconda/conda-forge prepend
``" | packaged by Anaconda, Inc. | "`` and the parser raises::

    ValueError: failed to parse CPython sys.version: '...'

The PyInstaller bootloader sometimes bundles an Anaconda-flavored interpreter
even when the build venv was conda-forge, so the frozen binary inherits the
broken sys.version. We pre-strip the ``" | packaged by ... | "`` decoration
once, at the very start of every PyInstaller startup, so every later hook
sees a parser-friendly string.

This file is named ``pyi_rth_aaa_*`` so it sorts ahead of every other
runtime hook (PyInstaller runs them in alphabetical order).
"""

import re
import sys


def _strip_anaconda_decoration(version: str) -> str:
    # Anaconda/conda-forge prepends ' | packaged by ... | ' between the
    # X.Y.Z release and the (main, date, ...) tuple. Drop it so the stdlib
    # platform._sys_version regex matches.
    pattern = re.compile(
        r"^(?P<release>\d+\.\d+\.\d+(?:[a-z]+\d+)?)\s*\|[^|]*\|\s*",
    )
    return pattern.sub(r"\g<release> ", version, count=1)


def _patch_sys_version() -> None:
    cleaned = _strip_anaconda_decoration(sys.version)
    if cleaned == sys.version:
        return
    sys.version = cleaned


_patch_sys_version()
