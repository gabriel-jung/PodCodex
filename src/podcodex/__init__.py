"""Podcodex: Podcast transcription and intelligence.

Importing this package is passive — it gives you ``__version__`` and
nothing else. Application entry points (the bundled FastAPI sidecar,
the MCP stdio entry, ``podcodex-api``, ``podcodex-bot``) call exactly
one ``bootstrap_for_*()`` from :mod:`podcodex.bootstrap` before doing
real work to install logging and the platform monkey-patches.
"""

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("podcodex")
except PackageNotFoundError:  # editable install before metadata is registered
    __version__ = "0.0.0+unknown"
