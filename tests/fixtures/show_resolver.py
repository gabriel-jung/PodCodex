"""Isolation for ``IndexStore._show_folder_resolver``.

The resolver is a *class* attribute, and the API registers a real one
process-wide the moment ``podcodex.rag.index_store`` is imported (see
``api/app.py``'s ``defer_until_imported``). So merely importing
``podcodex.api.app`` in one test module arms it for every test that runs
afterwards, and a module that clears it to ``None`` when it is done disarms
it for everything after that.

That matters because the resolver decides whether a store heals its own
collection metadata (``IndexStore._heal_collection_meta``): with one
registered, reads rewrite ``show`` / ``show_id`` / ``artwork_url`` from
``show.toml``. A leaked resolver therefore makes tests that assert on stored
collection metadata pass alone and fail in the suite, or the reverse.

Use in any module whose assertions depend on that state (this repo has no
root ``conftest.py`` by design, and the helpers here are plain callables
rather than pytest fixtures to match the rest of ``tests/fixtures``)::

    with show_folder_resolver(None):
        ...                                   # nothing heals
    with show_folder_resolver(lambda name: my_folder):
        ...                                   # heals from my_folder
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from podcodex.rag.index_store import IndexStore

Resolver = Callable[[str], "Path | str | None"] | None


@contextmanager
def show_folder_resolver(fn: Resolver) -> Iterator[Callable[[Resolver], None]]:
    """Install *fn* as the resolver, restoring the previous one on exit.

    Restores rather than clearing, so a module running after this one still
    sees whatever it expected. Yields a setter for tests that need to swap
    the resolver partway through.
    """
    previous = IndexStore._show_folder_resolver
    IndexStore.set_show_folder_resolver(fn)
    try:
        yield IndexStore.set_show_folder_resolver
    finally:
        IndexStore.set_show_folder_resolver(previous)
