"""One pydantic model persisted as a JSON file, cached by mtime.

The single facility behind ``config.json``, ``api_keys.json`` and
``provider_profiles.json``. Each used to carry its own copy of the cache, and
the copies drifted on whether they returned the cached object or a copy. They
also shared one bad habit: a file that failed to parse loaded as defaults,
and the next save wrote those defaults over it, erasing every registered show
folder or stored key.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

from loguru import logger
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


class JsonModelStore(Generic[M]):
    """Load, save and mutate one JSON-backed model.

    ``load`` returns a deep copy, so callers may modify what they get; only
    ``save`` (or ``mutate``) changes what later loads see. A file that exists
    but cannot be parsed or validated loads as defaults with a warning, and
    the next save first moves it aside as ``<name>.corrupt-<timestamp>``
    rather than overwriting it.
    """

    def __init__(
        self,
        path: Callable[[], Path],
        model: type[M],
        *,
        migrate: Callable[[dict], dict] | None = None,
        file_mode: int | None = None,
    ) -> None:
        self._path = path
        self._model = model
        self._migrate = migrate
        self._file_mode = file_mode
        self._cache: tuple[Path, float, M] | None = None
        self._corrupt = False
        # Serializes load-modify-save (see mutate). Route handlers run on
        # FastAPI's threadpool, so two edits could interleave and lose one.
        self._lock = threading.RLock()

    def invalidate(self) -> None:
        self._cache = None

    def load(self) -> M:
        """The stored value (a copy), or defaults when missing or unusable.

        Never raises: a transient read error also yields defaults here, for
        readers only. Writers go through :meth:`mutate`, which refuses them.
        """
        return self._load(strict=False)

    def _load(self, *, strict: bool) -> M:
        from podcodex.core._utils import mtime_settled

        path = self._path()
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return self._model()
        except OSError:
            if strict:
                raise
            mtime = -1.0

        cached = self._cache
        if cached is not None and cached[0] == path and cached[1] == mtime:
            return cached[2].model_copy(deep=True)

        try:
            value = self._read(path)
        except OSError as exc:
            if strict:
                raise
            logger.warning("{} could not be read ({}); using defaults", path, exc)
            return self._model()
        if value is None:
            self._corrupt = True
            return self._model()
        if mtime >= 0 and mtime_settled(mtime):
            self._cache = (path, mtime, value)
        return value.model_copy(deep=True)

    def _read(self, path: Path) -> M | None:
        """Parse *path*; None (logged) when its content is unusable.

        Raises OSError when the file cannot be read at all (locked by a sync
        tool or antivirus, permissions): that says nothing about its
        content, so it must not be treated as corrupt.
        """
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            if self._migrate is not None and isinstance(data, dict):
                data = self._migrate(data)
            return self._model.model_validate(data)
        except ValueError as exc:  # JSONDecodeError and ValidationError
            logger.warning(
                "{} is unreadable ({}); using defaults. It will be kept as a "
                ".corrupt copy on the next save.",
                path,
                exc,
            )
            return None

    def save(self, value: M) -> None:
        """Write *value*. A corrupt file is moved aside first, never lost.

        Raises OSError if the current file cannot be read: whether it is
        corrupt is then unknown, and overwriting it could lose real data.
        """
        self._save(value, known_readable=False)

    def _save(self, value: M, *, known_readable: bool) -> None:
        from podcodex.core._utils import atomic_write

        path = self._path()
        with self._lock:
            if not known_readable and path.exists() and self._read(path) is None:
                aside = path.with_name(
                    f"{path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
                )
                path.replace(aside)
                logger.warning("Moved unreadable {} aside to {}", path, aside.name)

            def _write(tmp: Path) -> None:
                tmp.write_text(value.model_dump_json(indent=2), encoding="utf-8")
                if self._file_mode is not None:
                    try:
                        tmp.chmod(self._file_mode)
                    except OSError:
                        pass

            atomic_write(path, _write, suffix=".json")
            self._cache = None

    def mutate(self, fn: Callable[[M], bool | None]) -> M:
        """Load, apply *fn* in place, save; all under the store lock.

        *fn* works on a copy, so a failed save leaves nothing half-applied
        in memory. Return ``False`` from *fn* to skip the save. A file that
        exists but cannot be read raises OSError instead of letting *fn*
        edit defaults that would then be written over it.
        """
        with self._lock:
            self._corrupt = False
            value = self._load(strict=True)
            if fn(value) is not False:
                # _load just parsed the file (or found it missing): only a
                # corrupt one needs the second read that moves it aside.
                self._save(value, known_readable=not self._corrupt)
            return value
