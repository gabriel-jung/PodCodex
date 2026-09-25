"""`core/json_store.JsonModelStore`: config.json, api_keys.json and
provider_profiles.json all go through it."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from podcodex.core.json_store import JsonModelStore


class _Cfg(BaseModel):
    folders: list[str] = []
    count: int = 0


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "config.json"
    return JsonModelStore(lambda: path, _Cfg), path


def test_missing_file_loads_defaults(store):
    s, _path = store
    assert s.load() == _Cfg()


def test_a_value_that_fails_validation_loads_defaults(store):
    """A mistyped field used to raise out of every load_config caller."""
    s, path = store
    path.write_text(json.dumps({"count": "many"}), encoding="utf-8")
    assert s.load() == _Cfg()


def test_saving_over_an_unreadable_file_keeps_it_aside(store):
    s, path = store
    path.write_text('{"folders": ["/shows/a"', encoding="utf-8")

    s.mutate(lambda cfg: setattr(cfg, "count", 1))

    (aside,) = path.parent.glob("config.json.corrupt-*")
    assert "/shows/a" in aside.read_text(encoding="utf-8")
    assert s.load().count == 1


def test_load_returns_a_copy(store):
    s, _path = store
    s.save(_Cfg(folders=["a"]))
    s.load().folders.append("leaked")
    assert s.load().folders == ["a"]


def test_a_failed_save_leaves_nothing_applied(store, monkeypatch):
    """mutate used to edit the cached object in place, so a failed save left
    the change visible to every reader until restart."""
    s, _path = store
    s.save(_Cfg(folders=["a"]))

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("podcodex.core._utils.atomic_write", boom)
    with pytest.raises(OSError):
        s.mutate(lambda cfg: cfg.folders.append("b"))
    monkeypatch.undo()

    assert s.load().folders == ["a"]


def test_returning_false_skips_the_save(store):
    s, path = store
    s.mutate(lambda cfg: False)
    assert not path.exists()


def test_a_read_error_aborts_the_mutation(store, monkeypatch):
    """A file locked by a sync tool is not a corrupt file: editing defaults
    and saving them would overwrite (or move aside) the real config."""
    from pathlib import Path

    s, path = store
    s.save(_Cfg(folders=["/shows/a"]))
    s.invalidate()
    real_read = Path.read_text

    def locked(self, *a, **k):
        if self == path:
            raise PermissionError("locked by another process")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", locked)
    with pytest.raises(OSError):
        s.mutate(lambda cfg: cfg.folders.append("/shows/b"))
    assert s.load() == _Cfg()  # readers still get defaults
    monkeypatch.undo()

    assert s.load().folders == ["/shows/a"]
    assert not list(path.parent.glob("*.corrupt-*"))
