"""Tests for podcodex.core._utils: timestamps, speaker placeholders, path
components."""

import pytest

from podcodex.core._utils import format_hms, parse_time


def test_format_hms_under_hour():
    assert format_hms(0) == "0m00"
    assert format_hms(578) == "9m38"
    assert format_hms(3599) == "59m59"


def test_format_hms_hour_and_over():
    assert format_hms(3600) == "1h00m00"
    assert format_hms(4186) == "1h09m46"


def test_format_hms_floors_not_rounds():
    # Fractional seconds truncate down (canonical wiki floor form), never up:
    # a start timestamp must not point past the passage it marks.
    assert format_hms(1.5) == "0m01"
    assert format_hms(292.787) == "4m52"
    assert format_hms(292.999) == "4m52"


def test_parse_time_seconds_forms():
    assert parse_time(4186) == 4186.0
    assert parse_time(4186.0) == 4186.0
    assert parse_time("4186") == 4186.0


def test_parse_time_clock_forms_equivalent():
    assert parse_time("1h09m46") == 4186.0
    assert parse_time("69m46") == 4186.0
    assert parse_time("9m38") == 578.0


def test_parse_time_rejects_out_of_range_fields():
    with pytest.raises(ValueError):
        parse_time("1h09m60")
    with pytest.raises(ValueError):
        parse_time("1h60m00")
    with pytest.raises(ValueError):
        parse_time("1:09:60")
    with pytest.raises(ValueError):
        parse_time("1:60:00")


def test_parse_time_colon_forms():
    """MCP clients pass "1:05"; get_context answered "unrecognized time"."""
    assert parse_time("1:05") == 65.0
    assert parse_time("69:46") == 4186.0
    assert parse_time("1:09:46") == 4186.0
    with pytest.raises(ValueError):
        parse_time("1:5")


def test_the_placeholder_and_an_empty_label_name_nobody():
    """The current placeholder is not a plausible human name, so declaring it
    changes nothing; an empty label is never a name, declared or not."""
    from podcodex.core._utils import NARRATOR_SPEAKER, is_unattributed

    assert is_unattributed(NARRATOR_SPEAKER) is True
    assert is_unattributed(NARRATOR_SPEAKER, {NARRATOR_SPEAKER}) is True
    assert is_unattributed("", {"Alice"}) is True
    assert is_unattributed("Alice") is False


@pytest.mark.legacy("narrator-rename")
def test_the_legacy_narrator_is_unattributed_unless_declared():
    """Pre-0.2.10 transcripts use "Narrator" as the placeholder, which a
    documentary can legitimately call someone: declared, it is a name, and
    it keeps its roster entry and airtime."""
    from podcodex.core._utils import (
        LEGACY_NARRATOR_SPEAKER,
        is_unattributed,
        speaker_airtime,
    )

    segs = [{"speaker": "Narrator", "start": 0.0, "end": 10.0, "text": "x"}]
    assert is_unattributed(LEGACY_NARRATOR_SPEAKER) is True
    assert is_unattributed("Narrator", {"Narrator"}) is False
    assert speaker_airtime(segs) == {}
    assert list(speaker_airtime(segs, {"Narrator"})) == ["Narrator"]


# ──────────────────────────────────────────────
# srt_to_segments / vtt_to_segments
# ──────────────────────────────────────────────
#
# The whole YouTube subtitle import path rides on these two parsers, and the
# empty-speaker footgun lives here: an untagged cue must come out as
# NARRATOR_SPEAKER, never as "". NARRATOR_SPEAKER's value itself changed in
# 0.2.10, so the tests assert against the constant, not the string.


@pytest.mark.parametrize("name", ["C:..", "c:evil", "D:", "a:b"])
def test_bad_path_component_refuses_a_windows_drive_prefix(name):
    """pathlib on Windows replaces or climbs out of the base for these."""
    from podcodex.core._utils import bad_path_component

    assert bad_path_component(name)


@pytest.mark.parametrize("name", ["Episode 3: the return", "ep: two", "show"])
def test_bad_path_component_allows_an_ordinary_colon(name):
    from podcodex.core._utils import bad_path_component

    assert not bad_path_component(name)


def test_atomic_write_fails_fast_on_an_unwritable_folder(tmp_path, monkeypatch):
    """tempfile.mkstemp on Windows retried a PermissionError up to TMP_MAX
    (~2 billion) times, spinning a core forever on a read-only folder."""
    import os

    from podcodex.core import _utils

    calls = []

    def deny(path, flags, mode=0o777):
        calls.append(path)
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(_utils.os, "open", deny)
    with pytest.raises(PermissionError):
        _utils.atomic_write(tmp_path / "show.toml", lambda p: p.write_text("x"))
    assert len(calls) == 1
    monkeypatch.undo()
    assert not os.listdir(tmp_path)


def test_atomic_write_retries_only_a_name_collision(tmp_path, monkeypatch):
    from podcodex.core import _utils

    real_open = _utils.os.open
    collisions = iter([True, True])

    def collide_twice(path, flags, mode=0o777):
        if next(collisions, False):
            raise FileExistsError(path)
        return real_open(path, flags, mode)

    monkeypatch.setattr(_utils.os, "open", collide_twice)
    target = tmp_path / "out.json"
    _utils.atomic_write(target, lambda p: p.write_text("ok"), durable=False)
    monkeypatch.undo()
    assert target.read_text() == "ok"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]
