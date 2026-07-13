"""Guard against constant drift between Python and TypeScript.

Backend constants mirrored by hand in the frontend:

- Speaker labels: ``podcodex.core._utils`` (NARRATOR_SPEAKER / BREAK_SPEAKER /
  REMOVE_SPEAKER / UNKNOWN_SPEAKERS) mirrored in ``frontend/src/lib/speakers.ts``.
- Speech density thresholds: ``podcodex.core.transcribe`` (MIN_DENSITY /
  MAX_DENSITY) mirrored in ``frontend/src/hooks/useSegmentFiltering.ts``.

These tests parse the .ts files with a regex and assert each constant matches
the Python value. If a constant changes on one side, update both, and this
test will tell you which side fell behind.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

from podcodex.core._utils import (
    BREAK_SPEAKER,
    NARRATOR_SPEAKER,
    REMOVE_SPEAKER,
    UNKNOWN_SPEAKERS,
)
from podcodex.core.transcribe import MAX_DENSITY, MIN_DENSITY

FRONTEND_SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
SPEAKERS_FILE = FRONTEND_SRC / "lib" / "speakers.ts"
FILTERING_FILE = FRONTEND_SRC / "hooks" / "useSegmentFiltering.ts"


@cache
def _ts_src(path: Path) -> str:
    assert path.exists(), f"missing {path}"
    return path.read_text(encoding="utf-8")


def _string_const(name: str) -> str:
    m = re.search(rf'export const {name}\s*=\s*"([^"]*)"', _ts_src(SPEAKERS_FILE))
    assert m, f'could not parse `export const {name} = "..."` in speakers.ts'
    return m.group(1)


def _set_const(name: str) -> set[str]:
    m = re.search(
        rf"export const {name}[^=]*=\s*new Set\(\s*\[([^\]]*)\]",
        _ts_src(SPEAKERS_FILE),
        re.DOTALL,
    )
    assert m, f"could not parse `export const {name} = new Set([...])` in speakers.ts"
    return set(re.findall(r'"([^"]*)"', m.group(1)))


def _number_const(name: str) -> float:
    m = re.search(rf"export const {name}\s*=\s*([\d.]+)\s*;", _ts_src(FILTERING_FILE))
    assert m, (
        f"could not parse `export const {name} = <number>` in useSegmentFiltering.ts"
    )
    return float(m.group(1))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NARRATOR_SPEAKER", NARRATOR_SPEAKER),
        ("BREAK_SPEAKER", BREAK_SPEAKER),
        ("REMOVE_SPEAKER", REMOVE_SPEAKER),
    ],
)
def test_string_const_matches(name: str, expected: str) -> None:
    assert _string_const(name) == expected


def test_unknown_speakers_matches() -> None:
    assert _set_const("UNKNOWN_SPEAKERS") == set(UNKNOWN_SPEAKERS)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("MIN_DENSITY", MIN_DENSITY),
        ("MAX_DENSITY", MAX_DENSITY),
    ],
)
def test_density_threshold_matches(name: str, expected: float) -> None:
    assert _number_const(name) == expected
