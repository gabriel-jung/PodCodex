"""Guard against speaker-label drift between Python and TypeScript.

Backend defines NARRATOR_SPEAKER / BREAK_SPEAKER / REMOVE_SPEAKER /
UNKNOWN_SPEAKERS in ``podcodex.core._utils``. Frontend mirrors them in
``frontend/src/lib/speakers.ts``. These tests parse the .ts file with a
regex and assert each constant matches the Python value.

If a constant changes on one side, update both — and this test will tell
you which side fell behind.
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

FRONTEND_FILE = (
    Path(__file__).resolve().parents[1] / "frontend" / "src" / "lib" / "speakers.ts"
)


@cache
def _frontend_src() -> str:
    assert FRONTEND_FILE.exists(), f"missing {FRONTEND_FILE}"
    return FRONTEND_FILE.read_text(encoding="utf-8")


def _string_const(name: str) -> str:
    m = re.search(rf'export const {name}\s*=\s*"([^"]*)"', _frontend_src())
    assert m, f'could not parse `export const {name} = "..."` in speakers.ts'
    return m.group(1)


def _set_const(name: str) -> set[str]:
    m = re.search(
        rf"export const {name}[^=]*=\s*new Set\(\s*\[([^\]]*)\]",
        _frontend_src(),
        re.DOTALL,
    )
    assert m, f"could not parse `export const {name} = new Set([...])` in speakers.ts"
    return set(re.findall(r'"([^"]*)"', m.group(1)))


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
