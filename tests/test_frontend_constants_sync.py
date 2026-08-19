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


# ── Loopback auth constants ─────────────────────────────────────────────
# Python: podcodex.core.api_token / podcodex.api.app. Mirrors: the frontend
# client (header + query param + CSRF pair), the Vite dev proxy (header +
# token filename), and the Tauri shell (token filename).

CLIENT_FILE = FRONTEND_SRC / "api" / "client.ts"
VITE_CONFIG_FILE = FRONTEND_SRC.parent / "vite.config.ts"
TAURI_LIB_FILE = FRONTEND_SRC.parents[1] / "src-tauri" / "src" / "lib.rs"


def _client_const(name: str) -> str:
    m = re.search(rf'const {name}\s*=\s*"([^"]*)"', _ts_src(CLIENT_FILE))
    assert m, f'could not parse `const {name} = "..."` in client.ts'
    return m.group(1)


def test_token_constants_match() -> None:
    from podcodex.core.api_token import TOKEN_HEADER, TOKEN_QUERY_PARAM

    assert _client_const("TOKEN_HEADER") == TOKEN_HEADER
    assert _client_const("TOKEN_QUERY_PARAM") == TOKEN_QUERY_PARAM


def test_csrf_constants_match() -> None:
    from podcodex.api.app import CSRF_HEADER, CSRF_VALUE

    assert _client_const("CSRF_HEADER") == CSRF_HEADER
    assert _client_const("CSRF_VALUE") == CSRF_VALUE


def test_vite_proxy_mirrors_token_header_and_filename() -> None:
    from podcodex.core.api_token import TOKEN_FILENAME, TOKEN_HEADER

    src = _ts_src(VITE_CONFIG_FILE)
    assert f'"{TOKEN_HEADER}"' in src, "vite proxy must inject the token header"
    assert f'"{TOKEN_FILENAME}"' in src, "vite proxy must read the token file"


def test_tauri_shell_mirrors_token_filename() -> None:
    from podcodex.core.api_token import TOKEN_FILENAME

    src = _ts_src(TAURI_LIB_FILE)
    assert f'"{TOKEN_FILENAME}"' in src, "get_api_token must read the token file"
