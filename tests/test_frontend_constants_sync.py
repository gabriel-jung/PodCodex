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


def test_pipeline_defaults_converters_cover_every_server_field() -> None:
    """The store's bundleToServer/serverToBundle must name every field of the
    server's PipelineAppDefaults models (hf token excepted by design), so a
    field added on one side can't silently drop out of the round-trip."""
    from podcodex.core.app_config import (
        PipelineAppDefaults,
        PipelineLLMDefaults,
        PipelineTranscribeDefaults,
    )

    store_src = _ts_src(FRONTEND_SRC / "stores" / "pipelineConfigStore.ts")
    fields = (
        set(PipelineAppDefaults.model_fields)
        | set(PipelineTranscribeDefaults.model_fields)
        | set(PipelineLLMDefaults.model_fields)
    ) - {"transcribe", "llm"}
    for name in sorted(fields):
        assert f"{name}:" in store_src, (
            f"server field '{name}' missing from the store's converters"
        )


def test_pipeline_defaults_values_match_frontend_initial_bundle() -> None:
    """INITIAL_BUNDLE (TS) and PipelineAppDefaults (Python) hold the same
    default values on purpose: a never-saved install (sentinel None) must
    behave identically on the server and in the pre-hydration client. The
    duplication is deliberate; this pins the values together."""
    from podcodex.core.app_config import PipelineAppDefaults

    store_src = _ts_src(FRONTEND_SRC / "stores" / "pipelineConfigStore.ts")
    m = re.search(
        r"const INITIAL_BUNDLE: ConfigBundle = \{(.*?)\n\};", store_src, re.DOTALL
    )
    assert m, "could not locate the INITIAL_BUNDLE literal"
    block = m.group(1)

    def ts_value(key: str):
        vm = re.search(rf"\b{key}: (\"[^\"]*\"|[\d.]+|true|false|null)", block)
        assert vm, f"could not parse INITIAL_BUNDLE.{key}"
        raw = vm.group(1)
        if raw.startswith('"'):
            return raw[1:-1]
        literals = {"true": True, "false": False, "null": None}
        return literals[raw] if raw in literals else float(raw)

    d = PipelineAppDefaults().model_dump()
    expected = {
        "modelSize": d["transcribe"]["model_size"],
        "batchSize": d["transcribe"]["batch_size"],
        "diarize": d["transcribe"]["diarize"],
        "clean": d["transcribe"]["clean"],
        "numSpeakers": d["transcribe"]["num_speakers"],
        "language": d["transcribe"]["language"],
        "mode": d["llm"]["mode"],
        "providerProfile": d["llm"]["provider_profile"],
        "keyName": d["llm"]["key_name"],
        "context": d["llm"]["context"],
        "sourceLang": d["llm"]["source_lang"],
        "batchMinutes": d["llm"]["batch_minutes"],
        "engine": d["engine"],
        "targetLang": d["target_lang"],
        "indexModel": d["index_model"],
        "indexChunker": d["index_chunker"],
        "transcribePreset": d["transcribe_preset"],
        "llmPreset": d["llm_preset"],
        "llmPresetTouched": d["llm_preset_touched"],
        "indexPreset": d["index_preset"],
    }
    for key, want in expected.items():
        got = ts_value(key)
        if isinstance(want, float) or isinstance(got, float):
            assert got == pytest.approx(want), f"{key}: TS {got!r} != Python {want!r}"
        else:
            assert got == want, f"{key}: TS {got!r} != Python {want!r}"


def test_no_frontend_api_path_triggers_a_redirect() -> None:
    """Every literal API path in the frontend must hit a route exactly.

    A path that misses by a trailing slash still "works" in the packaged app
    (FastAPI 307s and the browser follows it same-origin), but in dev the
    redirect's Location is absolute and points at the backend, so the browser
    leaves the Vite proxy — and with it the injected auth token — and gets a
    401 that React Query swallows. The symptom is a silently empty page.
    """
    import re

    from tests.fixtures.api_client import make_client

    paths = set()
    for f in (FRONTEND_SRC / "api").glob("*.ts"):
        paths.update(re.findall(r'"(/api/[a-zA-Z0-9/_-]*)"', f.read_text()))
    assert paths, "no API paths parsed out of frontend/src/api"

    with pytest.MonkeyPatch.context() as mp:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client = make_client(Path(tmp), mp)
            redirecting = sorted(
                p
                for p in paths
                if client.get(p, follow_redirects=False).status_code in (307, 308)
            )

    assert not redirecting, (
        "these frontend paths redirect (usually a missing trailing slash); "
        f"they break `make dev-no-tauri`: {redirecting}"
    )
