"""Pins the key space the batch version picker rides on.

The picker (StepConfigEditor's "Custom" source) sends
``BatchRequest.source_version_ids``, a map keyed by the *same* strings as
``BatchRequest.audio_paths``. For an episode with audio that key is the audio
path; for a subtitle-only import, which has no audio file, it is
``{output_dir}.virtual``, minted by ``lib/episodeRef.ts:getEpisodeBatchPath``
on the frontend and ``core/_utils.py:virtual_audio_path`` on the backend.

That second key space is load-bearing but almost invisible: the string just
flows through ``AudioPaths.from_audio`` and works because of how that resolves
a stem, and the two languages agree only by convention. It has broken once (a
version map keyed by ``audio_path || id`` missed every subtitle-only episode,
286 of 566 in a real library) and the failure is silent: the batch runs, using
a version the user did not pick.

These tests pin the mechanism rather than the batch run itself, which needs
models and a network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcodex.core._utils import (
    VIRTUAL_AUDIO_SUFFIX,
    AudioPaths,
    virtual_audio_path,
)
from podcodex.core.versions import load_version_by_id, save_version

FRONTEND_SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
EPISODE_REF_FILE = FRONTEND_SRC / "lib" / "episodeRef.ts"

SEGMENTS = [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Narrator"}]
PROVENANCE = {"step": "transcript", "type": "asr", "model": "tiny", "params": {}}


@pytest.fixture
def episode(tmp_path):
    """A subtitle-only episode: an output dir with a version, and no audio."""
    show = tmp_path / "MyShow"
    base = show / "ep1" / "ep1"
    base.parent.mkdir(parents=True)
    version_id = save_version(base, "transcript", SEGMENTS, PROVENANCE)
    return show, base, version_id


def test_virtual_key_resolves_to_the_real_episode(episode):
    """``{output_dir}.virtual`` must land on the same base as a real path.

    This is the whole reason a subtitle-only episode can be batched at all:
    ``AudioPaths.from_audio`` treats the fake filename's stem as the episode
    stem, so the version lookup finds the versions saved under it.
    """
    show, base, _version_id = episode
    virtual = virtual_audio_path(show / "ep1")

    p = AudioPaths.from_audio(virtual)

    assert p.base == base
    assert not Path(virtual).exists()  # nothing is created for the fake name


def test_picked_version_loads_through_the_virtual_key(episode):
    """The picker's chosen id must resolve from the key the frontend sends."""
    show, _base, version_id = episode
    virtual = virtual_audio_path(show / "ep1")
    source_version_ids = {virtual: version_id}

    # Exactly what the batch loop does: look the key up, then resolve it.
    p = AudioPaths.from_audio(virtual)
    resolved = load_version_by_id(p.base, source_version_ids[virtual])

    assert resolved is not None, "the user's pick was silently dropped"
    segments, step = resolved
    assert step == "transcript"
    assert segments == SEGMENTS


def test_audio_path_key_resolves_the_same_way(tmp_path):
    """The with-audio half of the key space, for symmetry."""
    show = tmp_path / "MyShow"
    show.mkdir()
    audio = show / "ep1.mp3"
    audio.write_bytes(b"fake audio")
    base = show / "ep1" / "ep1"
    base.parent.mkdir(parents=True)
    version_id = save_version(base, "transcript", SEGMENTS, PROVENANCE)

    source_version_ids = {str(audio): version_id}

    p = AudioPaths.from_audio(str(audio))
    resolved = load_version_by_id(p.base, source_version_ids[str(audio)])

    assert p.base == base
    assert resolved is not None
    assert resolved[0] == SEGMENTS


def test_unknown_version_id_resolves_to_none(episode):
    """A deleted or stale pick must be detectable, not silently substituted.

    ``_batch_llm_step`` relies on this returning None to log and skip the
    episode instead of falling through to its own default version.
    """
    show, base, _version_id = episode

    assert load_version_by_id(base, "no-such-version") is None


def test_frontend_mints_the_same_suffix_python_owns():
    """Cross-language pin on the actual value, not on a code shape.

    ``VIRTUAL_AUDIO_SUFFIX`` is the Python owner; ``getEpisodeBatchPath`` is
    the TypeScript one. If either renames the suffix the other keeps
    "working" while every subtitle-only episode silently loses its picked
    version, which is how this broke before.
    """
    src = EPISODE_REF_FILE.read_text(encoding="utf-8")
    fn = src[src.index("export function getEpisodeBatchPath") :]
    assert f'"{VIRTUAL_AUDIO_SUFFIX}"' in fn, (
        f"getEpisodeBatchPath no longer appends {VIRTUAL_AUDIO_SUFFIX!r}; the "
        "batch key space changed and AudioPaths-based resolution must be rechecked"
    )
