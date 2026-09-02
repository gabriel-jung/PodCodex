"""The ``auto`` source ladder used by index, translate, synthesize and batch.

One definition (``versions.resolve_canonical_ref``) feeds every consumer, so
the verified pointer, the edited-beats-freshness rule for ``corrected`` and
the output_dir-only transcript rung all agree with the speaker roster.
"""

from podcodex.api.routes._helpers import _resolve_source_segments, load_best_source
from podcodex.core._utils import AudioPaths
from podcodex.core.pipeline_db import get_pipeline_db
from podcodex.core.versions import save_version, version_path

SEGS_T = [{"speaker": "A", "text": "raw transcript", "start": 0.0, "end": 1.0}]
SEGS_C = [{"speaker": "A", "text": "corrected", "start": 0.0, "end": 1.0}]
SEGS_E = [{"speaker": "A", "text": "hand edited", "start": 0.0, "end": 1.0}]


def _prov(step, type_="raw", manual_edit=False):
    return {
        "step": step,
        "type": type_,
        "model": None,
        "params": {},
        "manual_edit": manual_edit,
    }


def _episode(tmp_path):
    ep = tmp_path / "show" / "ep"
    ep.mkdir(parents=True)
    return ep, ep / "ep"


def test_auto_finds_transcript_for_output_dir_only_episode(tmp_path):
    """Subtitle-only imports have no audio; the transcript rung must still
    resolve from the output dir instead of rebuilding paths off a synthetic
    audio path that points one level too deep."""
    ep, base = _episode(tmp_path)
    save_version(base, "transcript", SEGS_T, _prov("transcript"))
    p = AudioPaths.from_audio(None, output_dir=str(ep))

    segs, label = _resolve_source_segments(p, "auto")
    assert (segs, label) == (SEGS_T, "transcript")
    assert load_best_source(output_dir=str(ep)) == SEGS_T


def test_explicit_transcript_for_output_dir_only_episode(tmp_path):
    ep, base = _episode(tmp_path)
    save_version(base, "transcript", SEGS_T, _prov("transcript"))
    p = AudioPaths.from_audio(None, output_dir=str(ep))

    assert _resolve_source_segments(p, "transcript") == (SEGS_T, "transcript")


def test_auto_prefers_edited_corrected_over_newer_raw(tmp_path):
    """Same 'edited beats freshness' pick the speaker roster makes."""
    ep, base = _episode(tmp_path)
    save_version(base, "transcript", SEGS_T, _prov("transcript"))
    save_version(
        base, "corrected", SEGS_E, _prov("corrected", "edited", manual_edit=True)
    )
    save_version(base, "corrected", SEGS_C, _prov("corrected"))
    p = AudioPaths.from_audio(None, output_dir=str(ep))

    assert _resolve_source_segments(p, "auto") == (SEGS_E, "corrected")


def test_auto_honours_verified_pointer(tmp_path):
    ep, base = _episode(tmp_path)
    vid = save_version(base, "transcript", SEGS_T, _prov("transcript"))
    save_version(base, "corrected", SEGS_C, _prov("corrected"))
    get_pipeline_db(base.parent.parent).set_verified(base.name, "transcript", vid)
    p = AudioPaths.from_audio(None, output_dir=str(ep))

    assert _resolve_source_segments(p, "auto") == (SEGS_T, "transcript")


def test_auto_walks_past_an_unreadable_canonical_file(tmp_path):
    """The canonical ref is DB-only; when its file was truncated by a sync
    conflict the older readable versions must still serve."""
    ep, base = _episode(tmp_path)
    save_version(base, "transcript", SEGS_T, _prov("transcript"))
    vid = save_version(base, "corrected", SEGS_C, _prov("corrected"))
    version_path(base, "corrected", vid).write_text("{not json")
    p = AudioPaths.from_audio(None, output_dir=str(ep))

    assert _resolve_source_segments(p, "auto") == (SEGS_T, "transcript")
