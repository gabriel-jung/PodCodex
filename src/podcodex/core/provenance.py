"""Version provenance: what produced a seglist, and what it came from.

Domain logic, not transport. It used to live in ``api/routes/_helpers``,
which meant ``core`` and ``rag`` imported the API package to build a
provenance dict — and because ``api/routes/__init__`` eagerly imports all
nineteen route modules, a subprocess step worker loaded the whole API
surface plus fastapi to do it. On an install without the desktop extra
(``deploy/BOT.md``'s bot+rag+cpu has no fastapi at all) that was not a cost
but a crash: ``podcodex-reindex``, which ``deploy/SMOKE.md`` documents,
raised ``ModuleNotFoundError: fastapi``.

``_helpers`` re-exports every name here, so route modules are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from podcodex.core.source import SourceRef, SourceVersion


def _source_provenance(
    audio_path: str | None,
    output_dir: str | None,
    source: SourceRef | SourceVersion | None,
) -> dict | None:
    """Provenance of the version a step consumed, or None when unknown."""
    if source is None or not (audio_path or output_dir):
        return None
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import get_version_provenance

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return get_version_provenance(p.base, source.version_id, source.step)


def _build_source_chain(
    input_prov: dict | None, step: str, model: str | None, mode: str | None
) -> list[str] | None:
    """The input version's chain with this step appended.

    Returns e.g. ["youtube-subtitles", "ollama/qwen3:4b", "openai/gpt-4"], or
    None when the input carries no chain.
    """
    if not input_prov:
        return None
    input_params = input_prov.get("params") or {}
    prev_chain = list(input_params.get("source_chain", []))
    if not prev_chain and input_params.get("source"):
        # Legacy: build chain from source field
        prev_chain = [input_params["source"]]
    step_id = model or mode or step
    return prev_chain + [step_id] if prev_chain else None


def transcribe_prov_params(
    diarize: bool, source: str = "whisper", model: str | None = None, **extra: object
) -> dict:
    """Build provenance params for a transcribe step.

    Also builds a source_chain entry like ``"whisper/large-v3-turbo, diarized"``.
    """
    d: dict = {"diarize": diarize, "source": source}
    # Build a descriptive source chain entry for downstream steps
    label = f"{source}/{model}" if model else source
    if diarize:
        label += ", diarized"
    d["source_chain"] = [label]
    d.update(extra)
    return d


def llm_prov_params(
    mode: str,
    provider_profile: str | None = None,
    key_name: str | None = None,
    **extra: object,
) -> dict:
    """Build the LLM portion of provenance params."""
    d: dict = {"llm_mode": mode}
    if provider_profile:
        d["llm_provider_profile"] = provider_profile
    if key_name:
        d["llm_key_name"] = key_name
    d.update(extra)
    return d


def build_provenance(
    step: str,
    ptype: str = "raw",
    model: str | None = None,
    params: dict | None = None,
    manual_edit: bool = False,
    audio_path: str | None = None,
    output_dir: str | None = None,
    source: SourceRef | SourceVersion | None = None,
) -> dict:
    """Build a standard provenance dict for version tracking.

    *source* is the version the step consumed (``core.source.load_source``
    returns it). When given, the input's ``source_chain`` is extended with
    this step's model/mode identifier; the chain then describes what was
    actually read, not whichever version is newest by now.
    """
    params = dict(params) if params else {}
    # A hand-edited version is "validated" by definition, and the two flags
    # must agree: `is_edited` reads either, but only the type reaches the
    # filename, so a manual edit typed "raw" is indistinguishable from model
    # output once the DB is rebuilt from disk. Enforced here rather than at
    # each caller, which is how /translate/save-manual drifted.
    if manual_edit:
        ptype = "validated"
    if step != "transcript" and "source_chain" not in params:
        chain = _build_source_chain(
            _source_provenance(audio_path, output_dir, source),
            step,
            model,
            params.get("llm_mode"),
        )
        if chain:
            params["source_chain"] = chain
    return {
        "step": step,
        "type": ptype,
        "model": model,
        "params": params,
        "manual_edit": manual_edit,
    }


def build_edit_provenance(
    step: str,
    audio_path: str | None,
    output_dir: str | None,
) -> dict:
    """Build provenance for a manual edit by inheriting from the latest version of the same step.

    Edited versions keep the same model/params/source_chain as their parent
    so their label reflects the pipeline that produced them, just marked as
    ``type=validated`` + ``manual_edit=True``.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import get_latest_provenance

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    parent = get_latest_provenance(p.base, step) or {}
    return {
        "step": step,
        "type": "validated",
        "model": parent.get("model"),
        "params": dict(parent.get("params") or {}),
        "manual_edit": True,
    }


def enrich_correct_kwargs(
    audio_path: str | None,
    output_dir: str | None,
    fallback_source_lang: str,
    source: SourceRef | SourceVersion,
) -> dict:
    """Correct-prompt kwargs from the provenance of the transcript being corrected.

    Returns dict with ``source_lang``, ``engine``, ``engine_model``.
    """
    from podcodex.core.correct import transcript_provenance_info

    tc_info = transcript_provenance_info(
        _source_provenance(audio_path, output_dir, source)
    )
    return {
        "source_lang": tc_info["language"] or fallback_source_lang,
        "engine": tc_info["source"],
        "engine_model": tc_info["model"],
    }


def save_llm_run(
    step: str,
    save,
    *,
    source,
    llm,
    audio_path: str,
    output_dir: str | None,
    records: list[dict],
    **params: object,
) -> str:
    """Save an auto correct / translate run and record its batch outcomes.

    The shared tail of ``correct_and_save`` and ``translate_and_save``:
    provenance from the consumed *source* and the *llm* run, the save
    (``save(provenance) -> version_id``), then the ``llm_failures.json``
    section written once, carrying the id of the version its batch indices
    describe. Returns the version id.
    """
    from podcodex.core.llm_failures import record_run

    provenance = build_provenance(
        step,
        source=source,
        model=llm.model,
        audio_path=audio_path,
        output_dir=output_dir,
        params=llm_prov_params(llm.mode, **params),
    )
    version_id = save(provenance)
    record_run(
        audio_path,
        output_dir,
        step,
        model=llm.model,
        mode=llm.mode,
        records=records,
        version_id=version_id,
    )
    return version_id
