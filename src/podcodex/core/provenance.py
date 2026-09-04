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

from loguru import logger


def _build_source_chain(
    audio_path: str | None,
    output_dir: str | None,
    step: str,
    model: str | None,
    mode: str | None,
) -> list[str] | None:
    """Build a source chain by looking up the input version's chain and appending this step.

    Returns e.g. ["youtube-subtitles", "ollama/qwen3:4b", "openai/gpt-4"].
    """
    try:
        from podcodex.core._utils import AudioPaths
        from podcodex.core.versions import get_latest_provenance

        p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

        # Find the input version — walk backwards through the pipeline
        input_prov = None
        if step == "corrected":
            input_prov = get_latest_provenance(p.base, "transcript")
        else:
            # Translate and others: try corrected first, then transcript
            input_prov = get_latest_provenance(
                p.base, "corrected"
            ) or get_latest_provenance(p.base, "transcript")

        # Get existing chain or start from the input's source
        prev_chain: list[str] = []
        if input_prov:
            input_params = input_prov.get("params") or {}
            prev_chain = list(input_params.get("source_chain", []))
            if not prev_chain:
                # Legacy: build chain from source field
                source = input_params.get("source")
                if source:
                    prev_chain = [source]

        # Append this step's identifier
        step_id = model or mode or step
        return prev_chain + [step_id] if prev_chain else None
    except Exception:
        logger.opt(exception=True).debug("source chain build failed for {}", audio_path)
        return None


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
) -> dict:
    """Build a standard provenance dict for version tracking.

    When *audio_path* or *output_dir* is provided and the step is not
    ``transcript``, a ``source_chain`` is built by looking up the input
    version's chain and appending this step's model/mode identifier.
    """
    params = dict(params) if params else {}
    # A hand-edited version is "validated" by definition, and the two flags
    # must agree: `is_edited` reads either, but only the type reaches the
    # filename, so a manual edit typed "raw" is indistinguishable from model
    # output once the DB is rebuilt from disk. Enforced here rather than at
    # each caller, which is how /translate/save-manual drifted.
    if manual_edit:
        ptype = "validated"
    if (
        step != "transcript"
        and "source_chain" not in params
        and (audio_path or output_dir)
    ):
        chain = _build_source_chain(
            audio_path, output_dir, step, model, params.get("llm_mode")
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
) -> dict:
    """Look up transcript provenance and return kwargs for correct_segments.

    Returns dict with ``source_lang``, ``engine``, ``engine_model``.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.correct import transcript_provenance_info
    from podcodex.core.versions import get_latest_provenance

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    tc_prov = get_latest_provenance(p.base, "transcript")
    tc_info = transcript_provenance_info(tc_prov)
    return {
        "source_lang": tc_info["language"] or fallback_source_lang,
        "engine": tc_info["source"],
        "engine_model": tc_info["model"],
    }
