"""Shared version CRUD route factory for pipeline steps."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from podcodex.core._utils import AudioPaths, normalize_lang


def register_version_routes(
    router: APIRouter,
    step: str | None = None,
    *,
    lang_param: bool = False,
) -> None:
    """Add GET /versions, GET /versions/{id}, DELETE /versions/{id} to *router*.

    Args:
        step: Fixed step name (e.g. "transcript", "polished").
              If None, step is derived from the ``lang`` query param.
        lang_param: When True, all endpoints accept a ``lang`` query parameter.
    """

    def _resolve(audio_path: str, output_dir: str | None, lang: str | None):
        p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
        s = normalize_lang(lang) if lang_param and lang else step
        if not s:
            raise HTTPException(400, "Missing step or lang parameter")
        return p, s

    @router.get("/versions")
    async def list_step_versions(
        audio_path: str = Query(...),
        output_dir: str | None = Query(None),
        lang: str | None = Query(None),
    ) -> list[dict]:
        from podcodex.core.versions import list_versions

        p, s = _resolve(audio_path, output_dir, lang)
        return list_versions(p.base, s)

    @router.get("/versions/{version_id}")
    async def load_step_version(
        version_id: str,
        audio_path: str = Query(...),
        output_dir: str | None = Query(None),
        lang: str | None = Query(None),
    ) -> list[dict]:
        from podcodex.core.versions import load_version

        p, s = _resolve(audio_path, output_dir, lang)
        try:
            return load_version(p.base, s, version_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Version {version_id} not found")

    @router.delete("/versions/{version_id}")
    async def delete_step_version(
        version_id: str,
        audio_path: str = Query(...),
        output_dir: str | None = Query(None),
        lang: str | None = Query(None),
    ) -> dict:
        from podcodex.core.versions import delete_version

        p, s = _resolve(audio_path, output_dir, lang)
        if not delete_version(p.base, s, version_id):
            raise HTTPException(404, f"Version {version_id} not found")
        return {"status": "deleted", "version_id": version_id}
