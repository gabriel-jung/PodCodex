"""HTTP routes for `.podcodex` archive export/import.

Wraps :mod:`podcodex.bundle` for the desktop frontend. The Tauri shell
opens save/open dialogs natively and passes filesystem paths to these
endpoints — no multipart upload, no streaming required for v1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from podcodex.bundle import (
    ArchiveCorruptError,
    ArchivePreview,
    ConflictError,
    ExportResult,
    ImportResult,
    ManifestVersionError,
    export_index,
    export_show,
    import_archive,
    preview_archive,
)
from podcodex.bundle.conflicts import resolve_policy

router = APIRouter()


class ExportShowRequest(BaseModel):
    show_folder: str  # absolute or registered path
    output_path: str
    with_audio: bool = False
    index_only: bool = False


class ExportIndexRequest(BaseModel):
    show_folders: list[str]  # absolute paths
    output_path: str


class PreviewRequest(BaseModel):
    archive_path: str


class ImportRequest(BaseModel):
    archive_path: str
    shows_dir: str | None = None
    name: str | None = None
    on_conflict: Literal["auto", "rename", "replace", "abort"] = "auto"


@router.post("/export-show", response_model=ExportResult)
def post_export_show(req: ExportShowRequest) -> ExportResult:
    try:
        return export_show(
            Path(req.show_folder),
            Path(req.output_path),
            with_audio=req.with_audio,
            index_only=req.index_only,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/export-index", response_model=ExportResult)
def post_export_index(req: ExportIndexRequest) -> ExportResult:
    try:
        return export_index(
            [Path(p) for p in req.show_folders],
            Path(req.output_path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/preview", response_model=ArchivePreview)
def post_preview(req: PreviewRequest) -> ArchivePreview:
    try:
        return preview_archive(Path(req.archive_path))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ManifestVersionError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ArchiveCorruptError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/import", response_model=ImportResult)
def post_import(req: ImportRequest) -> ImportResult:
    try:
        preview = preview_archive(Path(req.archive_path))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ManifestVersionError, ArchiveCorruptError) as exc:
        raise HTTPException(422, str(exc)) from exc

    policy = resolve_policy(req.on_conflict, preview.manifest.mode)

    try:
        return import_archive(
            Path(req.archive_path),
            shows_dir=Path(req.shows_dir) if req.shows_dir else None,
            name=req.name,
            on_conflict=policy,
            manifest=preview.manifest,
        )
    except ConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ManifestVersionError, ArchiveCorruptError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
