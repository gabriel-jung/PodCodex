"""Pydantic models for `.podcodex` archive manifests.

Pure data + validation. No I/O. Archive read/write lives in
``export.py`` and ``import_show.py``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

SCHEMA_VERSION = 1
"""Bumped on breaking manifest changes; importer rejects unknown majors."""

MANIFEST_FILENAME = "manifest.json"
"""Path of the manifest within the archive root."""


class Mode(StrEnum):
    """Bundle mode — drives import path and which directories are present."""

    FULL = "full"
    INDEX_ONLY = "index-only"


class CollectionEntry(BaseModel):
    """One LanceDB collection included in the bundle."""

    name: str  # canonical "{show}__{model}__{chunker}" key
    model: str
    chunker: str
    dim: int  # vector dimensionality (needed to register on import)
    rows: int


class ShowEntry(BaseModel):
    """One show packaged in the bundle."""

    name: str  # display name (from show.toml)
    folder: str  # source folder name (basename, not full path)
    audio_included: bool = False
    collections: list[CollectionEntry] = []


class Manifest(BaseModel):
    """Root manifest written to ``manifest.json`` at archive root."""

    schema_version: int = SCHEMA_VERSION
    mode: Mode
    podcodex_version: str
    exported_at: str  # ISO-8601 UTC
    shows: list[ShowEntry]


class ArchivePreview(BaseModel):
    """Subset of manifest data presented to a user before import (UI/CLI confirm)."""

    archive_path: str
    manifest: Manifest
    size_bytes: int
    embedder_warnings: list[str] = []  # e.g. "model 'bge-m3' not installed locally"


class ExportResult(BaseModel):
    """Returned from ``export_show`` / ``export_index``."""

    output_path: str
    size_bytes: int
    mode: Mode
    shows_exported: int
    collections_exported: int
    audio_included: bool


class ImportResult(BaseModel):
    """Returned from ``import_archive``."""

    shows_dir: str
    mode: Mode
    shows_imported: list[str]  # final folder names after rename/replace
    collections_imported: list[str]
    conflicts_resolved: dict[str, str]  # original_name -> action_taken


class ManifestVersionError(Exception):
    """Raised when archive manifest schema_version is not understood."""


class ArchiveCorruptError(Exception):
    """Raised when archive structure or manifest fails validation."""


def show_member_prefix(folder: str) -> str:
    """Tar member prefix for a show's folder content."""
    return f"shows/{folder}/"


def manifest_to_json(m: Manifest) -> str:
    """Serialize manifest as canonical JSON for archive embedding."""
    return m.model_dump_json(indent=2)


def manifest_from_json(text: str) -> Manifest:
    """Parse and validate manifest JSON from archive root.

    Raises:
        ManifestVersionError: schema_version newer than this build understands.
        ArchiveCorruptError: JSON malformed or required fields missing.
    """
    try:
        m = Manifest.model_validate_json(text)
    except Exception as exc:  # pydantic ValidationError or json error
        raise ArchiveCorruptError(f"manifest invalid: {exc}") from exc

    if m.schema_version > SCHEMA_VERSION:
        raise ManifestVersionError(
            f"archive schema_version={m.schema_version} newer than supported "
            f"(this build understands up to {SCHEMA_VERSION}). "
            "Upgrade podcodex to import this archive."
        )
    return m


def default_archive_filename(folder: str | None, mode: Mode) -> Path:
    """Default output filename when caller doesn't specify ``-o``.

    For multi-show or `--all` exports, ``folder`` should be ``None`` and the
    name falls back to a generic ``shows-index.podcodex``.
    """
    if folder is None:
        return Path("shows-index.podcodex")
    suffix = "-index.podcodex" if mode == Mode.INDEX_ONLY else ".podcodex"
    return Path(f"{folder}{suffix}")
