"""Restore `.podcodex` archives into the local install.

Pure functions — no argparse, no prompts. CLI/API map their own UX onto
``ConflictPolicy`` before calling.
"""

from __future__ import annotations

import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from podcodex.bundle.conflicts import (
    ConflictError,
    ConflictPolicy,
    rename_suffix,
)
from podcodex.bundle.manifest import (
    MANIFEST_FILENAME,
    ArchiveCorruptError,
    ArchivePreview,
    ImportResult,
    Manifest,
    Mode,
    manifest_from_json,
)
from podcodex.rag.index_store import IndexStore, get_index_store

ProgressCallback = Callable[[str, float], None]


def _read_manifest(archive_path: Path) -> Manifest:
    """Open the archive only long enough to parse manifest.json."""
    with tarfile.open(archive_path, mode="r:*") as tf:
        try:
            f = tf.extractfile(MANIFEST_FILENAME)
        except KeyError as exc:
            raise ArchiveCorruptError(
                f"manifest.json missing from {archive_path.name}"
            ) from exc
        if f is None:
            raise ArchiveCorruptError(
                f"manifest.json is not a regular file in {archive_path.name}"
            )
        text = f.read().decode("utf-8")
    return manifest_from_json(text)


def _embedder_warnings(manifest: Manifest) -> list[str]:
    """Flag manifest models not registered in this install."""
    try:
        from podcodex.rag.defaults import MODELS
    except Exception:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for show in manifest.shows:
        for c in show.collections:
            if c.model in seen:
                continue
            seen.add(c.model)
            if c.model not in MODELS:
                out.append(
                    f"model '{c.model}' (collection '{c.name}') not registered in this install — "
                    "queries against this collection will fail until the model is added"
                )
    return out


def preview_archive(archive_path: Path) -> ArchivePreview:
    """Read manifest + summary without extracting any payload files."""
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    manifest = _read_manifest(archive_path)
    return ArchivePreview(
        archive_path=str(archive_path),
        manifest=manifest,
        size_bytes=archive_path.stat().st_size,
        embedder_warnings=_embedder_warnings(manifest),
    )


def _plan_folder_targets(
    manifest: Manifest,
    shows_dir: Path,
    name: str | None,
    on_conflict: ConflictPolicy,
    resolved: dict[str, str],
) -> dict[str, str]:
    """Map archive folder name → final folder name in ``shows_dir``."""
    existing = {p.name for p in shows_dir.iterdir() if p.is_dir()}
    out: dict[str, str] = {}
    for show in manifest.shows:
        target = name if name else show.folder
        if target in existing:
            if on_conflict == ConflictPolicy.RENAME:
                final = rename_suffix(target, existing)
                out[show.folder] = final
                resolved[f"folder:{target}"] = f"renamed:{final}"
                existing.add(final)
            elif on_conflict == ConflictPolicy.REPLACE:
                out[show.folder] = target
                resolved[f"folder:{target}"] = "replaced"
            else:
                raise ConflictError(f"folder '{target}' already exists in {shows_dir}")
        else:
            out[show.folder] = target
            existing.add(target)
    return out


def _plan_collections(
    manifest: Manifest,
    on_conflict: ConflictPolicy,
    store: IndexStore,
    resolved: dict[str, str],
) -> list[tuple[str, str, str, str, int]]:
    """Validate collection collisions, return planned ``(name, show, model, chunker, dim)`` tuples.

    ``RENAME`` falls back to REPLACE for collection collisions: collection
    names embed the original show normalization, so renaming would break
    addressing. The pragmatic case is re-importing the same archive — the
    new extraction is the same data, so overwriting is safe.
    """
    existing = set(store.list_collections())
    out: list[tuple[str, str, str, str, int]] = []
    for show in manifest.shows:
        for c in show.collections:
            if c.name in existing:
                if on_conflict == ConflictPolicy.ABORT:
                    raise ConflictError(f"collection '{c.name}' already exists")
                resolved[f"collection:{c.name}"] = "replaced"
            out.append((c.name, show.name, c.model, c.chunker, c.dim))
    return out


def _purge_collection_from_disk(name: str, index_root: Path) -> None:
    """Remove all on-disk entries for a collection (table directory + sidecar files)."""
    for entry in list(index_root.iterdir()):
        if entry.name == f"{name}.lance" or entry.name.startswith(name + "."):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                try:
                    entry.unlink()
                except OSError:
                    pass


def _resolve_target(
    member_name: str,
    shows_dir: Path | None,
    index_root: Path,
    folder_map: dict[str, str],
) -> Path | None:
    """Map archive member path → target filesystem path. ``None`` to skip."""
    if member_name == MANIFEST_FILENAME:
        return None
    if member_name.startswith("shows/"):
        if shows_dir is None:
            return None
        rest = member_name[len("shows/") :]
        original_folder, _, tail = rest.partition("/")
        final = folder_map.get(original_folder)
        if not final or not tail:
            return None
        return shows_dir / final / tail
    if member_name.startswith("lancedb/"):
        return index_root / member_name[len("lancedb/") :]
    return None


def _extract(
    archive_path: Path,
    shows_dir: Path | None,
    index_root: Path,
    folder_map: dict[str, str],
    progress: ProgressCallback | None,
) -> None:
    """Stream tar members to disk in one pass, rewriting paths via ``folder_map``."""
    written = 0
    with tarfile.open(archive_path, mode="r:*") as tf:
        for member in tf:
            if not member.isfile() or member.name == MANIFEST_FILENAME:
                continue
            target = _resolve_target(member.name, shows_dir, index_root, folder_map)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if member.mode:
                try:
                    target.chmod(member.mode & 0o7777)
                except OSError:
                    pass
            written += 1
            if progress and written % 10 == 0:
                progress(f"extract:{member.name}", -1.0)
    if progress:
        progress("extract:done", 1.0)


def import_archive(
    archive_path: Path,
    shows_dir: Path | None = None,
    *,
    name: str | None = None,
    on_conflict: ConflictPolicy = ConflictPolicy.RENAME,
    progress: ProgressCallback | None = None,
    manifest: Manifest | None = None,
) -> ImportResult:
    """Extract a `.podcodex` bundle into the local install.

    Args:
        archive_path: Path to a `.podcodex` archive.
        shows_dir: Where to write show folder content. Required for full
            mode; ignored for index-only.
        name: Override single-show folder name on disk. Only valid for
            single-show bundles.
        on_conflict: Resolution for folder/collection collisions.

            * ``RENAME`` — auto-suffix folder. Collection collisions raise.
            * ``REPLACE`` — overwrite existing folder + collections.
            * ``ABORT`` — raise on first collision.

        progress: Optional ``(message, fraction)`` callback.
        manifest: Pre-parsed manifest. Pass it in when the caller already
            ran :func:`preview_archive` to avoid a second tar open.

    Raises:
        ArchiveCorruptError, ManifestVersionError, ConflictError, ValueError,
        FileNotFoundError.
    """
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    if manifest is None:
        manifest = _read_manifest(archive_path)

    if name and len(manifest.shows) != 1:
        raise ValueError("--name only valid for single-show bundles")

    is_full = manifest.mode == Mode.FULL
    if is_full:
        if shows_dir is None:
            raise ValueError("shows_dir required for full bundle import")
        shows_dir = Path(shows_dir).resolve()
        shows_dir.mkdir(parents=True, exist_ok=True)

    store = get_index_store()
    resolved: dict[str, str] = {}

    folder_map = (
        _plan_folder_targets(manifest, shows_dir, name, on_conflict, resolved)
        if is_full
        else {}
    )
    collection_plan = _plan_collections(manifest, on_conflict, store, resolved)

    if is_full:
        for _original, final in folder_map.items():
            if resolved.get(f"folder:{final}") == "replaced":
                target = shows_dir / final
                if target.exists():
                    shutil.rmtree(target)

    for col_name, *_rest in collection_plan:
        if resolved.get(f"collection:{col_name}") != "replaced":
            continue
        try:
            store.delete_collection(col_name)
        except Exception as exc:
            logger.warning(f"failed to delete existing collection {col_name}: {exc}")
        _purge_collection_from_disk(col_name, store.path)

    _extract(archive_path, shows_dir, store.path, folder_map, progress)

    store.reconnect()

    for col_name, show_name, model, chunker, dim in collection_plan:
        store.ensure_collection(
            name=col_name, show=show_name, model=model, chunker=chunker, dim=dim
        )

    return ImportResult(
        shows_dir=str(shows_dir) if shows_dir else "",
        mode=manifest.mode,
        shows_imported=list(folder_map.values()),
        collections_imported=[c[0] for c in collection_plan],
        conflicts_resolved=resolved,
    )
