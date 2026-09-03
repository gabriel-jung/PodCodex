"""Restore `.podcodex` archives into the local install.

Pure functions — no argparse, no prompts. CLI/API map their own UX onto
``ConflictPolicy`` before calling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
import tarfile
from collections.abc import Callable, Container
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
from podcodex.core._utils import bad_path_component

if TYPE_CHECKING:
    from podcodex.rag.index_store import IndexStore

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
                    f"model '{c.model}' (collection '{c.name}') not registered in this install. "
                    "Queries against this collection will fail until the model is added."
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
        # ``show.folder`` is attacker-controlled (it comes from the archive
        # manifest); a name like ``../evil`` would let both the
        # REPLACE-policy ``rmtree`` and the extraction step operate outside
        # ``shows_dir``.
        if bad_path_component(show.folder):
            raise ArchiveCorruptError(
                f"unsafe folder name in manifest: {show.folder!r}"
            )
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
) -> list[tuple[str, str, str, str, int, str]]:
    """Validate collisions, return ``(name, show, model, chunker, dim, show_id)`` tuples.

    ``RENAME`` falls back to REPLACE for collection collisions: collection
    names embed the original show normalization, so renaming would break
    addressing. The pragmatic case is re-importing the same archive — the
    new extraction is the same data, so overwriting is safe.
    """
    from podcodex.rag.index_store import reserved_index_names

    existing = set(store.list_collections())
    # A name that collides with index state is not a traversal, so
    # bad_path_component passes it; see reserved_index_names.
    reserved = reserved_index_names()
    out: list[tuple[str, str, str, str, int, str]] = []
    for show in manifest.shows:
        for c in show.collections:
            # Manifest data is attacker-controlled; reject a hostile name
            # here, before extraction, instead of failing mid-import when
            # ensure_collection rejects it after files already landed.
            if bad_path_component(c.name) or c.name in reserved:
                raise ArchiveCorruptError(
                    f"unsafe collection name in manifest: {c.name!r}"
                )
            if c.name in existing:
                if on_conflict == ConflictPolicy.ABORT:
                    raise ConflictError(f"collection '{c.name}' already exists")
                resolved[f"collection:{c.name}"] = "replaced"
            out.append((c.name, show.name, c.model, c.chunker, c.dim, show.id))
    return out


def _mint_ids_for_legacy_shows(
    manifest: Manifest,
    folder_map: dict[str, str],
    shows_dir: Path | None,
    store: IndexStore,
) -> None:
    """Give a v1 archive's shows an identity on the way in.

    Archives written before ``ShowEntry.id`` existed carry collections keyed
    only by display name. Left alone they would import as orphans: no id to
    resolve them by, and a later rename would lose them exactly as before.

    An index-only import of a legacy archive has no show folder to mint into,
    so its collections stay unidentified until the show is registered and the
    ordinary migration picks them up.
    """
    from podcodex.ingest.show import ensure_show_id, load_show_meta, save_show_meta

    for show in manifest.shows:
        if not shows_dir:
            continue
        folder_name = folder_map.get(show.folder)
        if not folder_name:
            continue
        folder = shows_dir / folder_name
        if not folder.is_dir():
            continue

        if show.id:
            # Carried identity wins, so re-importing a show onto another
            # machine lands on the same collections it left with.
            meta = load_show_meta(folder)
            if meta is not None and meta.id != show.id:
                meta.id = show.id
                save_show_meta(folder, meta)
            sid = show.id
        else:
            sid = ensure_show_id(folder)

        for c in show.collections:
            try:
                store.set_collection_identity(c.name, show_id=sid, show=show.name)
            except Exception:
                logger.opt(exception=True).warning(
                    f"could not stamp identity on imported collection {c.name!r}"
                )


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


def _confine(target: Path, root: Path) -> Path | None:
    """Return *target* only if it stays inside *root* (Zip-Slip guard).

    ``root`` must already be resolved; callers resolve it once, not per
    archive member.
    """
    if not target.resolve().is_relative_to(root):
        logger.warning(f"skipping archive member outside {root}: {target}")
        return None
    return target


def _collection_member(first_segment: str, collection_names: Container[str]) -> bool:
    """True when a ``lancedb/`` member's first segment belongs to a planned collection.

    Same shape as ``_purge_collection_from_disk``: the table directory
    ``<name>.lance`` or a sidecar ``<name>.<ext>``. Anything else under
    ``lancedb/`` (``_show_passwords.lance``, ``_collections.lance``,
    ``index_origin.json``, a collection the manifest never declared) is
    index state the archive has no business replacing.
    """
    from podcodex.rag.index_store import reserved_index_names

    # Defence in depth. _plan_collections already refuses a manifest that
    # declares one of these, so reaching here means the plan was built by
    # some other path; the reserved files stay out either way.
    stem = first_segment.split(".", 1)[0]
    if stem in reserved_index_names():
        return False
    for name in collection_names:
        if first_segment == f"{name}.lance" or first_segment.startswith(name + "."):
            return True
    return False


def _resolve_target(
    member_name: str,
    show_roots: dict[str, Path] | None,
    index_root: Path,
    collection_names: Container[str],
) -> Path | None:
    """Map archive member path → target filesystem path. ``None`` to skip.

    ``show_roots`` maps each archive folder to its resolved destination
    root; a show member must stay inside its *own* folder, not merely
    inside the shows directory, so ``shows/a/../b/show.toml`` cannot
    overwrite a sibling show. ``lancedb/`` members are accepted only for
    the collections the manifest declared (see ``_collection_member``),
    and must stay inside ``index_root``. Everything else is skipped, so a
    crafted archive cannot write outside the roots it is allowed to fill.
    """
    if member_name == MANIFEST_FILENAME:
        return None
    if member_name.startswith("shows/"):
        if show_roots is None:
            return None
        rest = member_name[len("shows/") :]
        original_folder, _, tail = rest.partition("/")
        root = show_roots.get(original_folder)
        if root is None or not tail:
            return None
        return _confine(root / tail, root)
    if member_name.startswith("lancedb/"):
        rest = member_name[len("lancedb/") :]
        first, _, _tail = rest.partition("/")
        if not _collection_member(first, collection_names):
            logger.warning(
                f"skipping archive member outside collections: {member_name}"
            )
            return None
        return _confine(index_root / rest, index_root)
    return None


def _extract(
    archive_path: Path,
    shows_dir: Path | None,
    index_root: Path,
    folder_map: dict[str, str],
    collection_names: Container[str],
    progress: ProgressCallback | None,
) -> None:
    """Stream tar members to disk in one pass, rewriting paths via ``folder_map``."""
    # Resolve the containment roots once; _confine runs per member.
    show_roots = (
        {orig: (shows_dir / final).resolve() for orig, final in folder_map.items()}
        if shows_dir is not None
        else None
    )
    index_root = index_root.resolve()
    written = 0
    with tarfile.open(archive_path, mode="r:*") as tf:
        for member in tf:
            if not member.isfile() or member.name == MANIFEST_FILENAME:
                continue
            target = _resolve_target(
                member.name, show_roots, index_root, collection_names
            )
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
        FileNotFoundError, IndexOwnershipError (the archive carries
        collections and this index belongs to another machine).
    """
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    if manifest is None:
        manifest = _read_manifest(archive_path)

    if name and len(manifest.shows) != 1:
        raise ValueError("--name only valid for single-show bundles")
    if name and bad_path_component(name):
        raise ValueError(f"unsafe folder name: {name!r}")

    is_full = manifest.mode == Mode.FULL
    if is_full:
        if shows_dir is None:
            raise ValueError("shows_dir required for full bundle import")
        shows_dir = Path(shows_dir).resolve()
        shows_dir.mkdir(parents=True, exist_ok=True)

    # Imported here, not at module scope: index_store pulls pyarrow and
    # numpy (~150 ms), and podcodex.bundle's package __init__ puts this
    # module on the API's startup import path via routes/shows.py.
    from podcodex.rag.index_store import get_index_store

    store = get_index_store()

    # Ownership is checked once, up front. Every mode of this importer writes
    # to the index (new tables, `_collections` rows, and under REPLACE a table
    # drop), and on a replica the next sync from the owner erases all of it.
    # Discovering that halfway through is worse than not starting: the REPLACE
    # path used to catch the refused `delete_collection`, log a warning, purge
    # the table directory anyway and then find the stale `_collections` row
    # still there, leaving the collection registered with the old dim/model.
    if any(show.collections for show in manifest.shows):
        from podcodex.rag.index_origin import require_owner

        require_owner(store.path, "import a bundle into this index")

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

    _extract(
        archive_path,
        shows_dir,
        store.path,
        folder_map,
        frozenset(c[0] for c in collection_plan),
        progress,
    )

    store.reconnect()

    for col_name, show_name, model, chunker, dim, entry_id in collection_plan:
        store.ensure_collection(
            name=col_name,
            show=show_name,
            model=model,
            chunker=chunker,
            dim=dim,
            show_id=entry_id,
        )

    _mint_ids_for_legacy_shows(manifest, folder_map, shows_dir, store)

    return ImportResult(
        shows_dir=str(shows_dir) if shows_dir else "",
        mode=manifest.mode,
        shows_imported=list(folder_map.values()),
        collections_imported=[c[0] for c in collection_plan],
        conflicts_resolved=resolved,
    )
