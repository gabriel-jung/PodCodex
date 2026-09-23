"""podcodex.rag.show_id_migration — move an index from name-keyed to id-keyed.

Collections and show passwords used to be keyed by the show's display name, so
renaming a show orphaned its whole index and stranded its bot password. Shows
now carry a stable id in ``show.toml`` (``ingest.show.ensure_show_id``) and the
index keys on that instead.

Two properties this has to have, both learned from how the index is deployed:

**App-owned.** Only the desktop app has show folders. The bot reads an rsynced
copy with no ``show.toml`` anywhere, so a migration running there would mint
ids that diverge from the app's. The gate is the show-folder resolver, which
only the API registers.

**Nothing is renamed.** LanceDB OSS raises ``NotImplementedError`` for
``rename_table``, and it turns out not to be needed: the collection name is
now an internal detail (``IndexStore.resolve_collection`` is the only way in),
so a legacy collection keeps its name-derived name forever and simply gains a
``show_id``. That makes the migration a metadata write per collection: instant
on a large index, idempotent, and with no half-copied table to reconcile.

The migration never runs on the boot path: ``IndexStore`` pulls lancedb and
pyarrow, which ``tests/test_startup_offloading.py`` keeps out of
``podcodex.api.app``. It hangs off the first store open instead.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def migrate_index_to_show_ids(store) -> int:
    """Give every collection and password row of every registered show an id.

    Idempotent and resumable: rows that already carry a ``show_id`` are
    skipped, so a crash halfway through is repaired by the next run.

    Args:
        store: An open ``IndexStore``.

    Returns:
        The number of collections migrated in this pass.
    """
    from podcodex.ingest.show import ensure_show_id, show_display
    from podcodex.ingest.show_registry import registered_folders
    from podcodex.rag.index_store import _norm_label

    info = store.get_all_collection_info()
    pending = {
        name: meta for name, meta in info.items() if not (meta.get("show_id") or "")
    }
    passwords = store.get_show_password_entries()
    # Keyed case-insensitively: the label that protected a show is only as
    # exact as the user's typing at the time.
    legacy_passwords: dict[str, tuple[str, dict]] = {}
    for label, entry in sorted(passwords.items()):
        if entry.get("show_id"):
            continue
        key = _norm_label(label)
        kept = legacy_passwords.setdefault(key, (label, entry))
        if kept[1]["password_hash"] != entry["password_hash"]:
            # Nothing says which one the bot's users know; keep the first
            # by label and say so, rather than letting row order decide.
            logger.warning(
                f"show-id migration: legacy passwords {kept[0]!r} and {label!r} "
                f"differ only in case or spacing; keeping {kept[0]!r}. Set the "
                "password again from the show's settings if that is wrong."
            )
    # A password can exist without any collection: a show may be protected
    # before it is ever indexed. Returning early on `pending` alone left such
    # a row name-keyed forever, which reads as unprotected in the app while
    # the bot still demands the password.
    if not pending and not legacy_passwords:
        return 0

    folders = registered_folders()
    if not folders:
        return 0

    # Built from the display name, the identity assumption being retired,
    # used one last time to undo itself.
    by_label: dict[str, Path] = {}
    for folder in folders:
        by_label.setdefault(show_display(folder), folder)

    migrated = 0
    claimed: set[str] = set()

    for label, folder in by_label.items():
        mine = {
            n: m
            for n, m in pending.items()
            if n not in claimed and _row_belongs(n, m, label, folder)
        }
        # The password is keyed by whichever name was current when it was set.
        # A show renamed before upgrading is adopted through its table name
        # (see _row_belongs), and its password is under the old label its
        # collection rows still carry, so look there too.
        pw_keys = [_norm_label(label)] + [
            _norm_label(m.get("show") or "") for m in mine.values()
        ]
        pw_key = next((k for k in pw_keys if k in legacy_passwords), None)
        if not mine and pw_key is None:
            continue

        sid = ensure_show_id(folder)

        for name in mine:
            store.set_collection_identity(name, show_id=sid, show=label)
            claimed.add(name)
            migrated += 1

        if pw_key is not None:
            old_label, entry = legacy_passwords.pop(pw_key)
            store.set_show_password(
                sid, entry["password_hash"], show_label=label, legacy_label=old_label
            )
            logger.info(
                f"show-id migration: rekeyed password for {old_label!r} to {sid!r}"
            )

    orphans = sorted(n for n in pending if n not in claimed)
    if orphans:
        logger.warning(
            "show-id migration: "
            f"{len(orphans)} collection(s) match no registered show and were left "
            f"untouched: {', '.join(orphans)}. They stay searchable under their "
            "stored name; re-register or rename a show to that name to adopt them."
        )

    if legacy_passwords:
        stranded = ", ".join(sorted(label for label, _ in legacy_passwords.values()))
        logger.warning(
            "show-id migration: password(s) for "
            f"{stranded} match no registered show and stay keyed by name. The bot "
            "still enforces them under that name; set the password again from "
            "the show's settings to attach it to the show."
        )

    if migrated:
        logger.info(f"show-id migration: migrated {migrated} collection(s)")
    return migrated


def _row_belongs(name: str, meta: dict, label: str, folder: Path) -> bool:
    """Whether a pre-id collection belongs to the show in *folder*.

    Two bridges, because the display name is the only link and it is exactly
    the thing users are free to change:

    1. The row's stored label still matches the show's current name. The
       ordinary case.
    2. The collection's *table name* starts with the show's current
       name-derived prefix. This is what rescues a show renamed **before**
       upgrading: the row then says "Alpha" while ``show.toml`` says "Beta",
       so bridge 1 misses, but the table is still named from whichever name
       was current when it was indexed.

    Bridge 2 is checked against the folder's own name too, since a show
    created from a folder is usually indexed under the name it was given
    there.
    """
    from podcodex.rag.store import _normalize_show

    if (meta.get("show") or "").strip().lower() == label.strip().lower():
        return True
    prefixes = {f"{_normalize_show(label)}__", f"{_normalize_show(folder.name)}__"}
    return any(name.startswith(p) for p in prefixes if p != "__")
