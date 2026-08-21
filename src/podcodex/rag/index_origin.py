"""podcodex.rag.index_origin — which machine owns an index directory.

The index is replicated one way, desktop to bot, with ``rsync --delete-after``
(see ``deploy/BOT.md``). Anything written into a replica is destroyed by the
next sync. Show passwords used to be written by three processes across two
machines, so a password set on the bot host disappeared silently.

The rule enforced here: in one-way replication, each record has exactly one
writer, and the replica is read-only for that record.

"That record" is deliberately narrow. Guarded are the records that exist only
in the index and that two machines can meaningfully disagree about: show
passwords, collection identity and labels, and collection deletion. Derived
data is exempt on purpose, because the bot legitimately repairs its own
replica: the pub_date and episode_title backfills, and the collection-meta
heal, all write during ordinary reads, and every one of them is regenerable
and overwritten wholesale by the next sync. Guarding those would turn a read
into a failure to protect data nobody can lose.

Ownership is a property of the index, not of which binary is running, so a
bot-only deployment that created its own index still owns it.

An index with no marker is **unowned**, and writes are allowed. Every index
predating this module is in that state, and "first process to open it claims
it" would let a bot host that happens to start first take ownership, after
which the desktop, the real owner, would refuse its own writes. The marker is
therefore written only when the index directory is created.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from podcodex.core.machine_id import machine_id

ORIGIN_FILENAME = "index_origin.json"


class IndexOwnershipError(RuntimeError):
    """Raised when a write targets an index this machine does not own."""


def read_origin(index_path: Path | str) -> str:
    """Return the owning machine id, or "" when unstamped or unreadable.

    A corrupt or half-transferred marker reads as unowned on purpose: an
    unreadable file must not lock the real owner out of its own index.
    """
    path = Path(index_path) / ORIGIN_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            f"Unreadable index origin marker at {path} ({exc}); treating index as unowned"
        )
        return ""
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("origin_id", "") or "").strip()


def _write_origin(index_path: Path | str, origin_id: str) -> None:
    from podcodex.core._utils import atomic_write

    path = Path(index_path) / ORIGIN_FILENAME
    body = json.dumps(
        {
            "origin_id": origin_id,
            "stamped_at": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda p: p.write_text(body, encoding="utf-8"), suffix=".json")


def stamp_origin_if_new(index_path: Path | str, was_empty: bool) -> None:
    """Stamp this machine as owner, but only for a freshly created index.

    Args:
        index_path: The index directory.
        was_empty: True when this process found the directory absent or
            empty, meaning it is creating the index rather than opening
            somebody else's.
    """
    if not was_empty or read_origin(index_path):
        return
    local = machine_id()
    try:
        _write_origin(index_path, local)
    except OSError as exc:
        logger.warning(f"Could not stamp index origin at {index_path}: {exc}")
        return
    logger.info(f"Index at {index_path} stamped as owned by {local}")


def claim_origin(index_path: Path | str) -> str:
    """Take ownership of an existing index. Returns the previous owner id."""
    previous = read_origin(index_path)
    local = machine_id()
    _write_origin(index_path, local)
    logger.warning(
        f"Index at {index_path} claimed by {local} (previous owner: {previous or 'none'}). "
        "Further rsync from the previous owner will overwrite this index."
    )
    return previous


def is_replica(index_path: Path | str) -> bool:
    """True when the index is stamped by a different machine than this one."""
    origin = read_origin(index_path)
    if not origin:
        return False
    return origin != machine_id()


def require_owner(index_path: Path | str, action: str) -> None:
    """Raise ``IndexOwnershipError`` when this machine may not write to the index.

    Args:
        index_path: The index directory.
        action: Human-readable description of the attempted write, used in
            the error message (for example "set a password").
    """
    if not is_replica(index_path):
        return
    raise IndexOwnershipError(
        f"Cannot {action}: this index is owned by machine "
        f"{read_origin(index_path)!r}, not {machine_id()!r}. "
        "It is a replica, and the next rsync from the owner would overwrite the change. "
        "Make the change on the owning machine, or run with --claim-index to take "
        "ownership here (after which syncing from the old owner will overwrite it)."
    )
