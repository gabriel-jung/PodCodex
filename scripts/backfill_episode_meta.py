"""One-shot backfill for stale .episode_meta.json + RAG chunk metadata.

YouTube ingest path historically wrote .episode_meta.json from flat-extraction
RSSEpisode (often missing pub_date / description / duration) and never refreshed
it after cache_youtube_subtitles backfilled the feed cache with the per-video
extraction. Result: chunks indexed without pub_date even though the data lived
in .feed_cache.json all along.

This script repairs both layers, in order:

  Phase A — refresh per-episode meta files
      Walk every configured show folder. For each episode subdir with an
      .episode_meta.json, look up the matching feed-cache entry by guid and
      fill in any field that the meta file has empty but the cache holds.

  Phase B — heal LanceDB chunk metadata
      Walk every collection. For each indexed episode, re-merge meta JSON
      and the scalar pub_date column from the (now refreshed) episode meta
      file and update rows in place. Avoids re-embedding.

Dry-run by default. Pass --apply to write changes.

Usage:
    .venv/bin/python scripts/backfill_episode_meta.py            # dry-run
    .venv/bin/python scripts/backfill_episode_meta.py --apply
    .venv/bin/python scripts/backfill_episode_meta.py --apply --show DataGen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from podcodex.ingest.rss import (  # noqa: E402
    EPISODE_META_FILE,
    RSSEpisode,
    clean_description,
    episode_stem as rss_episode_stem,
    fill_empty_fields,
    load_feed_cache,
    save_episode_meta,
)
from podcodex.ingest.show import load_show_meta  # noqa: E402


def _load_show_folders() -> list[Path]:
    from podcodex.api.routes.config import _load

    cfg = _load()
    return [Path(p) for p in cfg.show_folders if Path(p).is_dir()]


def _build_folder_map(folders: list[Path]) -> dict[str, Path]:
    """Map show name (lowercase) → folder, in one pass over the configured list."""
    out: dict[str, Path] = {}
    for folder in folders:
        meta = load_show_meta(folder)
        name = (meta.name if meta else None) or folder.name
        out[name.lower()] = folder
    return out


def phase_a_refresh_meta_files(
    folder_map: dict[str, Path], *, apply: bool, show_filter: str | None
) -> tuple[dict[str, int], dict[Path, dict[str, RSSEpisode]]]:
    """Refresh ``.episode_meta.json`` files from ``.feed_cache.json``.

    Returns ``(counters, meta_by_show)`` where ``meta_by_show[show_folder]``
    maps episode stem → up-to-date ``RSSEpisode``. Phase B consumes that
    map to skip re-reading the same files we just wrote.
    """
    counts = {"shows_seen": 0, "episodes_seen": 0, "updated": 0, "no_match": 0}
    meta_by_show: dict[Path, dict[str, RSSEpisode]] = {}
    for show_name, show_folder in folder_map.items():
        if show_filter and show_filter.lower() not in show_name:
            continue
        cached = load_feed_cache(show_folder)
        if not cached:
            print(f"[skip] {show_name}: no feed cache")
            continue
        cache_by_guid = {ep.guid: ep for ep in cached if ep.guid}
        cache_by_stem = {
            rss_episode_stem(ep, show_folder): ep for ep in cached if ep.guid
        }
        counts["shows_seen"] += 1
        per_show: dict[str, RSSEpisode] = {}
        meta_by_show[show_folder] = per_show
        print(f"\n=== {show_name} ({show_folder}) ===")

        for episode_dir in sorted(p for p in show_folder.iterdir() if p.is_dir()):
            meta_path = episode_dir / EPISODE_META_FILE
            if not meta_path.is_file():
                continue
            counts["episodes_seen"] += 1
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_ep = RSSEpisode(**raw)
            except (OSError, json.JSONDecodeError, TypeError):
                print(f"  [corrupt] {episode_dir.name}")
                continue
            cache_ep = cache_by_guid.get(meta_ep.guid) or cache_by_stem.get(
                episode_dir.name
            )
            if not cache_ep:
                counts["no_match"] += 1
                print(f"  [no-cache-match] {episode_dir.name}")
                continue
            changed = fill_empty_fields(
                meta_ep, cache_ep, prefer_longer_description=True
            )
            per_show[episode_dir.name] = meta_ep
            if not changed:
                continue
            counts["updated"] += 1
            print(f"  [update] {episode_dir.name}: {', '.join(changed)}")
            if apply:
                save_episode_meta(episode_dir, meta_ep)
    return counts, meta_by_show


def _meta_targets(meta_ep: RSSEpisode, stem: str) -> dict[str, object]:
    """Build the ``{meta_key: target_value}`` map a chunk should carry.

    Mirrors the keys ``podcodex.rag.chunker._meta_fields`` writes into the
    extras blob: ``episode_title``, ``pub_date``, ``episode_number``,
    ``description``. ``episode_title`` is omitted when it would equal the
    stem (the chunker uses the stem as the display fallback).
    """
    from podcodex.rag.index_store import _normalize_pub_date

    targets: dict[str, object] = {}
    pub = _normalize_pub_date(meta_ep.pub_date) or ""
    if pub:
        targets["pub_date"] = pub
    desc = (meta_ep.description or "").strip()
    if desc:
        targets["description"] = clean_description(desc)
    if meta_ep.episode_number is not None:
        targets["episode_number"] = meta_ep.episode_number
    title = (meta_ep.title or "").strip()
    if title and title != stem:
        targets["episode_title"] = title
    return targets


def phase_b_heal_chunks(
    folder_map: dict[str, Path],
    *,
    apply: bool,
    show_filter: str | None,
    meta_by_show: dict[Path, dict[str, RSSEpisode]],
) -> dict[str, int]:
    """Update LanceDB chunk meta + ``pub_date`` column from refreshed meta files."""
    from podcodex.ingest.rss import load_episode_meta
    from podcodex.rag.index_store import _escape, get_index_store

    counts = {
        "collections_seen": 0,
        "chunks_scanned": 0,
        "rows_updated": 0,
        "episodes_healed": 0,
    }
    store = get_index_store()
    info = store.get_all_collection_info()

    for collection, col_meta in info.items():
        show_name = (col_meta.get("show") or "").strip()
        if not show_name:
            continue
        if show_filter and show_filter.lower() not in show_name.lower():
            continue
        show_folder = folder_map.get(show_name.lower())
        if not show_folder:
            print(f"[skip-col] {collection}: no show folder mapped for {show_name!r}")
            continue
        counts["collections_seen"] += 1
        print(f"\n--- collection {collection} (show={show_name}) ---")

        table = store._table(collection)  # noqa: SLF001 — script-only
        try:
            rows = (
                table.search()
                .select(["episode", "chunk_index", "pub_date", "meta"])
                .limit(10_000_000)
                .to_list()
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [scan-failed] {exc}")
            continue
        counts["chunks_scanned"] += len(rows)

        by_episode: dict[str, list[dict]] = {}
        for r in rows:
            by_episode.setdefault(r.get("episode") or "", []).append(r)

        meta_cache = meta_by_show.get(show_folder, {})

        for stem, chunks in by_episode.items():
            if not stem:
                continue
            meta_ep = meta_cache.get(stem) or load_episode_meta(show_folder / stem)
            if not meta_ep:
                continue
            targets = _meta_targets(meta_ep, stem)
            if not targets:
                continue
            target_pub = targets.get("pub_date", "")

            # Per-row JSON merge: only chunks whose meta blob is missing
            # one of the target keys need an UPDATE. The pub_date scalar
            # column is uniform per episode so it gets a single batched
            # UPDATE outside this loop.
            rows_needing_meta_update: list[tuple[int, dict]] = []
            for r in chunks:
                try:
                    blob = json.loads(r.get("meta") or "{}")
                except Exception:  # noqa: BLE001
                    blob = {}
                blob_changed = False
                for key, value in targets.items():
                    cur = blob.get(key)
                    if cur is None or (isinstance(cur, str) and not cur.strip()):
                        blob[key] = value
                        blob_changed = True
                if blob_changed:
                    rows_needing_meta_update.append(
                        (int(r.get("chunk_index", -1)), blob)
                    )

            # Find chunks whose pub_date scalar column is empty.
            pub_col_chunk_indexes = [
                int(r.get("chunk_index", -1))
                for r in chunks
                if target_pub and not (r.get("pub_date") or "").strip()
            ]

            if not rows_needing_meta_update and not pub_col_chunk_indexes:
                continue

            counts["episodes_healed"] += 1
            counts["rows_updated"] += max(
                len(rows_needing_meta_update), len(pub_col_chunk_indexes)
            )
            print(
                f"  [heal] {stem}: meta={len(rows_needing_meta_update)} "
                f"pub_col={len(pub_col_chunk_indexes)}"
            )
            if not apply:
                continue
            stem_clause = f"episode = '{_escape(stem)}'"
            if pub_col_chunk_indexes:
                # Single batched UPDATE — the scalar value is identical
                # across every chunk of the episode.
                try:
                    table.update(
                        where=(
                            f"{stem_clause} AND (pub_date IS NULL OR pub_date = '')"
                        ),
                        values={"pub_date": target_pub},
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [update-failed pub_date] {stem}: {exc}")
            for ci, blob in rows_needing_meta_update:
                try:
                    table.update(
                        where=f"{stem_clause} AND chunk_index = {ci}",
                        values={"meta": json.dumps(blob, ensure_ascii=False)},
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [update-failed meta] {stem}#{ci}: {exc}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default is dry-run)",
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        help="Only process shows whose name contains this substring (case-insensitive)",
    )
    parser.add_argument(
        "--skip-meta",
        action="store_true",
        help="Skip phase A (only heal LanceDB chunks)",
    )
    parser.add_argument(
        "--skip-chunks",
        action="store_true",
        help="Skip phase B (only refresh .episode_meta.json files)",
    )
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    if args.show:
        print(f"Show filter: {args.show}")

    folders = _load_show_folders()
    if not folders:
        print("No show folders configured. Add some via the desktop app first.")
        return
    folder_map = _build_folder_map(folders)
    print(f"Configured show folders: {len(folders)}")

    meta_by_show: dict[Path, dict[str, RSSEpisode]] = {}
    if not args.skip_meta:
        print("\n========== Phase A — refresh .episode_meta.json ==========")
        a, meta_by_show = phase_a_refresh_meta_files(
            folder_map, apply=args.apply, show_filter=args.show
        )
        print(f"\nPhase A summary: {a}")

    if not args.skip_chunks:
        print("\n========== Phase B — heal LanceDB chunk meta ==========")
        b = phase_b_heal_chunks(
            folder_map,
            apply=args.apply,
            show_filter=args.show,
            meta_by_show=meta_by_show,
        )
        print(f"\nPhase B summary: {b}")

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
