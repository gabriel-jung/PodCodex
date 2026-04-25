"""podcodex-import — restore `.podcodex` archives.

Examples::

    podcodex-import bundle.podcodex                           # full bundle
    podcodex-import bundle.podcodex --shows-dir ~/Podcasts    # explicit target
    podcodex-import bundle.podcodex --on-conflict replace     # bot deploy
    podcodex-import bundle.podcodex --name "My Renamed Show"  # rename on import
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from podcodex.bundle import (
    Mode,
    import_archive,
    preview_archive,
)
from podcodex.bundle.conflicts import resolve_policy
from podcodex.cli.resolve import default_shows_dir


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="podcodex-import",
        description="Restore a .podcodex archive into the local PodCodex install.",
    )
    p.add_argument("archive", type=Path, help="Path to a .podcodex file.")
    p.add_argument(
        "--shows-dir",
        type=Path,
        help="Where to write show folder content for full bundles "
        "(default: parent of first registered show).",
    )
    p.add_argument(
        "--name",
        help="Override the imported folder name. Single-show bundles only.",
    )
    p.add_argument(
        "--on-conflict",
        choices=["auto", "rename", "replace", "abort"],
        default="auto",
        help="Resolution for folder/collection collisions. "
        "'auto' = rename for full bundles, replace for index-only.",
    )
    return p.parse_args(argv)


def _print_progress(msg: str, frac: float) -> None:
    if frac < 0:
        print(f"  [...] {msg}", file=sys.stderr)
    else:
        print(f"  [{int(round(frac * 100)):3d}%] {msg}", file=sys.stderr)


def _print_preview_summary(preview) -> None:
    print(f"Archive: {preview.archive_path}", file=sys.stderr)
    print(f"  mode: {preview.manifest.mode}", file=sys.stderr)
    print(f"  size: {preview.size_bytes / 1e6:.2f} MB", file=sys.stderr)
    for show in preview.manifest.shows:
        col_summary = ", ".join(c.name for c in show.collections)
        audio = " +audio" if show.audio_included else ""
        print(
            f"  show: {show.name} (folder={show.folder!r}{audio}, "
            f"collections=[{col_summary}])",
            file=sys.stderr,
        )
    for w in preview.embedder_warnings:
        print(f"  WARNING: {w}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    preview = preview_archive(args.archive)
    _print_preview_summary(preview)

    policy = resolve_policy(args.on_conflict, preview.manifest.mode)

    shows_dir: Path | None = None
    if preview.manifest.mode == Mode.FULL:
        if args.shows_dir:
            shows_dir = Path(args.shows_dir).expanduser().resolve()
        else:
            shows_dir = default_shows_dir()
            if shows_dir is None:
                sys.exit("--shows-dir required (no registered shows to derive default)")
            print(f"  shows-dir (default): {shows_dir}", file=sys.stderr)

    result = import_archive(
        args.archive,
        shows_dir=shows_dir,
        name=args.name,
        on_conflict=policy,
        progress=_print_progress,
    )

    logger.info(
        f"Imported {len(result.shows_imported)} show(s), "
        f"{len(result.collections_imported)} collection(s)"
    )
    for k, v in result.conflicts_resolved.items():
        logger.info(f"  conflict {k} → {v}")


if __name__ == "__main__":
    main()
