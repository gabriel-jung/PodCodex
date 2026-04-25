"""podcodex-export — build `.podcodex` archives.

Examples::

    podcodex-export "My Show"                                 # full bundle
    podcodex-export "My Show" --with-audio                    # full + audio
    podcodex-export "My Show" --index-only                    # one show, index only
    podcodex-export "Show A" "Show B" --index-only            # selective deploy
    podcodex-export --all --index-only                        # every show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from podcodex.bundle import export_index, export_show
from podcodex.bundle.manifest import Mode, default_archive_filename
from podcodex.cli.resolve import all_registered_show_folders, resolve_show_folder


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="podcodex-export",
        description="Export PodCodex shows into a portable .podcodex archive.",
    )
    p.add_argument(
        "shows",
        nargs="*",
        help="One or more show names or folder paths.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Export every registered show. Requires --index-only.",
    )
    p.add_argument(
        "--with-audio",
        action="store_true",
        help="Include audio files in a full bundle (single-show only).",
    )
    p.add_argument(
        "--index-only",
        action="store_true",
        help="Export only LanceDB collections (no show folder content).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output archive path (default: <show>.podcodex or shows-index.podcodex).",
    )
    return p.parse_args(argv)


def _print_progress(msg: str, frac: float) -> None:
    pct = int(round(frac * 100))
    print(f"  [{pct:3d}%] {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Argument validation
    if args.all and not args.index_only:
        sys.exit("--all requires --index-only")
    if args.with_audio and args.index_only:
        sys.exit("--with-audio cannot be combined with --index-only")
    if not args.all and not args.shows:
        sys.exit("provide at least one show name/path, or --all --index-only")

    # Resolve shows → folder paths
    if args.all:
        show_folders = all_registered_show_folders()
        if not show_folders:
            sys.exit("no registered shows on disk")
    else:
        show_folders = [resolve_show_folder(s)[0] for s in args.shows]

    multi_show = len(show_folders) > 1
    if multi_show and not args.index_only:
        sys.exit("multiple shows requires --index-only (full bundles are single-show)")
    if multi_show and args.with_audio:
        sys.exit("--with-audio cannot be combined with multiple shows")

    # Default output path
    if args.output:
        output = args.output
    elif multi_show or args.all:
        output = default_archive_filename(None, Mode.INDEX_ONLY)
    else:
        mode = Mode.INDEX_ONLY if args.index_only else Mode.FULL
        output = default_archive_filename(show_folders[0].name, mode)

    # Run
    if multi_show or args.all:
        result = export_index(show_folders, output, progress=_print_progress)
    else:
        result = export_show(
            show_folders[0],
            output,
            with_audio=args.with_audio,
            index_only=args.index_only,
            progress=_print_progress,
        )

    logger.info(
        f"Exported {result.shows_exported} show(s), "
        f"{result.collections_exported} collection(s) "
        f"({result.size_bytes / 1e6:.2f} MB) → {result.output_path}"
    )


if __name__ == "__main__":
    main()
