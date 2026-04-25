"""podcodex.bundle — `.podcodex` archive format for sharing shows + selective bot deploy.

Pure functions, no argparse / sys.exit / interactive prompts. CLI and API
endpoints wrap these as thin adapters.
"""

from podcodex.bundle.conflicts import ConflictError, ConflictPolicy
from podcodex.bundle.export import export_index, export_show
from podcodex.bundle.import_show import import_archive, preview_archive
from podcodex.bundle.manifest import (
    SCHEMA_VERSION,
    ArchiveCorruptError,
    ArchivePreview,
    CollectionEntry,
    ExportResult,
    ImportResult,
    Manifest,
    ManifestVersionError,
    Mode,
    ShowEntry,
)

__all__ = [
    "SCHEMA_VERSION",
    "ArchiveCorruptError",
    "ArchivePreview",
    "CollectionEntry",
    "ConflictError",
    "ConflictPolicy",
    "ExportResult",
    "ImportResult",
    "Manifest",
    "ManifestVersionError",
    "Mode",
    "ShowEntry",
    "export_index",
    "export_show",
    "import_archive",
    "preview_archive",
]
