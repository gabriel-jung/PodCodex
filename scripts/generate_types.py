#!/usr/bin/env python3
"""Generate TypeScript interfaces from Pydantic models.

Imports all API-facing Pydantic models, converts their JSON schemas to
TypeScript interface declarations, and writes them to
``frontend/src/api/generated-types.ts``.

Usage::

    .venv/bin/python scripts/generate_types.py
    # or: make types
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# ── Collect all models ──────────────────────────────────────────────────────

# Models are imported here so the script fails fast if any import is broken.
# Duplicate class names across modules get a module prefix.

from podcodex.api.schemas import (  # noqa: E402
    CreateFromRSSRequest,
    CreateFromRSSResponse,
    CreateFromYouTubeRequest,
    CreateFromYouTubeResponse,
    EpisodeOut,
    PipelineDefaultsSchema,
    RegisterShowRequest,
    RSSEpisodeOut,
    Segment,
    ShowMeta,
    TaskResponse,
    UnifiedEpisodeOut,
)
from podcodex.api.routes.batch import BatchRequest  # noqa: E402
from podcodex.api.routes.config import AppConfig  # noqa: E402
from podcodex.api.routes.index import IndexRequest  # noqa: E402
from podcodex.api.routes._helpers import (  # noqa: E402
    ApplyManualRequest as CorrectApplyManualRequest,
    LLMRequest as CorrectRequest,
    ManualPromptsRequest as CorrectManualPromptsRequest,
)
from podcodex.api.routes.episodes import (  # noqa: E402
    EpisodeListItem,
    EpisodeMeta,
)
from podcodex.api.routes.search import (  # noqa: E402
    ExactRequest,
    RandomRequest,
    SearchRequest,
    SearchResult as SearchResultSchema,
)
from podcodex.api.routes.shows import (  # noqa: E402
    MoveShowRequest,
    ShowSummary,
)
from podcodex.api.routes.synthesize import (  # noqa: E402
    AssembleRequest,
    ExtractSelectedRequest,
    ExtractVoicesRequest,
    GenerateRequest,
)
from podcodex.api.routes.transcribe import TranscribeRequest  # noqa: E402
from podcodex.api.routes.youtube import (  # noqa: E402
    YouTubeDownloadRequest,
    YouTubeSubsRequest,
)
from podcodex.api.routes.translate import (  # noqa: E402
    ApplyManualRequest as TranslateApplyManualRequest,
    ManualPromptsRequest as TranslateManualPromptsRequest,
    TranslateRequest,
)
from podcodex.api.routes.bundle import (  # noqa: E402
    ExportIndexRequest,
    ExportShowRequest,
    ImportRequest,
    PreviewRequest,
)
from podcodex.bundle.manifest import (  # noqa: E402
    ArchivePreview,
    CollectionEntry,
    ExportResult,
    ImportResult,
    Manifest,
    ShowEntry,
)

# (name_override, model_class) — name_override=None uses the class name.
MODELS: list[tuple[str | None, type[BaseModel]]] = [
    # schemas.py
    (None, PipelineDefaultsSchema),
    (None, ShowMeta),
    (None, EpisodeOut),
    (None, RSSEpisodeOut),
    (None, Segment),
    (None, UnifiedEpisodeOut),
    (None, CreateFromRSSRequest),
    (None, RegisterShowRequest),
    (None, CreateFromRSSResponse),
    (None, CreateFromYouTubeRequest),
    (None, CreateFromYouTubeResponse),
    (None, TaskResponse),
    # routes
    (None, AppConfig),
    (None, ShowSummary),
    (None, MoveShowRequest),
    (None, TranscribeRequest),
    ("CorrectRequest", CorrectRequest),
    ("CorrectManualPromptsRequest", CorrectManualPromptsRequest),
    ("CorrectApplyManualRequest", CorrectApplyManualRequest),
    (None, EpisodeListItem),
    (None, EpisodeMeta),
    (None, TranslateRequest),
    ("TranslateManualPromptsRequest", TranslateManualPromptsRequest),
    ("TranslateApplyManualRequest", TranslateApplyManualRequest),
    (None, BatchRequest),
    (None, SearchRequest),
    ("SearchResultSchema", SearchResultSchema),
    (None, ExactRequest),
    (None, RandomRequest),
    (None, IndexRequest),
    (None, ExtractVoicesRequest),
    (None, ExtractSelectedRequest),
    (None, GenerateRequest),
    (None, AssembleRequest),
    (None, YouTubeDownloadRequest),
    (None, YouTubeSubsRequest),
    # bundle
    (None, CollectionEntry),
    (None, ShowEntry),
    (None, Manifest),
    (None, ArchivePreview),
    (None, ExportResult),
    (None, ImportResult),
    (None, ExportShowRequest),
    (None, ExportIndexRequest),
    (None, PreviewRequest),
    (None, ImportRequest),
]

# ── JSON Schema → TypeScript converter ──────────────────────────────────────

# Maps JSON Schema types to TypeScript types.
_PRIMITIVE_MAP = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
}


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref pointer like ``#/$defs/Foo`` to its schema dict."""
    name = ref.rsplit("/", 1)[-1]
    return defs.get(name, {})


def _schema_to_ts(schema: dict[str, Any], defs: dict[str, Any]) -> str:
    """Convert a JSON Schema property to a TypeScript type string."""
    if "$ref" in schema:
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        return ref_name

    # anyOf (used by Pydantic for Optional / Union types)
    if "anyOf" in schema:
        types = []
        has_null = False
        for variant in schema["anyOf"]:
            if variant.get("type") == "null":
                has_null = True
            else:
                types.append(_schema_to_ts(variant, defs))
        ts = " | ".join(types) if types else "unknown"
        if has_null:
            ts += " | null"
        return ts

    # Inline literal/enum (e.g. `Literal["a", "b"]`) — must precede the
    # primitive-type check or `type: "string"` would shadow the values.
    if "enum" in schema:
        return " | ".join(json.dumps(v) for v in schema["enum"])

    schema_type = schema.get("type")

    if schema_type in _PRIMITIVE_MAP:
        return _PRIMITIVE_MAP[schema_type]

    if schema_type == "array":
        items = schema.get("items", {})
        item_type = _schema_to_ts(items, defs)
        return f"{item_type}[]"

    if schema_type == "object":
        additional = schema.get("additionalProperties")
        if additional and isinstance(additional, dict):
            val_type = _schema_to_ts(additional, defs)
            return f"Record<string, {val_type}>"
        return "Record<string, unknown>"

    # const
    if "const" in schema:
        return json.dumps(schema["const"])

    return "unknown"


def _allows_null(schema: dict[str, Any]) -> bool:
    """True if this JSON schema permits a null value."""
    if schema.get("type") == "null":
        return True
    for variant in schema.get("anyOf", []):
        if variant.get("type") == "null":
            return True
    return False


def _model_to_ts(name: str, model: type[BaseModel]) -> str:
    """Convert a Pydantic model to a TypeScript interface string.

    Response models (anything not ending in ``Request``) emit a field as
    required when it has a non-null default: FastAPI always serializes
    such fields, so downstream TS should not have to narrow them.

    Request models keep defaulted fields optional — callers are allowed
    to omit them and let the server fill the default.
    """
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    is_input = name.endswith("Request")

    lines = [f"export interface {name} {{"]
    for prop_name, prop_schema in properties.items():
        ts_type = _schema_to_ts(prop_schema, defs)
        promote = (
            not is_input and "default" in prop_schema and not _allows_null(prop_schema)
        )
        is_required = prop_name in required or promote
        optional = "" if is_required else "?"
        lines.append(f"  {prop_name}{optional}: {ts_type};")
    lines.append("}")
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "frontend"
    / "src"
    / "api"
    / "generated-types.ts"
)

HEADER = """\
// AUTO-GENERATED — do not edit manually.
// Regenerate with: make types  (or: .venv/bin/python scripts/generate_types.py)
//
// Source: Pydantic models in src/podcodex/api/

"""


def _collect_enum_defs(model: type[BaseModel]) -> dict[str, list[str]]:
    """Return ``{enum_name: [values]}`` for string-enum ``$defs`` on the model."""
    schema = model.model_json_schema()
    out: dict[str, list[str]] = {}
    for name, sub in schema.get("$defs", {}).items():
        if sub.get("type") == "string" and isinstance(sub.get("enum"), list):
            out[name] = list(sub["enum"])
    return out


def main() -> None:
    """Generate TypeScript interfaces and write to the output file."""
    blocks: list[str] = [HEADER]
    seen_names: set[str] = set()

    # Pass 1: gather string enums referenced via $ref so they get a type alias.
    enums: dict[str, list[str]] = {}
    for _name_override, model in MODELS:
        for ename, values in _collect_enum_defs(model).items():
            if ename not in enums:
                enums[ename] = values
    for ename in sorted(enums):
        if ename in seen_names:
            continue
        seen_names.add(ename)
        union = " | ".join(json.dumps(v) for v in enums[ename])
        blocks.append(f"export type {ename} = {union};")

    for name_override, model in MODELS:
        name = name_override or model.__name__
        if name in seen_names:
            print(f"WARNING: duplicate name {name}, skipping", file=sys.stderr)
            continue
        seen_names.add(name)
        blocks.append(_model_to_ts(name, model))

    content = "\n\n".join(blocks) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Generated {len(seen_names)} interfaces → {OUTPUT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
