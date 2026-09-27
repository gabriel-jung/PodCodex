"""Pytest plugin: the ``legacy`` marker, for tests that expire with a migration.

Loaded for every run via ``addopts`` in ``pyproject.toml``.

Some tests pin behaviour that exists only for data or installs predating a
migration: label-keyed index rows, the pre-split ``allowed_shows`` list, flat
GPU installs. They matter while such data is out there and become dead weight
the day the migration code goes. Marking them ties each one to the code it
retires with::

    @pytest.mark.legacy("show-id")
    def test_...

``pytest --collect-only -q -m legacy`` lists them all. A name outside
``MIGRATIONS`` fails collection, so that dict stays the one place that says
what retires with what.
"""

from __future__ import annotations

import pytest

# name -> what the legacy data is, and the source that retires with it.
MIGRATIONS: dict[str, str] = {
    "show-id": "index rows, folders and guild keys from before show.toml ids; "
    "rag/show_id_migration.py, the label fallbacks in IndexStore, "
    "show_registry.mint_missing_show_ids, bot/access._normalize_show_ids",
    "allowed-shows-split": "the pre-split guild allowed_shows list (and its "
    "older default_shows name); ServerSettings.allowed_shows and the legacy "
    "halves of _unlocked_ids / _pinned_ids / _drop_unlock / _handle_setup",
    "name-keyed-passwords": "password rows keyed by display name; "
    "IndexStore._delete_legacy_password_rows and the label fallbacks in "
    "_purge_show_from_index / _relabel_password (retire with show-id)",
    "gpu-flat-layout": "GPU installs from before per-install folders; the flat "
    "branches of api/gpu_backend.py (active_install_dir, "
    "_legacy_install_interrupted, _carry_over, _prune_old_installs)",
    "narrator-rename": 'pre-0.2.10 transcripts using "Narrator"; '
    "LEGACY_NARRATOR_SPEAKER (Python and speakers.ts) and the declared "
    "clause of is_unattributed",
    "pub-date-column": "chunk tables from before the scalar pub_date column; "
    "IndexStore._ensure_pub_date_column, Hit.rss_pub_date fallback",
    "youtube-id-bridge": "episode meta from before the youtube_id field; "
    "ingest/rss._bridge_legacy_youtube_id",
    "episode-title-backfill": "chunks indexed before titles reached chunk "
    "meta; IndexStore._ensure_episode_title_backfill",
    "fts-v2": "FTS indexes from older lancedb or before accent folding; the "
    "rebuild branch of IndexStore._ensure_fts",
    "bundle-v1": "archives with manifest schema 1 (no show id); the mint "
    "branch of bundle/import_show._settle_show_identities",
    "slug-stem": "episode dirs named by bare slug; the last fallback of "
    "ingest/rss.episode_stem",
    "synth-manifest-v1": "synth manifests without per-segment model and "
    "language; the setdefault loop in core/synthesize.load_manifest",
    "pipeline-db-schema": "pipeline.db from before provenance and the "
    "polished rename; the ALTER TABLE list in core/pipeline_db.py",
    "collections-schema": "_collections tables without artwork_url / "
    "show_id; IndexStore._ensure_collections_schema",
    "bot-state-cwd": "bot state files in the working directory; the cwd move "
    "in bot/bot._resolve_state_path",
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "legacy(migration): pins behaviour that only exists for data predating "
        "<migration>; delete with that migration's code (tests/fixtures/legacy.py)",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for mark in item.iter_markers("legacy"):
            name = mark.args[0] if mark.args else None
            if name not in MIGRATIONS:
                raise pytest.UsageError(
                    f"{item.nodeid}: legacy marker names {name!r}; use one of "
                    f"{sorted(MIGRATIONS)} or add it to tests/fixtures/legacy.py"
                )
