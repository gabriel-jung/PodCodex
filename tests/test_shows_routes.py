"""Show routes: register, list, meta round-trip, rename, and the shows
list the home page reads."""

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tests.fixtures.api_client import make_client

    return make_client(tmp_path, monkeypatch)


def test_register_and_list_shows(client, tmp_path):
    show_dir = tmp_path / "myshow"
    show_dir.mkdir()

    r = client.post("/api/shows/register", json={"path": str(show_dir)})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/api/shows/")
    assert r.status_code == 200
    shows = r.json()
    assert len(shows) == 1
    assert shows[0]["path"] == str(show_dir.resolve())
    assert shows[0]["name"] == "myshow"


def test_register_rejects_missing_folder(client, tmp_path):
    missing = tmp_path / "does_not_exist"
    r = client.post("/api/shows/register", json={"path": str(missing)})
    assert r.status_code == 400


def test_show_meta_round_trip(client, tmp_path):
    show_dir = tmp_path / "show"
    show_dir.mkdir()
    client.post("/api/shows/register", json={"path": str(show_dir)})

    # Default meta for a new show: name derived from folder.
    r = client.get(f"/api/shows/{show_dir}/meta")
    assert r.status_code == 200
    default = r.json()
    assert default["name"] == "show"

    # Update and read back.
    updated = {
        "name": "My Podcast",
        "rss_url": "https://example.com/rss",
        "youtube_url": "",
        "language": "English",
        "speakers": [],
        "artwork_url": "https://example.com/art.jpg",
        "broadcast_number_pattern": r"\((\d+)\)",
        "pipeline": {
            "model_size": "large-v3",
            "diarize": True,
            "llm_mode": "ollama",
            "llm_provider": "",
            "llm_models_by_mode": {"ollama": "qwen3:4b"},
            "target_lang": "",
        },
    }
    r = client.put(f"/api/shows/{show_dir}/meta", json=updated)
    assert r.status_code == 200

    r = client.get(f"/api/shows/{show_dir}/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "My Podcast"
    assert body["rss_url"] == "https://example.com/rss"
    assert body["broadcast_number_pattern"] == r"\((\d+)\)"
    assert body["pipeline"]["model_size"] == "large-v3"
    assert body["pipeline"]["llm_models_by_mode"] == {"ollama": "qwen3:4b"}


def test_broadcast_preview(client, tmp_path):
    from podcodex.ingest.rss import RSSEpisode, save_feed_cache

    show_dir = tmp_path / "show"
    show_dir.mkdir()
    client.post("/api/shows/register", json={"path": str(show_dir)})
    save_feed_cache(
        show_dir,
        [
            RSSEpisode(guid="b", title="(252) John Powell", pub_date="", feed_order=0),
            RSSEpisode(guid="a", title="(251) Older", pub_date="", feed_order=1),
        ],
    )

    # Newest episode title drives the preview; pattern extracts its number.
    r = client.get(
        f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": r"\((\d+)\)"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "(252) John Powell"
    assert body["number"] == 252
    assert body["error"] is None

    # Invalid regex surfaces an error instead of raising.
    r = client.get(f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": "("})
    assert r.status_code == 200
    assert r.json()["error"]
    assert r.json()["number"] is None

    # Empty pattern: title still returned, no number, no error.
    r = client.get(f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": ""})
    assert r.json()["number"] is None
    assert r.json()["error"] is None

    # Valid pattern with no capture group: explicit error, not a false
    # "no match" (the regex matched; it just captures nothing).
    r = client.get(
        f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": r"\d+"}
    )
    assert "capture group" in (r.json()["error"] or "")


def test_broadcast_preview_invalid_pattern_without_title(client, tmp_path):
    """An invalid regex must be surfaced even when the show has no titled
    episode (no feed cache): otherwise it autosaves silently."""
    show_dir = tmp_path / "show"
    show_dir.mkdir()
    client.post("/api/shows/register", json={"path": str(show_dir)})
    r = client.get(f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": "("})
    assert r.status_code == 200
    body = r.json()
    assert body["title"] is None
    assert body["error"]


def test_get_meta_missing_show_returns_404(client):
    r = client.get("/api/shows/nonexistent/meta")
    assert r.status_code == 404


def _rename_with_password(client, tmp_path, write_password):
    """Register "My Show", let *write_password* store its password, rename it
    to "Renamed"; return ``(sid, password entries)``."""
    from podcodex.ingest.show import show_id
    from podcodex.rag.index_store import get_index_store
    from tests.fixtures.api_client import registered_show

    folder = registered_show(client, tmp_path / "My Show", "My Show")
    sid = show_id(folder)
    store = get_index_store()
    write_password(store, sid)

    r = client.put(
        f"/api/shows/{folder}/meta",
        json={"name": "Renamed", "speakers": [], "pipeline": {}},
    )

    assert r.status_code == 200, r.text
    assert show_id(folder) == sid
    return sid, store.get_show_password_entries()


def test_rename_keeps_the_show_id_and_carries_the_password(client, tmp_path):
    """The label changes; identity, which every store keys on, does not, and
    the password row takes the new label (the bot shows it)."""
    sid, entries = _rename_with_password(
        client,
        tmp_path,
        lambda store, sid: store.set_show_password(sid, "h", show_label="My Show"),
    )
    assert list(entries) == [sid]
    assert entries[sid]["label"] == "Renamed"


@pytest.mark.legacy("name-keyed-passwords")
def test_rename_rekeys_a_name_keyed_password(client, tmp_path):
    """A row written before ids existed is keyed by the old name; the rename
    moves it onto the id or the bot keeps enforcing the old name."""
    row = {"show_id": "", "show_label": "", "show": "My Show", "password_hash": "h"}
    sid, entries = _rename_with_password(
        client, tmp_path, lambda store, _sid: store._passwords_table().add([row])
    )
    assert list(entries) == [sid]
    assert entries[sid]["label"] == "Renamed"


def test_rename_to_a_taken_name_is_rejected(client, tmp_path):
    from tests.fixtures.api_client import registered_show

    registered_show(client, tmp_path / "A", "Alpha")
    b = registered_show(client, tmp_path / "B", "Beta")

    r = client.put(
        f"/api/shows/{b}/meta", json={"name": "Alpha", "speakers": [], "pipeline": {}}
    )

    assert r.status_code == 409, r.text

    # A pre-existing collision must not block unrelated edits: B is already
    # called Alpha here, so saving it again (with a language change) is fine.
    from podcodex.ingest.show import ShowMeta, load_show_meta, save_show_meta

    save_show_meta(b, ShowMeta(name="Alpha"))
    r = client.put(
        f"/api/shows/{b}/meta",
        json={"name": "Alpha", "language": "en", "speakers": [], "pipeline": {}},
    )
    assert r.status_code == 200, r.text
    assert load_show_meta(b).language == "en"


def test_shows_list_counts_go_through_the_reconcile(tmp_path, monkeypatch):
    """The home card must not report a flag the show page would demote."""
    from podcodex.core.app_config import AppConfig
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db
    from tests.fixtures.api_client import make_client

    show = tmp_path / "show"
    (show / "ep1").mkdir(parents=True)
    client = make_client(
        tmp_path, monkeypatch, config=AppConfig(show_folders=[str(show)])
    )
    get_pipeline_db(show).mark("ep1", transcribed=True)  # no version file on disk
    try:
        (card,) = client.get("/api/shows/").json()
        assert card["transcribed_count"] == 0
    finally:
        close_pipeline_db(show)


def test_shows_list_does_not_open_the_index_and_survives_a_failed_reconcile(
    tmp_path, monkeypatch
):
    import podcodex.core.episode_status as status_mod
    from podcodex.api.routes import shows as shows_routes
    from podcodex.core.app_config import AppConfig
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db
    from tests.fixtures.api_client import make_client

    show = tmp_path / "show"
    (show / "ep1" / "transcript").mkdir(parents=True)
    (show / "ep1" / "transcript" / "20260101T000000000000Z_raw.json").write_text("[]")
    client = make_client(
        tmp_path, monkeypatch, config=AppConfig(show_folders=[str(show)])
    )
    get_pipeline_db(show).mark("ep1", transcribed=True)
    monkeypatch.setattr(
        status_mod, "lance_indexed_stems", lambda _p: pytest.fail("opened the index")
    )
    try:
        assert client.get("/api/shows/").json()[0]["transcribed_count"] == 1

        def boom(*_a, **_k):
            raise OSError("share unmounted")

        monkeypatch.setattr(shows_routes, "reconcile_show_status", boom)
        assert client.get("/api/shows/").json()[0]["transcribed_count"] == 1
    finally:
        close_pipeline_db(show)


# ── Destructive show routes refuse an unregistered folder ───────────────


def test_delete_unregistered_show_forbidden(client, tmp_path):
    """delete_files runs rmtree; an unregistered directory must be refused."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()
    (victim / "keep.txt").write_text("important")

    r = client.post(f"/api/shows/{victim}/delete", json={"delete_files": True})
    assert r.status_code == 403
    assert victim.exists()  # nothing deleted


def test_delete_registered_show_allowed(client, tmp_path):
    """A registered show still deletes normally."""
    show = tmp_path / "myshow"
    show.mkdir()
    client.post("/api/shows/register", json={"path": str(show)})

    r = client.post(f"/api/shows/{show}/delete", json={"delete_files": True})
    assert r.status_code == 200
    assert not show.exists()


def test_update_meta_unregistered_show_forbidden(client, tmp_path):
    """update_show_meta writes show.toml; an unregistered dir must be refused."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()

    meta = {
        "name": "Injected",
        "rss_url": "",
        "youtube_url": "",
        "language": "en",
        "speakers": [],
        "artwork_url": "",
        "broadcast_number_pattern": "",
        "pipeline": {},
    }
    r = client.put(f"/api/shows/{victim}/meta", json=meta)
    assert r.status_code == 403
    assert not (victim / "show.toml").exists()


def test_move_unregistered_show_forbidden(client, tmp_path):
    """move runs shutil.move/rmtree on the source; refuse an unregistered dir."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()
    dest = tmp_path / "dest"

    r = client.post(f"/api/shows/{victim}/move", json={"new_path": str(dest)})
    assert r.status_code == 403
    assert victim.exists()
