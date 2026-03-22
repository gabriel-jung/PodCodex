"""Tests for podcodex.ingest.show — show.toml read/write."""

from podcodex.ingest.show import ShowMeta, load_show_meta, save_show_meta


def test_save_and_load_roundtrip(tmp_path):
    meta = ShowMeta(
        name="Mi Podcast",
        rss_url="https://example.com/feed.xml",
        speakers=["Alice", "Bob"],
        language="es",
    )
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)

    assert loaded is not None
    assert loaded.name == "Mi Podcast"
    assert loaded.rss_url == "https://example.com/feed.xml"
    assert loaded.speakers == ["Alice", "Bob"]
    assert loaded.language == "es"


def test_save_minimal(tmp_path):
    meta = ShowMeta(name="Simple Show")
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)

    assert loaded is not None
    assert loaded.name == "Simple Show"
    assert loaded.rss_url == ""
    assert loaded.speakers == []
    assert loaded.language == ""


def test_load_missing_returns_none(tmp_path):
    assert load_show_meta(tmp_path) is None


def test_save_creates_directory(tmp_path):
    folder = tmp_path / "new_show"
    meta = ShowMeta(name="New Show")
    path = save_show_meta(folder, meta)

    assert path.exists()
    assert folder.is_dir()


def test_load_partial_fields(tmp_path):
    """show.toml with only name — other fields should default."""
    (tmp_path / "show.toml").write_text('name = "Only Name"\n')
    loaded = load_show_meta(tmp_path)

    assert loaded is not None
    assert loaded.name == "Only Name"
    assert loaded.rss_url == ""
    assert loaded.speakers == []


def test_save_and_load_special_characters(tmp_path):
    """Show names with quotes and backslashes survive roundtrip."""
    meta = ShowMeta(name='Show "with" quotes', speakers=["O'Brien", "Back\\slash"])
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)

    assert loaded is not None
    assert loaded.name == 'Show "with" quotes'
    assert loaded.speakers == ["O'Brien", "Back\\slash"]
