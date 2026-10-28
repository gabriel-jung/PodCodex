"""Tests for the bundle CLIs (podcodex-export, podcodex-import) and their helpers."""

from __future__ import annotations

import pytest

import podcodex.core.app_config as app_config
from podcodex.cli.resolve import expected_errors, resolve_show_folder
from podcodex.ingest.show import ShowMeta, save_show_meta
from podcodex.ingest.show_registry import register_folder


@pytest.fixture(autouse=True)
def _own_config(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.json")


def test_a_registered_name_beats_a_same_named_directory(tmp_path, monkeypatch):
    """Path-first resolution let any folder in the cwd shadow a show and get a
    show.toml written into it."""
    show = tmp_path / "shows" / "real"
    save_show_meta(show, ShowMeta(name="Talk"))
    register_folder(show)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Talk").mkdir()

    folder, name = resolve_show_folder("Talk")
    assert folder == show.resolve() and name == "Talk"
    assert not (tmp_path / "Talk" / "show.toml").exists()


def test_a_plain_directory_is_not_a_show(tmp_path):
    (tmp_path / "random").mkdir()
    with pytest.raises(SystemExit) as exc:
        resolve_show_folder(str(tmp_path / "random"))
    assert "Not a show folder" in str(exc.value.code)


def test_a_show_folder_path_resolves(tmp_path):
    show = tmp_path / "s"
    save_show_meta(show, ShowMeta(name="By Path"))
    assert resolve_show_folder(str(show)) == (show.resolve(), "By Path")


def test_expected_failures_exit_with_one_line():
    from podcodex.rag.index_origin import IndexOwnershipError

    with pytest.raises(SystemExit) as exc, expected_errors():
        raise IndexOwnershipError("refused")
    assert str(exc.value.code).startswith("error: refused")


def test_import_cli_reports_a_wrong_file_without_a_traceback(tmp_path, monkeypatch):
    import podcodex.bootstrap as bootstrap
    from podcodex.cli.import_show import main

    # The real bootstrap loads the developer's .env; not wanted in a test.
    monkeypatch.setattr(bootstrap, "bootstrap_for_cli", lambda: None)

    bad = tmp_path / "x.podcodex"
    bad.write_text("not an archive")
    with pytest.raises(SystemExit) as exc:
        main([str(bad)])
    assert str(exc.value.code).startswith("error: ")
