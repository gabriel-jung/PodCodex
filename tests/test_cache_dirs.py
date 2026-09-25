"""Tests for podcodex.core.cache directory resolution.

The model cache must sit under the one definition of the data dir
(``app_paths.data_dir()``). A second definition here is what used to send the
bot's HF cache to ``~/.podcodex/models`` inside the container, a path on no
compose volume, so BGE-M3 was re-downloaded on every recreate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcodex.core import app_paths, cache


def test_cache_dir_lives_under_the_data_dir(isolated_data_dir: Path) -> None:
    assert cache.get_cache_dir() == isolated_data_dir / "models"


def test_cache_dir_follows_app_paths_without_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No PODCODEX_DATA_DIR (the bot, MCP, dev): still the platform data dir."""
    monkeypatch.delenv("PODCODEX_CACHE_DIR", raising=False)
    monkeypatch.delenv("PODCODEX_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr("sys.platform", "linux")
    app_paths.data_dir.cache_clear()
    try:
        assert cache.get_cache_dir() == app_paths.data_dir() / "models"
    finally:
        app_paths.data_dir.cache_clear()


def test_explicit_override_still_wins(
    isolated_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("PODCODEX_CACHE_DIR", str(override))
    assert cache.get_cache_dir() == override


def test_hf_cache_getters_are_pure_and_agree(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wire_model_caches is the single env setter; the getters only name
    the dirs, and the hub dir is what loaders pass explicitly."""
    import os

    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(var, raising=False)

    hf_dir = cache.get_hf_cache_dir()
    assert hf_dir == isolated_data_dir / "models" / "huggingface"
    assert cache.get_hf_hub_dir() == hf_dir / "hub"
    assert "HF_HOME" not in os.environ


HF_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")


def _hf_env(monkeypatch: pytest.MonkeyPatch, setter) -> dict[str, str]:
    """Run one cache setter in an environment with no HF vars, return them."""
    import os

    for var in HF_VARS:
        monkeypatch.delenv(var, raising=False)
    setter()
    return {var: os.environ.get(var, "") for var in HF_VARS}


ML_VARS = (*HF_VARS, "TORCH_HOME", "SENTENCE_TRANSFORMERS_HOME")


def test_wire_model_caches_sets_the_whole_layout(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One setter for the layout: HF vars agree with ``get_hf_cache_dir`` and
    torch / sentence-transformers sit next to them under the model cache."""
    import os

    for var in ML_VARS:
        monkeypatch.delenv(var, raising=False)
    cache.wire_model_caches()

    models = isolated_data_dir / "models"
    assert os.environ["HF_HOME"] == str(models / "huggingface")
    assert os.environ["HF_HUB_CACHE"] == str(models / "huggingface" / "hub")
    assert os.environ["TRANSFORMERS_CACHE"] == os.environ["HF_HUB_CACHE"]
    assert os.environ["TORCH_HOME"] == str(models / "torch")
    assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == str(
        models / "sentence-transformers"
    )


def test_wire_model_caches_follows_the_cache_override(
    isolated_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PODCODEX_CACHE_DIR moves every cache, not only the HF half."""
    import os

    for var in ML_VARS:
        monkeypatch.delenv(var, raising=False)
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("PODCODEX_CACHE_DIR", str(override))
    cache.wire_model_caches()
    for var in ML_VARS:
        assert os.environ[var].startswith(str(override)), var


def test_wire_model_caches_keeps_preset_values(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Tauri shell presets the hub and torch vars; they win."""
    import os

    for var in ML_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", "/preset/hub")
    cache.wire_model_caches()
    assert os.environ["HF_HUB_CACHE"] == "/preset/hub"


@pytest.mark.parametrize(
    "entry",
    [
        "bootstrap_for_bundled_sidecar",
        "bootstrap_for_mcp_stdio",
        "bootstrap_for_dev",
        "bootstrap_for_subprocess_child",
    ],
)
def test_every_bootstrap_wires_caches_before_anything_else(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """huggingface_hub reads the cache vars once, at import, and the eager
    patches import transformers. Wiring after any other bootstrap step sends
    pyannote and Qwen3-TTS weights to ~/.cache/huggingface."""
    from podcodex import bootstrap

    calls: list[str] = []
    for name in dir(bootstrap):
        if name.startswith("_") and callable(getattr(bootstrap, name)):
            if name in {"_get_deferred_finder"}:
                continue
            monkeypatch.setattr(
                bootstrap, name, lambda *a, _n=name, **k: calls.append(_n)
            )
    getattr(bootstrap, entry)()
    assert calls and calls[0] == "_wire_model_caches", calls


def test_delete_cached_model_removes_a_hub_layout_model(
    isolated_data_dir: Path,
) -> None:
    hub = cache.get_hf_cache_dir() / "hub"
    target = hub / "models--BAAI--bge-m3"
    (target / "blobs").mkdir(parents=True)
    (target / "blobs" / "weights.bin").write_bytes(b"x" * 16)

    assert cache.delete_cached_model("models--BAAI--bge-m3") is True
    assert not target.exists()
    assert hub.is_dir()


def test_delete_cached_model_removes_a_flat_layout_model(
    isolated_data_dir: Path,
) -> None:
    """faster-whisper passes ``cache_dir=`` and lands one level higher."""
    hf_root = cache.get_hf_cache_dir()
    target = hf_root / "models--Systran--faster-whisper-large-v3"
    target.mkdir(parents=True)
    (target / "model.bin").write_bytes(b"y")

    assert cache.delete_cached_model("models--Systran--faster-whisper-large-v3") is True
    assert not target.exists()


def test_delete_cached_model_ignores_an_unknown_id(isolated_data_dir: Path) -> None:
    assert cache.delete_cached_model("models--nobody--nothing") is False


def test_delete_cached_model_refuses_a_traversing_id(isolated_data_dir: Path) -> None:
    """The id reaches here from a request body, and the call is an rmtree."""
    hf_root = cache.get_hf_cache_dir()
    outside = hf_root.parent / "sentence-transformers"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "keep.txt").write_text("keep", encoding="utf-8")

    assert cache.delete_cached_model("../sentence-transformers") is False
    assert cache.delete_cached_model("hub/../../sentence-transformers") is False
    assert (outside / "keep.txt").exists()


@pytest.mark.parametrize("model_id", ["..", ".", "hub", "../x", "", "models--a/../.."])
def test_delete_cached_model_refuses_anything_but_one_model_dir(
    isolated_data_dir: Path, model_id: str
) -> None:
    """``hub / ".."`` passed the old parent check and would have removed the
    whole HuggingFace cache."""
    hf_root = cache.get_hf_cache_dir()
    (hf_root / "hub" / "models--x--y").mkdir(parents=True)

    assert cache.delete_cached_model(model_id) is False
    assert (hf_root / "hub" / "models--x--y").is_dir()
