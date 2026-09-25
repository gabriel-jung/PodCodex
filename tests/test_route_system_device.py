"""Tests for /api/system/device GET + POST routes."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcodex.api.app import app  # noqa: E402
from podcodex.core import user_settings  # noqa: E402
from tests.fixtures.api_client import client_for


@pytest.fixture
def isolated(isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("PODCODEX_DEVICE", raising=False)
    return isolated_data_dir


@pytest.fixture
def client() -> TestClient:
    return client_for(app)


def test_get_device_includes_persisted_override(
    isolated: Path, client: TestClient
) -> None:
    user_settings.set_device_override("cpu")
    resp = client.get("/api/system/device")
    assert resp.status_code == 200
    body = resp.json()
    assert body["persisted_override"] == "cpu"


def test_get_device_default_persisted_is_auto(
    isolated: Path, client: TestClient
) -> None:
    resp = client.get("/api/system/device")
    assert resp.status_code == 200
    assert resp.json()["persisted_override"] == "auto"


def test_post_device_persists_and_updates_env(
    isolated: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    resp = client.post("/api/system/device", json={"override": "cpu"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["persisted_override"] == "cpu"
    assert body["restart_required"] is True
    # Live env updated for the running process.
    assert os.environ.get("PODCODEX_DEVICE") == "cpu"
    # File persisted.
    assert user_settings.get_device_override() == "cpu"


def test_post_device_auto_clears_persisted(isolated: Path, client: TestClient) -> None:
    user_settings.set_device_override("cpu")
    resp = client.post("/api/system/device", json={"override": "auto"})
    assert resp.status_code == 200
    assert resp.json()["persisted_override"] == "auto"
    assert user_settings.get_device_override() == "auto"


def test_post_device_rejects_invalid_value(isolated: Path, client: TestClient) -> None:
    resp = client.post("/api/system/device", json={"override": "metal"})
    assert resp.status_code == 422  # Pydantic Literal validation


def test_post_device_cuda_without_gpu_returns_400(
    isolated: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    resp = client.post("/api/system/device", json={"override": "cuda"})
    assert resp.status_code == 400
    assert "CUDA" in resp.json()["detail"]
    # Persisted state is unchanged on failure.
    assert user_settings.get_device_override() == "auto"
