"""WebSocket progress: what every progress bar and result panel reads."""

from __future__ import annotations

import threading

import pytest

from tests.fixtures.api_client import make_client

WS = "/api/ws?token=test-token"
HOST = {"host": "127.0.0.1:18811"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Entered, so the lifespan binds the broadcast loop.
    with make_client(tmp_path, monkeypatch) as c:
        yield c


def _until(ws, task_id, status):
    seen = []
    while True:
        msg = ws.receive_json()
        if msg.get("task_id") != task_id:
            continue
        seen.append(msg)
        if msg["status"] == status:
            return seen


def test_a_task_streams_progress_then_its_result(client):
    from podcodex.api.tasks import task_manager

    go = threading.Event()

    def work(progress_cb):
        go.wait(5.0)
        progress_cb(0.5, "Halfway there")
        return {"count": 3}

    with client.websocket_connect(WS, headers=HOST) as ws:
        info = task_manager.submit("transcribe", "/ws/ep.mp3", work)
        go.set()
        seen = _until(ws, info.task_id, "completed")

    # A broadcast sends the task's state when it runs, so an intermediate
    # message can already carry a later state; the milestone survives in steps.
    assert "Halfway there" in seen[-1]["steps"]
    assert seen[-1]["result"] == {"count": 3}
    assert seen[-1]["progress"] == 1.0


def test_a_client_connecting_mid_task_gets_its_state(client):
    from podcodex.api.tasks import task_manager

    started, release = threading.Event(), threading.Event()

    def work(progress_cb):
        progress_cb(0.2, "Loading model")
        started.set()
        release.wait(5.0)

    info = task_manager.submit("transcribe", "/ws/ep2.mp3", work)
    try:
        assert started.wait(5.0)
        with client.websocket_connect(WS, headers=HOST) as ws:
            replay = ws.receive_json()
        assert replay["task_id"] == info.task_id
        assert replay["status"] == "running"
        assert "Loading model" in replay["steps"]
    finally:
        release.set()
