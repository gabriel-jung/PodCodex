"""Run heavy pipeline work in a subprocess so the FastAPI event loop stays free.

The uvicorn server is a single Python process. Steps that hold the GIL for long
stretches (embedding encode, LanceDB writes, Whisper transcribe, LLM streaming)
freeze the event loop, which hangs *every* other request including the
progress WebSocket. Running them in a spawned child isolates the GIL.

A caller passes a dotted entry path like ``"podcodex.rag.index_job:run"``
plus picklable ``kwargs``. The runner launches a fresh Python process,
forwards progress/log events over a Queue, watches a cancel Event, and
returns the child's result (or re-raises its exception). Blocking: runs in
the task's worker thread — we explicitly want the thread to wait on IPC
rather than run Python code, since that is what frees the server's GIL.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

from loguru import logger

# `spawn` so torch / LanceDB / sentence-transformers start from a clean
# interpreter; fork on macOS is unsafe with those libs.
_CTX = mp.get_context("spawn")

# Sentinel task timeout so a runaway child does not wedge the thread forever.
# Most steps finish in minutes; 4 hours is a very loose ceiling.
_HARD_TIMEOUT_SEC = 4 * 60 * 60


def _early_child_log(message: str) -> None:
    """Append a line to the persistent server.log without going through
    loguru / podcodex.

    On Windows --noconsole frozen builds, the spawned child has no usable
    stdio. If anything dies before podcodex/__init__.py runs in the child
    (DLL load failure during multiprocessing.spawn bootstrap, missing
    msvcrt, etc.), no traceback ever reaches the parent's IPC queue. This
    raw-write fallback lets us see at least *that* the child entered
    Python code and how far it got.
    """
    try:
        from pathlib import Path

        data_dir = os.environ.get("PODCODEX_DATA_DIR")
        if not data_dir:
            return
        log_dir = Path(data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "server.log", "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts} | CHILD    | [pid {os.getpid()}] {message}\n")
    except Exception:
        pass  # best-effort; never raise


def _install_log_forwarder(prog_q: Any) -> None:
    """Forward this child's loguru output back to the parent via prog_q.

    The child runs in a separate process; loguru sinks the parent registers
    don't see anything emitted here, so the in-app per-task log expander
    only ever showed progress-callback messages — none of the rich
    transcribe/diarize/correct output. Pushing each line onto prog_q with
    a "log" tag lets the parent attach those lines to ``info.log`` and
    broadcast them over the task WebSocket.

    Drops on a full queue rather than blocking: the parent's progress
    poller drains prog_q every 250ms, so steady-state pressure is low,
    but a model-load burst could overflow the 512-slot bound. Losing a
    handful of UI log lines is preferable to stalling the child.
    """
    try:
        from loguru import logger as _logger

        def sink(message: Any) -> None:
            try:
                prog_q.put_nowait(("log", str(message).rstrip()))
            except Exception:
                pass

        _logger.add(sink, level="INFO", format="{name}: {message}")
    except Exception:
        pass


def _child_entry(
    entry_path: str,
    kwargs: dict[str, Any],
    prog_q: Any,
    result_q: Any,
    cancel_ev: Any,
) -> None:
    """Import and invoke the entry function inside the spawned child."""
    _early_child_log(f"_child_entry start, entry={entry_path}")

    # Spawn re-execs the frozen binary and bypasses server.py:main(), so the
    # parent's bootstrap (transformers doc patch, HF symlink, Windows console)
    # never runs in the child. Apply patches here before any heavy import.
    try:
        from podcodex.bootstrap import bootstrap_for_subprocess_child

        _early_child_log("running bootstrap_for_subprocess_child")
        bootstrap_for_subprocess_child()
        _early_child_log("bootstrap_for_subprocess_child returned ok")
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb

        _early_child_log(
            f"bootstrap_for_subprocess_child failed: {exc!r}\n{_tb.format_exc()}"
        )

    import importlib

    mod_name, fn_name = entry_path.split(":")
    try:
        _early_child_log(f"importing {mod_name}")
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        _early_child_log(f"resolved {fn_name} on {mod_name}")
    except Exception as exc:
        tb = traceback.format_exc()
        _early_child_log(f"import failed: {exc!r}\n{tb}")
        result_q.put(("err", f"{type(exc).__name__}: {exc}", tb))
        return

    # Install the log forwarder *after* importing the entry module so
    # podcodex/__init__.py has finished registering its own sinks first;
    # adding ours mid-init could race that setup.
    _install_log_forwarder(prog_q)

    def progress_cb(frac: float, message: str) -> None:
        try:
            prog_q.put(("progress", float(frac), str(message)))
        except Exception:
            pass

    def cancelled() -> bool:
        return bool(cancel_ev.is_set())

    try:
        _early_child_log("invoking entry function")
        result = fn(progress_cb=progress_cb, cancelled=cancelled, **kwargs)
        _early_child_log("entry function returned ok")
        result_q.put(("ok", result))
    except BaseException as exc:
        tb = traceback.format_exc()
        _early_child_log(f"entry function raised: {exc!r}\n{tb}")
        result_q.put(("err", f"{type(exc).__name__}: {exc}", tb))


def run_in_subprocess(
    entry_path: str,
    kwargs: dict[str, Any],
    on_progress: Callable[[float, str], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Any:
    """Run ``entry_path(progress_cb, cancelled, **kwargs)`` in a spawned child.

    Blocks the caller thread until the child exits or cancellation is
    honored. Progress messages from the child are forwarded to
    ``on_progress``; loguru log lines emitted by the child are forwarded
    to ``on_log`` (see ``_install_log_forwarder``). A set ``cancel_event``
    is relayed to the child; if the child does not exit within 10 s of
    the signal it is terminated.
    """
    # Cap progress queue so a chatty child cannot grow RSS unboundedly if
    # the parent stalls; the child's progress_cb already swallows Full.
    prog_q = _CTX.Queue(maxsize=512)
    result_q = _CTX.Queue()
    cancel_ev = _CTX.Event()

    proc = _CTX.Process(
        target=_child_entry,
        args=(entry_path, kwargs, prog_q, result_q, cancel_ev),
        daemon=False,
    )
    proc.start()
    logger.debug("subprocess_runner: started pid={} for {}", proc.pid, entry_path)

    poll_ms = 250
    deadline = time.monotonic() + _HARD_TIMEOUT_SEC
    try:
        while True:
            if (
                cancel_event is not None
                and cancel_event.is_set()
                and not cancel_ev.is_set()
            ):
                cancel_ev.set()
                logger.info("subprocess_runner: cancel requested for pid={}", proc.pid)

            if time.monotonic() > deadline:
                logger.error("subprocess_runner: hard timeout pid={}", proc.pid)
                raise TimeoutError("Subprocess exceeded hard timeout")

            try:
                msg = prog_q.get(timeout=poll_ms / 1000.0)
            except _queue.Empty:
                if not proc.is_alive():
                    break
                continue

            if not msg:
                continue
            kind = msg[0]
            if kind == "progress" and on_progress is not None:
                try:
                    on_progress(msg[1], msg[2])
                except Exception:
                    logger.opt(exception=True).warning("on_progress raised")
            elif kind == "log" and on_log is not None:
                try:
                    on_log(msg[1])
                except Exception:
                    pass

        # Drain any progress / log events the child enqueued just before exit.
        while True:
            try:
                msg = prog_q.get_nowait()
            except _queue.Empty:
                break
            if not msg:
                continue
            kind = msg[0]
            if kind == "progress" and on_progress is not None:
                try:
                    on_progress(msg[1], msg[2])
                except Exception:
                    pass
            elif kind == "log" and on_log is not None:
                try:
                    on_log(msg[1])
                except Exception:
                    pass

        try:
            outcome = result_q.get(timeout=5)
        except _queue.Empty:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Cancelled")
            raise RuntimeError(f"Subprocess {proc.pid} exited without a result")

        if outcome[0] == "ok":
            return outcome[1]
        _, err_msg, tb = outcome
        raise RuntimeError(f"{err_msg}\n{tb}")

    finally:
        if proc.is_alive():
            # Give a graceful window only if we actually asked to stop.
            if cancel_ev.is_set():
                proc.join(timeout=10)
            if proc.is_alive():
                logger.warning("subprocess_runner: terminating pid={}", proc.pid)
                proc.terminate()
                proc.join(timeout=3)
            if proc.is_alive():
                logger.error("subprocess_runner: killing pid={}", proc.pid)
                try:
                    os.kill(proc.pid, 9)  # type: ignore[arg-type]
                except Exception:
                    pass
                proc.join(timeout=1)
        try:
            prog_q.close()
            result_q.close()
        except Exception:
            pass
