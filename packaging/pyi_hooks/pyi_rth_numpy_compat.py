"""PyInstaller runtime hook: numpy 2.x / torch ABI mismatch fallback.

torch wheels compiled against numpy 1.x can fail in frozen bundles where
the FrozenImporter loads numpy 2.x first. ``torch.from_numpy`` then raises
``RuntimeError: Numpy is not available`` even though numpy is importable.

This hook patches ``torch.from_numpy`` with a ctypes ``memmove`` fallback
that bypasses the C-level numpy ABI version check. Adapted from voicebox.
"""

import sys
import threading


def _torch_fully_loaded() -> bool:
    """Wait for torch to finish bootstrapping before we touch ``from_numpy``.

    The race we hit otherwise: the polling thread sees ``torch`` in
    ``sys.modules`` as soon as ``torch.__init__`` STARTS, then overwrites
    ``torch.from_numpy`` with a Python wrapper. ``torch._torch_docs`` (which
    runs later in the same import chain) then calls
    ``add_docstr(torch.from_numpy, ...)`` and the C-side ``add_docstr``
    raises ``TypeError: don't know how to add docstring to type 'function'``.

    Indicators that torch is past _torch_docs:
      - ``torch._torch_docs`` is in ``sys.modules`` (the file ran)
      - ``torch.nn.functional`` is in ``sys.modules`` (it imports _torch_docs)
    """
    return "torch._torch_docs" in sys.modules and "torch.nn.functional" in sys.modules


def _patch_torch_from_numpy() -> None:
    import time

    # Wait up to 360 s for torch to fully load.
    for _ in range(7200):
        time.sleep(0.05)
        torch = sys.modules.get("torch")
        if torch is None or not _torch_fully_loaded():
            continue
        if getattr(torch, "_podcodex_from_numpy_patched", False):
            return

        try:
            import ctypes
            import numpy as np

            _orig = torch.from_numpy

            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
                "uint8": torch.uint8,
                "bool": torch.bool,
                "complex64": torch.complex64,
                "complex128": torch.complex128,
            }

            def _safe_from_numpy(arr):
                try:
                    return _orig(arr)
                except RuntimeError:
                    a = np.ascontiguousarray(arr)
                    key = str(a.dtype)
                    if key not in dtype_map:
                        raise TypeError(
                            f"pyi_rth_numpy_compat: unsupported numpy dtype "
                            f"{key!r}; add an explicit mapping rather than "
                            f"silently copying bytes into the wrong dtype."
                        )
                    out = torch.empty(list(a.shape), dtype=dtype_map[key])
                    ctypes.memmove(out.data_ptr(), a.ctypes.data, a.nbytes)
                    return out

            torch.from_numpy = _safe_from_numpy
            torch._podcodex_from_numpy_patched = True
        except Exception:
            pass
        return


threading.Thread(target=_patch_torch_from_numpy, daemon=True).start()
