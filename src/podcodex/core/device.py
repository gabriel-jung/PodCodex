"""Device, dtype, and compute-type resolution — single source of truth.

All callsites that previously did ``torch.cuda.is_available()`` directly
or hardcoded ``compute_type="float16"`` / ``torch_dtype=torch.bfloat16``
should route through this module instead. Two reasons:

1. **PODCODEX_DEVICE env override.** Set ``PODCODEX_DEVICE=cpu`` to skip
   GPU init even when CUDA is available — useful for ``make dev-no-tauri``
   testing on a workstation that has a GPU, or as an escape hatch when
   the installed torch wheel doesn't match the GPU (e.g. cu128 wheels on
   a Pascal box). ``PODCODEX_DEVICE=cuda`` raises if CUDA is unavailable.
   Default is ``auto``.

2. **Pascal GPUs need different precision.** GTX 10xx / Titan Xp / P40 /
   P100 have compute capability 6.x. CTranslate2 refuses ``float16`` on
   them (FP16 runs at 1/64 of FP32 speed, so the library blocks rather
   than silently degrading). Qwen3-TTS's default ``bfloat16`` requires
   sm_80+ (Ampere). This module picks the right precision per capability:

   ====================================  =================  =============
   Compute capability                    compute_type       torch_dtype
   ====================================  =================  =============
   sm_80+ (Ampere/Ada/Blackwell)         float16            bfloat16
   sm_70 / sm_75 (Volta/Turing)          float16            float16
   sm_60 / sm_61 / sm_62 (Pascal)        int8_float32       float32
   CPU / forced cpu                      int8               float32
   ====================================  =================  =============

Env-var reads are lazy (inside each function), not at import time — keeps
this module safe to import before ``bootstrap_for_*()`` runs.
"""

from __future__ import annotations

import os
from typing import Any, Literal

DeviceOverride = Literal["auto", "cpu", "cuda"]
_VALID_OVERRIDES: frozenset[str] = frozenset({"auto", "cpu", "cuda"})


def user_override() -> DeviceOverride:
    """Return the active ``PODCODEX_DEVICE`` value, normalized.

    Unknown / unset values fall back to ``"auto"``. The env var is read
    fresh on every call — bootstrap may set it after import time.
    """
    raw = os.environ.get("PODCODEX_DEVICE", "auto").strip().lower()
    if raw in _VALID_OVERRIDES:
        return raw  # type: ignore[return-value]
    return "auto"


def _compute_capability() -> tuple[int, int] | None:
    """Return (major, minor) compute capability of cuda:0, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def cuda_available() -> bool:
    """Replacement for ``torch.cuda.is_available()`` that honors the env override.

    Returns False if ``PODCODEX_DEVICE=cpu`` regardless of physical GPU,
    True if ``PODCODEX_DEVICE=cuda`` (caller is asserting; we trust them
    here — ``resolve_device`` does the validation), else falls through
    to torch's own check.
    """
    forced = user_override()
    if forced == "cpu":
        return False
    if forced == "cuda":
        return True
    return _probe_cuda()


def physical_cuda_available() -> bool:
    """Probe torch directly, ignoring ``PODCODEX_DEVICE`` overrides.

    Use when validating whether a *new* override is feasible (e.g. user
    flipping the GPU panel from CPU → CUDA): the existing override would
    short-circuit ``cuda_available`` and produce circular validation.
    """
    return _probe_cuda()


def _probe_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_str() -> str:
    """Return ``"cuda"`` or ``"cpu"`` (no index suffix). For pyannote, whisperx,
    sentence-transformers etc. that take a string device argument."""
    return "cuda" if cuda_available() else "cpu"


def resolve_device() -> tuple[str, str]:
    """Return ``(device, compute_type)`` for WhisperX / faster-whisper / CTranslate2.

    Validates ``PODCODEX_DEVICE=cuda`` against actual availability — raises
    if the caller demanded CUDA but none is present.
    """
    forced = user_override()
    if forced == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "PODCODEX_DEVICE=cuda was set, but no CUDA device is available. "
                    "Unset the env var or set PODCODEX_DEVICE=cpu / auto."
                )
        except ImportError as exc:
            raise RuntimeError(
                "PODCODEX_DEVICE=cuda was set, but torch is not installed."
            ) from exc

    if not cuda_available():
        return "cpu", "int8"

    cc = _compute_capability()
    if cc is None:
        return "cuda", "float16"
    major, _minor = cc
    if major >= 7:
        return "cuda", "float16"
    return "cuda", "int8_float32"


def torch_dtype() -> Any:
    """Return ``torch.bfloat16`` / ``torch.float16`` / ``torch.float32`` based on
    detected compute capability. CPU and Pascal both get float32 (Pascal lacks
    bfloat16 hardware and FP16 matmul is unusably slow)."""
    import torch

    if not cuda_available():
        return torch.float32
    cc = _compute_capability()
    if cc is None:
        return torch.float16
    major, _minor = cc
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return torch.float32


def assert_kernels_available() -> None:
    """Raise if a CUDA device is detected but the installed torch wheel was not
    compiled with kernels for its compute capability.

    Catches the user-visible failure mode where someone with a GTX 1080
    (sm_61) installs the default cu128 wheel, which only ships sm_75+.
    The cryptic runtime error is ``CUDA error: no kernel image is
    available for execution on the device``; this raises early with a
    clear, actionable message.

    No-op if ``PODCODEX_DEVICE=cpu`` (no GPU init wanted) or no CUDA
    device is physically present.
    """
    if user_override() == "cpu":
        return
    cc = _compute_capability()
    if cc is None:
        return
    try:
        import torch

        arch_list = list(torch.cuda.get_arch_list())
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        return
    if not arch_list:
        return

    major, minor = cc
    target = f"sm_{major}{minor}"
    # PTX JIT can run an sm_X.Y binary on any sm_X.Z device of the same
    # major, so any "sm_{major}*" entry in arch_list is a kernel match.
    major_prefix = f"sm_{major}"
    if any(a.startswith(major_prefix) for a in arch_list):
        return

    raise RuntimeError(
        f"GPU detected ({device_name}, compute capability {major}.{minor}) "
        f"but the installed PyTorch wheel has no kernels for {target}. "
        f"Available archs: {arch_list}. "
        f"Reinstall torch with a wheel that supports your GPU "
        f"(see deploy/PASCAL.md for older GPUs), or set "
        f"PODCODEX_DEVICE=cpu to skip GPU initialization."
    )


def device_info() -> dict[str, Any]:
    """Diagnostic snapshot for the health endpoint.

    Returns a dict with the resolved device, compute_type, dtype, GPU name,
    compute capability, arch list, and whether the env override is active.
    Safe to call at any time — never raises.
    """
    forced = user_override()
    info: dict[str, Any] = {
        "override": forced if forced != "auto" else None,
        "available": cuda_available(),
    }
    try:
        device, compute_type = resolve_device()
        info["device"] = device
        info["compute_type"] = compute_type
    except Exception as exc:
        info["device"] = "unknown"
        info["compute_type"] = "unknown"
        info["error"] = repr(exc)

    try:
        import torch

        dtype = torch_dtype()
        info["torch_dtype"] = str(dtype).removeprefix("torch.")
        if cuda_available():
            cc = _compute_capability()
            if cc is not None:
                info["compute_capability"] = f"{cc[0]}.{cc[1]}"
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
            try:
                info["arch_list"] = list(torch.cuda.get_arch_list())
            except Exception:
                pass
    except Exception:
        pass
    return info
