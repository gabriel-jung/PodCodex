"""Tests for podcodex.core.device — env override, capability mapping, kernel guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcodex.core import device


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with PODCODEX_DEVICE unset."""
    monkeypatch.delenv("PODCODEX_DEVICE", raising=False)


def _fake_torch(
    *,
    cuda_available: bool = True,
    capability: tuple[int, int] = (8, 0),
    arch_list: list[str] | None = None,
    device_name: str = "FakeGPU",
) -> MagicMock:
    """Build a torch stand-in with the cuda surface the device module touches."""
    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = cuda_available
    fake.cuda.get_device_capability.return_value = capability
    fake.cuda.get_device_name.return_value = device_name
    fake.cuda.get_arch_list.return_value = (
        arch_list if arch_list is not None else ["sm_70", "sm_75", "sm_80", "sm_90"]
    )
    # Real torch dtype attributes — referenced by torch_dtype()
    import torch as _real

    fake.bfloat16 = _real.bfloat16
    fake.float16 = _real.float16
    fake.float32 = _real.float32
    return fake


# ──────────────────────────────────────────────
# Env override
# ──────────────────────────────────────────────


def test_cpu_override_forces_cpu_even_with_gpu(monkeypatch):
    monkeypatch.setenv("PODCODEX_DEVICE", "cpu")
    fake = _fake_torch(cuda_available=True, capability=(8, 0))
    with patch.dict("sys.modules", {"torch": fake}):
        assert device.cuda_available() is False
        assert device.device_str() == "cpu"
        assert device.resolve_device() == ("cpu", "int8")


def test_cuda_override_with_no_gpu_raises(monkeypatch):
    monkeypatch.setenv("PODCODEX_DEVICE", "cuda")
    fake = _fake_torch(cuda_available=False)
    with patch.dict("sys.modules", {"torch": fake}):
        with pytest.raises(RuntimeError, match="PODCODEX_DEVICE=cuda"):
            device.resolve_device()


def test_auto_with_no_gpu_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("PODCODEX_DEVICE", "auto")
    fake = _fake_torch(cuda_available=False)
    with patch.dict("sys.modules", {"torch": fake}):
        assert device.cuda_available() is False
        assert device.resolve_device() == ("cpu", "int8")


# ──────────────────────────────────────────────
# Compute-capability → compute_type
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "capability,expected",
    [
        ((8, 0), ("cuda", "float16")),  # Ampere
        ((8, 9), ("cuda", "float16")),  # Ada
        ((9, 0), ("cuda", "float16")),  # Hopper
        ((10, 0), ("cuda", "float16")),  # Blackwell
        ((7, 5), ("cuda", "float16")),  # Turing
        ((7, 0), ("cuda", "float16")),  # Volta
        ((6, 1), ("cuda", "int8_float32")),  # Pascal GTX 1080
        ((6, 0), ("cuda", "int8_float32")),  # Pascal P100
    ],
)
def test_resolve_device_per_capability(capability, expected):
    fake = _fake_torch(cuda_available=True, capability=capability)
    with patch.dict("sys.modules", {"torch": fake}):
        assert device.resolve_device() == expected


# ──────────────────────────────────────────────
# Compute-capability → torch_dtype
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "capability,dtype_name",
    [
        ((8, 0), "bfloat16"),
        ((8, 9), "bfloat16"),
        ((10, 0), "bfloat16"),
        ((7, 5), "float16"),
        ((7, 0), "float16"),
        ((6, 1), "float32"),
        ((6, 0), "float32"),
    ],
)
def test_torch_dtype_per_capability(capability, dtype_name):
    import torch as _real

    fake = _fake_torch(cuda_available=True, capability=capability)
    with patch.dict("sys.modules", {"torch": fake}):
        assert device.torch_dtype() is getattr(_real, dtype_name)


def test_torch_dtype_cpu_returns_float32(monkeypatch):
    monkeypatch.setenv("PODCODEX_DEVICE", "cpu")
    import torch as _real

    fake = _fake_torch(cuda_available=True, capability=(8, 0))
    with patch.dict("sys.modules", {"torch": fake}):
        assert device.torch_dtype() is _real.float32


# ──────────────────────────────────────────────
# Kernel guard
# ──────────────────────────────────────────────


def test_assert_kernels_available_passes_when_arch_matches():
    fake = _fake_torch(
        cuda_available=True, capability=(8, 0), arch_list=["sm_75", "sm_80"]
    )
    with patch.dict("sys.modules", {"torch": fake}):
        device.assert_kernels_available()  # no raise


def test_assert_kernels_available_raises_when_pascal_missing_from_cu128():
    fake = _fake_torch(
        cuda_available=True,
        capability=(6, 1),
        arch_list=["sm_75", "sm_80", "sm_90"],
        device_name="GeForce GTX 1080",
    )
    with patch.dict("sys.modules", {"torch": fake}):
        with pytest.raises(RuntimeError, match="GTX 1080.*sm_61"):
            device.assert_kernels_available()


def test_assert_kernels_available_passes_for_pascal_with_sm_61():
    fake = _fake_torch(
        cuda_available=True,
        capability=(6, 1),
        arch_list=["sm_60", "sm_61", "sm_70", "sm_75"],
    )
    with patch.dict("sys.modules", {"torch": fake}):
        device.assert_kernels_available()  # no raise


def test_assert_kernels_available_noop_when_cpu_forced(monkeypatch):
    monkeypatch.setenv("PODCODEX_DEVICE", "cpu")
    fake = _fake_torch(cuda_available=True, capability=(6, 1), arch_list=["sm_80"])
    with patch.dict("sys.modules", {"torch": fake}):
        device.assert_kernels_available()  # no raise


def test_assert_kernels_available_noop_with_no_gpu():
    fake = _fake_torch(cuda_available=False)
    with patch.dict("sys.modules", {"torch": fake}):
        device.assert_kernels_available()


# ──────────────────────────────────────────────
# device_info diagnostic
# ──────────────────────────────────────────────


def test_device_info_reports_override(monkeypatch):
    monkeypatch.setenv("PODCODEX_DEVICE", "cpu")
    fake = _fake_torch(cuda_available=True, capability=(8, 0))
    with patch.dict("sys.modules", {"torch": fake}):
        info = device.device_info()
    assert info["override"] == "cpu"
    assert info["device"] == "cpu"
    assert info["compute_type"] == "int8"
    assert info["available"] is False


def test_device_info_reports_pascal_capability():
    fake = _fake_torch(
        cuda_available=True,
        capability=(6, 1),
        arch_list=["sm_60", "sm_61", "sm_70"],
        device_name="GeForce GTX 1080",
    )
    with patch.dict("sys.modules", {"torch": fake}):
        info = device.device_info()
    assert info["device"] == "cuda"
    assert info["compute_type"] == "int8_float32"
    assert info["compute_capability"] == "6.1"
    assert info["torch_dtype"] == "float32"
    assert info["gpu_name"] == "GeForce GTX 1080"
