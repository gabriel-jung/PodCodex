"""The MiMi mask patch in `core/synthesize.py` (see ML_RUNTIME.md).

It replaces a private transformers function, so a transformers upgrade can
remove or reshape its target without any error until a synthesis fails deep
inside a forward pass. These pin the target's existence and that the
broadcast builder produces the same mask values as the vmap original.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
mu = pytest.importorskip("transformers.masking_utils")

from podcodex.core import synthesize  # noqa: E402


@pytest.fixture
def patched(monkeypatch):
    """Apply the patch, restoring transformers and the idempotence flag."""
    original = mu._vmap_for_bhqkv
    monkeypatch.setattr(mu, "_vmap_for_bhqkv", original)
    monkeypatch.setattr(synthesize, "_SDPA_MASK_PATCHED", False)
    synthesize._patch_sdpa_mask_for_mimi_vmap_bug()
    return original, mu._vmap_for_bhqkv


def test_the_patch_target_exists():
    assert callable(getattr(mu, "_vmap_for_bhqkv", None))


def _aranges(b=2, h=3, q=5, kv=5):
    return torch.arange(b), torch.arange(h), torch.arange(q), torch.arange(kv)


def test_causal_mask_matches_the_vmap_builder(patched):
    original, broadcast = patched
    args = _aranges()
    expected = original(mu.causal_mask_function)(*args)
    got = broadcast(mu.causal_mask_function)(*args)
    assert torch.equal(got.expand_as(expected), expected)


def test_causal_and_padding_mask_matches_a_cell_by_cell_reference(patched):
    """The vmap original raises on this one on CPU (indexing a tensor by
    index tensors calls .item(), the bug the patch exists for), so the
    reference is built one cell at a time."""
    _original, broadcast = patched
    padding = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    fn = mu.and_masks(mu.causal_mask_function, mu.padding_mask_function(padding))
    b, h, q, kv = 2, 3, 5, 5
    expected = torch.tensor(
        [
            [
                [
                    [
                        bool(fn(bi, hi, torch.tensor(qi), torch.tensor(ki)))
                        for ki in range(kv)
                    ]
                    for qi in range(q)
                ]
                for hi in range(h)
            ]
            for bi in range(b)
        ]
    )
    got = broadcast(fn)(*_aranges(b, h, q, kv))
    assert torch.equal(got.expand_as(expected), expected)


def test_a_missing_target_fails_loudly(monkeypatch):
    monkeypatch.delattr(mu, "_vmap_for_bhqkv")
    monkeypatch.setattr(synthesize, "_SDPA_MASK_PATCHED", False)
    with pytest.raises(RuntimeError, match="_vmap_for_bhqkv"):
        synthesize._patch_sdpa_mask_for_mimi_vmap_bug()
