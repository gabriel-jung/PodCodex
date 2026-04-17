"""Tests for _approx_substring (Sellers' algorithm) in index_store."""

from __future__ import annotations

import pytest

from podcodex.rag.index_store import _approx_substring, fold_text


# ── Distance basics ─────────────────────────────────────────────────────


def test_empty_pattern_returns_zero():
    assert _approx_substring("", "anything here", 5) == (0, 0, 0)


def test_exact_substring_in_text():
    hit = _approx_substring("fox", "the quick brown fox jumps", 1)
    assert hit is not None
    d, s, e = hit
    assert d == 0
    assert "the quick brown fox jumps"[s:e] == "fox"


def test_no_match_returns_none():
    assert _approx_substring("fox", "the quick brown dog", 1) is None


def test_zero_tolerance_rejects_one_typo():
    assert _approx_substring("fox", "foy is here", 0) is None


def test_one_typo_within_tolerance():
    """Sub within tolerance — span is leftmost among ties (algo picks
    minimum end index when several substrings share the same distance)."""
    hit = _approx_substring("fox", "foy is here", 1)
    assert hit is not None
    d, _s, _e = hit
    assert d == 1


def test_insertion_in_text():
    # pattern="cat", text="caat" — one insertion; any dist-1 alignment fine
    hit = _approx_substring("cat", "caat", 1)
    assert hit is not None
    assert hit[0] == 1


def test_deletion_in_text():
    # pattern="cart", text contains "cat" — one deletion from pattern
    hit = _approx_substring("cart", "a cat sat", 1)
    assert hit is not None
    assert hit[0] == 1


def test_prefers_minimum_distance_over_tolerance():
    # text has both an exact and a 1-edit match; algorithm should pick dist=0.
    text = "abc def abd"
    hit = _approx_substring("abc", text, 2)
    assert hit is not None
    assert hit[0] == 0  # exact match present


# ── Regression: êtres de lumière / lumière très ─────────────────────────


def test_etres_de_lumiere_rejects_lumiere_tres():
    """The bug: 'lumière très' was accepted as a fuzzy match for
    'êtres de lumière' because the old token-window fuzzy ignored order."""
    q = fold_text("êtres de lumière")  # "etres de lumiere" — len 16
    text = fold_text("ces paroles, lumière très claire, sont d'une portée")
    # ~12% of 16 = 2 edits. Reordering "lumière très" to match "etres de
    # lumiere" needs far more than 2 edits.
    assert _approx_substring(q, text, max(1, len(q) // 8)) is None


def test_etres_de_lumiere_accepts_one_typo():
    """One real typo in the phrase should still match."""
    q = fold_text("êtres de lumière")
    text = fold_text("les etre de lumiere apparaissent")  # missing 's' in "etre"
    hit = _approx_substring(q, text, max(1, len(q) // 8))
    assert hit is not None
    assert hit[0] == 1


def test_etres_de_lumiere_accepts_exact_accent_folded():
    q = fold_text("êtres de lumière")
    text = fold_text("les êtres de lumière descendent")
    hit = _approx_substring(q, text, max(1, len(q) // 8))
    assert hit is not None
    assert hit[0] == 0
    assert text[hit[1] : hit[2]] == "etres de lumiere"


# ── Order preservation ──────────────────────────────────────────────────


def test_order_matters():
    """Substring search cannot match reversed word order cheaply."""
    q = "cat dog"
    text = "a dog and a cat in hat"
    assert _approx_substring(q, text, 1) is None


def test_same_words_in_order_ok():
    q = "cat dog"
    text = "a cat and a dog"
    # 5 edits: "cat and a dog" → "cat dog" needs removing "and a "
    # so at dist 1 it should fail; at a larger budget it would pass.
    assert _approx_substring(q, text, 1) is None
    hit = _approx_substring(q, text, 8)
    assert hit is not None


# ── Longer phrases ──────────────────────────────────────────────────────


def test_long_phrase_one_typo():
    q = fold_text("la conscience collective evolue")
    text = fold_text("parfois la conscience collective evoque un changement")
    # "evolue" -> "evoque" is 2 substitutions. Allow up to ~12%.
    hit = _approx_substring(q, text, max(1, len(q) // 8))
    assert hit is not None
    assert hit[0] <= 2


def test_long_phrase_many_differences_rejected():
    q = fold_text("la conscience collective evolue")
    text = fold_text("mais la foule applaudit le discours tres longtemps")
    assert _approx_substring(q, text, max(1, len(q) // 8)) is None


# ── Ligatures and accents via fold_text ─────────────────────────────────


def test_ligature_folds_and_matches():
    q = fold_text("cœur battant")
    text = fold_text("son coeur battant s'accelere")
    hit = _approx_substring(q, text, 0)
    assert hit is not None
    assert hit[0] == 0


# ── Early exit safety ───────────────────────────────────────────────────


def test_early_exit_on_long_text_no_match():
    """Early termination when every cell exceeds max_dist returns None fast."""
    q = "needle"
    text = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    assert _approx_substring(q, text, 1) is None


# ── Parametrized small cases ────────────────────────────────────────────


@pytest.mark.parametrize(
    "pattern,text,max_dist,expected_dist",
    [
        ("kitten", "the kitten sleeps", 0, 0),
        # kitten vs substring "sittin": k→s, e→i = 2 subs (no insert needed
        # when we can pick a 6-char substring of "sitting")
        ("kitten", "the sitting sleeps", 2, 2),
        ("abcdef", "zzabcdefzz", 0, 0),
        ("abc", "a-b-c", 2, 2),  # two insertions
        ("abc", "axbxc", 2, 2),  # two insertions
    ],
)
def test_parametrized(pattern, text, max_dist, expected_dist):
    hit = _approx_substring(pattern, text, max_dist)
    assert hit is not None
    assert hit[0] == expected_dist
