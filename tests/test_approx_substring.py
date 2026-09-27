"""Tests for approx_substring (Sellers' algorithm) in index_store."""

from __future__ import annotations

import pytest

from podcodex.rag.index_store import approx_substring, fold_text


# ── Distance basics ─────────────────────────────────────────────────────


def test_empty_pattern_returns_zero():
    assert approx_substring("", "anything here", 5) == (0, 0, 0)


def test_no_match_returns_none():
    assert approx_substring("fox", "the quick brown dog", 1) is None


def test_zero_tolerance_rejects_one_typo():
    assert approx_substring("fox", "foy is here", 0) is None


def test_deletion_in_text():
    # pattern="cart", text contains "cat" — one deletion from pattern
    hit = approx_substring("cart", "a cat sat", 1)
    assert hit is not None
    assert hit[0] == 1


def test_prefers_minimum_distance_over_tolerance():
    # text has both an exact and a 1-edit match; algorithm should pick dist=0.
    text = "abc def abd"
    hit = approx_substring("abc", text, 2)
    assert hit is not None
    assert hit[0] == 0  # exact match present


# ── Word order: êtres de lumière / lumière très ─────────────────────────


def test_etres_de_lumiere_rejects_lumiere_tres():
    """'lumière très' is not a fuzzy match for 'êtres de lumière': the
    match must respect word order, not just share tokens."""
    q = fold_text("êtres de lumière")  # "etres de lumiere" — len 16
    text = fold_text("ces paroles, lumière très claire, sont d'une portée")
    # ~12% of 16 = 2 edits. Reordering "lumière très" to match "etres de
    # lumiere" needs far more than 2 edits.
    assert approx_substring(q, text, max(1, len(q) // 8)) is None


def test_etres_de_lumiere_accepts_one_typo():
    """One real typo in the phrase should still match."""
    q = fold_text("êtres de lumière")
    text = fold_text("les etre de lumiere apparaissent")  # missing 's' in "etre"
    hit = approx_substring(q, text, max(1, len(q) // 8))
    assert hit is not None
    assert hit[0] == 1


def test_etres_de_lumiere_accepts_exact_accent_folded():
    q = fold_text("êtres de lumière")
    text = fold_text("les êtres de lumière descendent")
    hit = approx_substring(q, text, max(1, len(q) // 8))
    assert hit is not None
    assert hit[0] == 0
    assert text[hit[1] : hit[2]] == "etres de lumiere"


# ── Order preservation ──────────────────────────────────────────────────


def test_same_words_in_order_ok():
    q = "cat dog"
    text = "a cat and a dog"
    # 5 edits: "cat and a dog" → "cat dog" needs removing "and a "
    # so at dist 1 it should fail; at a larger budget it would pass.
    assert approx_substring(q, text, 1) is None
    hit = approx_substring(q, text, 8)
    assert hit is not None


# ── Longer phrases ──────────────────────────────────────────────────────


def test_long_phrase_one_typo():
    q = fold_text("la conscience collective evolue")
    text = fold_text("parfois la conscience collective evoque un changement")
    # "evolue" -> "evoque" is 2 substitutions. Allow up to ~12%.
    hit = approx_substring(q, text, max(1, len(q) // 8))
    assert hit is not None
    assert hit[0] <= 2


def test_long_phrase_many_differences_rejected():
    q = fold_text("la conscience collective evolue")
    text = fold_text("mais la foule applaudit le discours tres longtemps")
    assert approx_substring(q, text, max(1, len(q) // 8)) is None


# ── Ligatures and accents via fold_text ─────────────────────────────────


def test_ligature_folds_and_matches():
    q = fold_text("cœur battant")
    text = fold_text("son coeur battant s'accelere")
    hit = approx_substring(q, text, 0)
    assert hit is not None
    assert hit[0] == 0


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
    hit = approx_substring(pattern, text, max_dist)
    assert hit is not None
    assert hit[0] == expected_dist
