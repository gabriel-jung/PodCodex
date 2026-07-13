"""Chonkie mocks shared by chunker-adjacent tests (no model loading)."""

from unittest.mock import MagicMock, patch


def make_mock_chunk(text: str, start: int, end: int, token_count: int = 10):
    c = MagicMock()
    c.text = text
    c.start_index = start
    c.end_index = end
    c.token_count = token_count
    return c


def mock_chonkie():
    """Return (context manager, mock module) that mocks chonkie in sys.modules."""
    mock_mod = MagicMock()
    return patch.dict("sys.modules", {"chonkie": mock_mod}), mock_mod
