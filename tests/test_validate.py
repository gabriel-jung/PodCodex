"""Tests for podcodex.core.validate_segments_json."""

from podcodex.core import validate_segments_json


def test_valid_segments():
    data = [{"text": "hello", "start": 0.0}]
    assert validate_segments_json(data) is None


def test_not_a_list():
    assert "JSON array" in validate_segments_json("hello")


def test_dict_instead_of_list():
    err = validate_segments_json({"segments": [], "text": "hi"})
    assert "object with keys" in err
    assert "Whisper" in err


def test_empty_list():
    assert "empty" in validate_segments_json([])


def test_elements_not_dicts():
    assert "object" in validate_segments_json(["a", "b"])


def test_missing_required_field():
    err = validate_segments_json([{"speaker": "A"}])
    assert "text" in err


def test_custom_required_fields():
    err = validate_segments_json([{"text": "hi"}], required=("text", "start"))
    assert "start" in err


def test_all_required_present():
    data = [{"text": "hi", "start": 0.0, "end": 1.0}]
    assert validate_segments_json(data, required=("text", "start")) is None
