"""Tests for podcodex.core.subtitles: the text / SRT / VTT formatters and the
SRT / VTT parsers the YouTube subtitle import rides on."""

import pytest

from podcodex.core.subtitles import segments_to_srt, segments_to_text, segments_to_vtt


def test_segments_to_text_empty():
    assert segments_to_text([]) == ""


def test_segments_to_text_contains_speaker_text_and_timestamps():
    segments = [{"speaker": "Alice", "start": 1.0, "end": 3.5, "text": "Hello"}]
    out = segments_to_text(segments)
    assert "Alice" in out
    assert "Hello" in out
    assert "1.000s" in out
    assert "3.500s" in out


def test_segments_to_text_preserves_order():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "Hi"},
    ]
    out = segments_to_text(segments)
    assert out.index("Alice") < out.index("Bob")


def test_segments_to_srt_format():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 1.5, "text": "Hi"}]
    out = segments_to_srt(segments)
    assert "00:00:00,000 --> 00:00:01,500" in out
    assert "Alice: Hi" in out


def test_segments_to_vtt_format():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "Hi"}]
    out = segments_to_vtt(segments)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000" in out
    assert "<v Alice>Hi" in out


def test_exports_name_nobody_for_the_placeholder():
    from podcodex.core._utils import NARRATOR_SPEAKER

    segs = [
        {"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "bonjour"},
        {"speaker": NARRATOR_SPEAKER, "start": 1.0, "end": 2.0, "text": "plus tard"},
    ]
    out = segments_to_srt(segs)
    assert f"{NARRATOR_SPEAKER}:" not in out
    assert "Alice: bonjour" in out


@pytest.mark.legacy("narrator-rename")
def test_exports_keep_a_declared_narrator_named():
    """A show's own narrator must not be the only unnamed line in an export."""
    segs = [{"speaker": "Narrator", "start": 1.0, "end": 2.0, "text": "plus tard"}]
    # Undeclared, it is the legacy placeholder and names nobody.
    assert "Narrator:" not in segments_to_srt(segs)
    # Declared, it is a name like any other.
    assert "Narrator: plus tard" in segments_to_srt(segs, declared={"Narrator"})


# ── srt_to_segments / vtt_to_segments ────────────────────────────────────
#
# The whole YouTube subtitle import path rides on these two parsers, and the
# empty-speaker footgun lives here: an untagged cue must come out as
# NARRATOR_SPEAKER, never as "". NARRATOR_SPEAKER's value itself changed in
# 0.2.10, so the tests assert against the constant, not the string.


def test_srt_parses_timestamps_and_speaker_prefix():
    from podcodex.core.subtitles import srt_to_segments

    srt = (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Alice: Hello there\n"
        "\n"
        "2\n"
        "00:01:02,250 --> 00:01:03,000\n"
        "Bob: Hi\n"
    )
    segs = srt_to_segments(srt)
    assert [s["speaker"] for s in segs] == ["Alice", "Bob"]
    assert [s["text"] for s in segs] == ["Hello there", "Hi"]
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 1.5
    assert segs[1]["start"] == 62.25


def test_srt_without_a_speaker_prefix_defaults_to_narrator():
    from podcodex.core._utils import NARRATOR_SPEAKER
    from podcodex.core.subtitles import srt_to_segments

    srt = "1\n00:00:00,000 --> 00:00:02,000\nJust narration\n"
    segs = srt_to_segments(srt)
    assert segs[0]["speaker"] == NARRATOR_SPEAKER
    assert segs[0]["speaker"] != ""


def test_srt_does_not_mistake_punctuated_text_for_a_speaker():
    """ "Wait, what?: no" is a sentence, not a "Speaker: text" prefix."""
    from podcodex.core._utils import NARRATOR_SPEAKER
    from podcodex.core.subtitles import srt_to_segments

    srt = "1\n00:00:00,000 --> 00:00:02,000\nWait, what: no\n"
    segs = srt_to_segments(srt)
    assert segs[0]["speaker"] == NARRATOR_SPEAKER
    assert segs[0]["text"] == "Wait, what: no"


def test_vtt_voice_tag_becomes_the_speaker():
    from podcodex.core.subtitles import vtt_to_segments

    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<v Alice>Hello there</v>\n"
    segs = vtt_to_segments(vtt)
    assert segs == [
        {"speaker": "Alice", "text": "Hello there", "start": 0.0, "end": 1.0}
    ]


def test_vtt_without_a_voice_tag_defaults_to_narrator():
    """The YouTube import case: no <v> tag anywhere in the track."""
    from podcodex.core._utils import NARRATOR_SPEAKER
    from podcodex.core.subtitles import vtt_to_segments

    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nplain line\n"
    segs = vtt_to_segments(vtt)
    assert segs[0]["speaker"] == NARRATOR_SPEAKER
    assert segs[0]["speaker"] != ""


def test_vtt_strips_cue_settings_markup_and_entities():
    from podcodex.core.subtitles import vtt_to_segments

    vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000 align:start position:0%\n"
        "plain <i>line</i> &amp; more\n"
    )
    segs = vtt_to_segments(vtt)
    assert segs[0]["text"] == "plain line & more"
    assert segs[0]["start"] == 1.0 and segs[0]["end"] == 2.0


def test_overlapping_youtube_cues_collapse_into_one_segment():
    """Repeated consecutive cues extend the previous one instead of doubling."""
    from podcodex.core.subtitles import vtt_to_segments

    vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\nhello\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello\n\n"
        "00:00:02.000 --> 00:00:03.000\nworld\n"
    )
    segs = vtt_to_segments(vtt)
    assert [s["text"] for s in segs] == ["hello", "world"]
    assert segs[0]["end"] == 2.0


def test_both_parsers_ignore_blocks_without_a_timestamp():
    from podcodex.core.subtitles import srt_to_segments, vtt_to_segments

    assert srt_to_segments("") == []
    assert vtt_to_segments("WEBVTT\n\nNOTE nothing to see\n") == []


def test_srt_round_trips_through_the_formatter():
    from podcodex.core.subtitles import segments_to_srt, srt_to_segments

    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 1.5, "text": "Hi"},
        {"speaker": "Bob", "start": 1.5, "end": 3.0, "text": "Hello"},
    ]
    parsed = srt_to_segments(segments_to_srt(segments))
    assert parsed == segments


def test_vtt_round_trips_through_the_formatter():
    from podcodex.core.subtitles import segments_to_vtt, vtt_to_segments

    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "Hi"},
        {"speaker": "Bob", "start": 1.0, "end": 2.0, "text": "Hello"},
    ]
    parsed = vtt_to_segments(segments_to_vtt(segments))
    assert parsed == segments


def test_srt_colons_in_ordinary_text_are_not_speakers():
    """An ordinary `Word: rest` cue does not mint a speaker named Word."""
    from podcodex.core._utils import NARRATOR_SPEAKER
    from podcodex.core.subtitles import srt_to_segments

    srt = "\n\n".join(
        f"{i + 1}\n00:00:0{i},000 --> 00:00:0{i},900\n{text}"
        for i, text in enumerate(
            [
                "Welcome back to the show.",
                "Note: this episode was recorded live.",
                "We talked about many things.",
                "So here's the thing: it worked.",
                "Thanks for listening.",
            ]
        )
    )
    segs = srt_to_segments(srt)
    assert {s["speaker"] for s in segs} == {NARRATOR_SPEAKER}
    assert "Note: this episode was recorded live." in [s["text"] for s in segs]


def test_srt_recurring_and_capitalised_labels_are_speakers():
    from podcodex.core._utils import NARRATOR_SPEAKER
    from podcodex.core.subtitles import srt_to_segments

    srt = "\n\n".join(
        f"{i + 1}\n00:00:0{i},000 --> 00:00:0{i},900\n{text}"
        for i, text in enumerate(
            [
                "Some narration first.",
                "More narration here.",
                "Alice: Hello.",
                "Then a pause.",
                "Alice: Me again.",
                "HOST: Welcome.",
                "Closing words now.",
            ]
        )
    )
    speakers = [s["speaker"] for s in srt_to_segments(srt)]
    assert "Alice" in speakers and "HOST" in speakers
    assert NARRATOR_SPEAKER in speakers


def _cue(start, text, speaker="A", dur=1.0):
    return {"speaker": speaker, "text": text, "start": start, "end": start + dur}


def test_rolling_subtitles_keep_only_the_new_words():
    """YouTube auto-captions repeat the previous line before adding words."""
    from podcodex.core.subtitles import merge_parsed_cues

    cues = [
        _cue(0, "we went down to the river"),
        _cue(1, "we went down to the river and found a boat"),
        _cue(2, "and found a boat tied to the old dock"),
        _cue(3, "tied to the old dock so we climbed in"),
    ]
    texts = [c["text"] for c in merge_parsed_cues(cues)]
    assert texts == [
        "we went down to the river",
        "and found a boat",
        "tied to the old dock",
        "so we climbed in",
    ]


def test_a_short_repeat_is_not_stripped():
    """The collapse strips a real repeat, not any overlap down to one
    character: "Okay. So what now?" after "Okay." keeps its "Okay."."""
    from podcodex.core.subtitles import merge_parsed_cues

    cues = [
        _cue(0, "we went down to the river"),
        _cue(1, "we went down to the river and found a boat"),
        _cue(2, "and found a boat tied to the old dock"),
        _cue(3, "Okay."),
        _cue(4, "Okay. So what now?"),
    ]
    texts = [c["text"] for c in merge_parsed_cues(cues)]
    assert texts[-1] == "Okay. So what now?"


def test_html_entities_are_decoded_once():
    from podcodex.core.subtitles import merge_parsed_cues

    (cue,) = merge_parsed_cues([_cue(0, "a &amp;lt; b&nbsp;&nbsp;c &quot;d&quot;")])
    assert cue["text"] == 'a &lt; b c "d"'


@pytest.mark.parametrize(
    "seconds,expected",
    [(1.001, "00:00:01,001"), (59.9996, "00:01:00,000"), (3661.5, "01:01:01,500")],
)
def test_subtitle_timestamps_round_to_the_millisecond(seconds, expected):
    """Truncating each field dropped a millisecond (1.001 came out ,000)."""
    from podcodex.core.subtitles import _parse_ts, _srt_ts, _vtt_ts

    assert _srt_ts(seconds) == expected
    assert _vtt_ts(seconds) == expected.replace(",", ".")
    assert _parse_ts(expected) == round(seconds, 3)
