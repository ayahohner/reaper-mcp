"""Parsing and note statistics in compose_tools, without REAPER."""

import pytest

from reaper_mcp.compose_tools import note_stats, parse_key, parse_notes, parse_pitch


@pytest.mark.parametrize(("text", "pitch"), [
    ("60", 60), ("C4", 60), ("A4", 69), ("F1", 29), ("F#1", 30), ("Bb0", 22), ("C-1", 0), ("G9", 127),
])
def test_parse_pitch(text, pitch):
    assert parse_pitch(text) == pitch


@pytest.mark.parametrize("text", ["H2", "C", "128", "G#9", "xyz"])
def test_parse_pitch_refuses(text):
    with pytest.raises(ValueError):
        parse_pitch(text)


def test_parse_notes_defaults_and_fields():
    assert parse_notes("F1@0 C2@1.5:0.25 Ab1@2:1:64") == [
        (0.0, 1.0, 29, 100), (1.5, 0.25, 36, 100), (2.0, 1.0, 32, 64)]


def test_parse_notes_chords_and_separators():
    assert parse_notes("F1+C2@0:2:90,\nG1@2") == [
        (0.0, 2.0, 29, 90), (0.0, 2.0, 36, 90), (2.0, 1.0, 31, 100)]


@pytest.mark.parametrize("text", ["F1", "F1@-1", "F1@0:0", "F1@0:1:0", "F1@0:1:90:3"])
def test_parse_notes_refuses(text):
    with pytest.raises(ValueError):
        parse_notes(text)


def test_parse_key():
    tonic, scale = parse_key("F minor")
    assert tonic == 5
    assert scale == {5, 7, 8, 10, 0, 1, 3}
    assert parse_key("C# phrygian")[1] == {1, 2, 4, 6, 8, 9, 11}
    with pytest.raises(ValueError):
        parse_key("F")
    with pytest.raises(ValueError):
        parse_key("F bebop")


def test_note_stats():
    notes = [(0.0, 0.5, 29, 100), (0.5, 0.5, 29, 100), (1.0, 1.0, 30, 100),
             (4.1, 1.0, 41, 60), (4.5, 1.0, 41, 70)]
    stats = note_stats(notes, scale=parse_key("F minor")[1])
    assert stats["notes"] == 5
    assert stats["range"] == ["F1", "F2"]
    assert stats["velocity"]["longest_same_run"] == 3
    assert stats["notes_per_bar"] == [3, 2]
    assert stats["off_16th_grid"]["count"] == 1
    assert stats["same_pitch_overlaps"] == 1  # the two F2s
    assert stats["outside_key"] == {"count": 1, "pitches": ["F#1"]}


def test_note_stats_empty():
    assert note_stats([], scale=None) == {"notes": 0}
