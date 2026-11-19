"""Root-note parsing and zone layout for load_multisample, without REAPER."""

import pytest

from reaper_mcp.sampler_tools import root_of, zones


@pytest.mark.parametrize(("name", "middle_c", "root"), [
    ("Super_48.aif", "C3", 48),
    ("MinDnc24.aif", "C3", 24),
    ("FormantPad60.wav", "C3", 60),
    ("Dark Background C3.wav", "C3", 60),
    ("Formant Atmo World F#1.wav", "C3", 42),
    ("Ether Vent C2-SF01.wav", "C3", 48),
    ("Piano C4.wav", "C4", 60),
    ("Bass Eb1.wav", "C4", 27),
    ("Pad A-1.wav", "C3", 21),
])
def test_root_of(name, middle_c, root):
    assert root_of(name, middle_c=middle_c) == root


def test_root_of_without_a_root():
    assert root_of("Noise.wav", middle_c="C3") is None
    assert root_of("Take 12 of 48.wav", middle_c="C3") is None  # two numbers: ambiguous


def test_zones_split_halfway_and_cover_the_keyboard():
    assert zones([36, 39, 42]) == [(0, 37), (38, 40), (41, 127)]
    assert zones([60]) == [(0, 127)]
