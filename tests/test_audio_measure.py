"""audio_measure against signals whose answers are known without the code."""

import numpy as np
import pytest

from reaper_mcp import audio_measure as m

RATE = 48000


def tone(hz, seconds=2.0, amplitude=0.5, rate=RATE):
    t = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * hz * t)


def both(left, right=None):
    return np.stack([left, left if right is None else right], axis=1)


def test_stereo_duplicates_mono_and_drops_extra_channels():
    assert m.stereo(np.ones(4)).shape == (4, 2)
    assert m.stereo(np.ones((4, 6))).shape == (4, 2)


def test_db_floors_silence():
    assert m.db(0.0) == m.FLOOR_DB
    assert m.db(1.0) == 0.0


def test_level_summary_of_a_sine():
    data = both(tone(1000, amplitude=0.5))
    level = m.level_summary(data, RATE)
    assert level["peak_db"] == pytest.approx(20 * np.log10(0.5), abs=0.05)
    assert level["rms_db"] == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.05)
    assert level["crest_db"] == pytest.approx(3.0, abs=0.1)
    assert level["clipped_samples"] == 0
    # A full-scale 1 kHz sine reads -3.01 LUFS per channel, so 0 LUFS on both;
    # at amplitude 0.5 it is 6.02 dB lower.
    assert level["lufs"] == pytest.approx(-6.02, abs=0.3)


def test_true_peak_catches_intersample_peaks():
    # A sine at fs/4 sampled 45 degrees off its crest never shows its peak in the samples.
    n = np.arange(4800)
    x = np.sin(2 * np.pi * n / 4 + np.pi / 4)
    data = both(x)
    assert m.amp_db(float(np.max(np.abs(data)))) == pytest.approx(-3.01, abs=0.05)
    assert m.true_peak_db(data) == pytest.approx(0.0, abs=0.3)


def test_level_over_time_follows_a_step():
    quiet, loud = tone(1000, 1.0, 0.05), tone(1000, 1.0, 0.5)
    levels = m.level_over_time(both(np.concatenate([quiet, loud])), RATE, 1.0)
    assert len(levels) == 2
    assert levels[1] - levels[0] == pytest.approx(20.0, abs=0.5)


def test_spectrum_puts_a_tone_in_its_octave():
    spectrum = m.spectrum_summary(both(tone(250)), RATE)
    bands = dict(zip(spectrum["bands_hz"], spectrum["band_db"], strict=True))
    assert max(bands, key=bands.get) == 250
    assert bands[250] - bands[1000] > 40
    assert spectrum["centroid_hz"] == pytest.approx(250, abs=10)


def test_third_octave_resolution_has_more_bands():
    spectrum = m.spectrum_summary(both(tone(1000)), RATE, "third")
    assert len(spectrum["bands_hz"]) > 20
    bands = dict(zip(spectrum["bands_hz"], spectrum["band_db"], strict=True))
    assert max(bands, key=bands.get) == 1000.0


def test_noise_is_flatter_than_a_tone():
    noise = np.random.default_rng(0).standard_normal(RATE) * 0.1
    assert m.spectrum_summary(both(noise), RATE)["flatness"] > 0.5
    assert m.spectrum_summary(both(tone(440)), RATE)["flatness"] < 0.01


def test_stereo_of_mono_content():
    result = m.stereo_summary(both(tone(80) + tone(1000)), RATE)
    assert result["correlation"] == pytest.approx(1.0)
    assert result["width_db"] == m.FLOOR_DB
    assert result["mono_loss_db"] == pytest.approx(0.0)
    assert result["bands"]["sub"]["correlation"] == pytest.approx(1.0)


def test_stereo_of_opposite_polarity_cancels_in_mono():
    x = tone(1000)
    result = m.stereo_summary(both(x, -x), RATE)
    assert result["correlation"] == pytest.approx(-1.0)
    assert result["mono_loss_db"] < -60


def test_stereo_of_uncorrelated_noise_is_wide():
    rng = np.random.default_rng(1)
    data = both(rng.standard_normal(RATE) * 0.1, rng.standard_normal(RATE) * 0.1)
    result = m.stereo_summary(data, RATE)
    assert abs(result["correlation"]) < 0.05
    assert result["width_db"] == pytest.approx(0.0, abs=0.5)
    assert result["mono_loss_db"] == pytest.approx(-3.0, abs=0.3)


def test_stereo_skips_empty_bands():
    result = m.stereo_summary(both(tone(1000)), RATE)
    assert "sub" not in result["bands"]
    assert "mid" in result["bands"]


def test_onsets_of_clicks():
    clicks = np.zeros(RATE * 2)
    for second in (0.25, 0.75, 1.25, 1.75):
        clicks[int(second * RATE):int(second * RATE) + 200] = 0.8
    assert m.onset_summary(both(clicks), RATE)["onsets"] == 4


def test_masking_names_the_louder_track_and_its_range():
    bass = tone(100, amplitude=0.5)
    quiet_bass = tone(110, amplitude=0.05)
    hat = tone(8000, amplitude=0.05)
    result = m.masking({"bass": both(bass), "pad": both(quiet_bass), "hat": both(hat)}, RATE)
    assert result["pad"]["covered_pct"] > 90
    assert result["pad"]["worst"]["track"] == "bass"
    lo, hi = result["pad"]["worst"]["range_hz"]
    assert lo <= 110 <= hi
    assert result["bass"]["covered_pct"] < 5
    assert result["hat"]["covered_pct"] < 5


def test_masking_of_a_silent_track():
    result = m.masking({"a": both(np.zeros(RATE)), "b": both(tone(100))}, RATE)
    assert result["a"] == {"covered_pct": 0.0, "worst": None}
