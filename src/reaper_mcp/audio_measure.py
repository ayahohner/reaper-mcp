"""Measurements of rendered audio, as pure functions of sample arrays.

Nothing here talks to REAPER, so every number can be checked against a
synthetic signal. Arrays are float, shape (frames, 2): left and right.
Levels are dB relative to digital full scale unless a name says otherwise.

Kept compact on purpose: each result goes back to the model as text, so
lists of numbers share one header (the band centres) instead of repeating
keys, and everything is rounded to what a mix decision can use.
"""

from __future__ import annotations

import numpy as np

FLOOR_DB = -120.0

# Octave bands by centre frequency; each spans centre/sqrt(2) to centre*sqrt(2).
OCTAVE_CENTRES = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)
THIRD_CENTRES = tuple(round(1000 * 2 ** (n / 3), 1) for n in range(-15, 14))  # 31.5 Hz .. 20 kHz

# Stereo bands: the low end should be centred, the top can be wide.
STEREO_BANDS = {"sub": (20, 120), "low": (120, 500), "mid": (500, 2000),
                "high": (2000, 8000), "air": (8000, 20000)}


def db(power: float) -> float:
    """Power ratio to dB, floored so silence is a number, not -inf."""
    return float(max(10 * np.log10(power) if power > 0 else FLOOR_DB, FLOOR_DB))


def amp_db(amplitude: float) -> float:
    return db(amplitude * amplitude)


def stereo(data: np.ndarray) -> np.ndarray:
    """Any render as (frames, 2): mono is duplicated, extra channels dropped."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2]


# Level


def loudness(data: np.ndarray, rate: int) -> float:
    """Integrated loudness, LUFS (ITU-R BS.1770 via pyloudnorm); FLOOR_DB when
    too short (under 0.4 s) or silent."""
    import pyloudnorm

    if len(data) < int(0.4 * rate) or not np.any(data):
        return FLOOR_DB
    value = pyloudnorm.Meter(rate).integrated_loudness(data)
    return float(value) if np.isfinite(value) else FLOOR_DB


def true_peak_db(data: np.ndarray) -> float:
    """Peak after 4x oversampling (BS.1770 annex 2), in dBTP."""
    from scipy.signal import resample_poly

    if not np.any(data):
        return FLOOR_DB
    upsampled = resample_poly(data, 4, 1, axis=0)
    return amp_db(float(np.max(np.abs(upsampled))))


def level_over_time(data: np.ndarray, rate: int, segment_seconds: float) -> list[float]:
    """Loudness of each segment, LUFS; segments shorter than 0.4 s are measured
    as RMS in dB instead, which BS.1770 gating cannot measure."""
    import pyloudnorm

    step = max(1, int(round(segment_seconds * rate)))
    meter = pyloudnorm.Meter(rate) if step >= int(0.4 * rate) else None
    levels = []
    for start in range(0, len(data), step):
        chunk = data[start:start + step]
        if len(chunk) < step // 2:
            break
        if meter is not None and len(chunk) >= int(0.4 * rate) and np.any(chunk):
            value = meter.integrated_loudness(chunk)
            levels.append(round(float(value), 1) if np.isfinite(value) else FLOOR_DB)
        else:
            levels.append(round(db(float(np.mean(chunk ** 2))), 1))
    return levels


def level_summary(data: np.ndarray, rate: int) -> dict:
    mono = data.mean(axis=1)
    rms = db(float(np.mean(data ** 2)))
    peak = amp_db(float(np.max(np.abs(data)))) if np.any(data) else FLOOR_DB
    return {
        "lufs": round(loudness(data, rate), 1),
        "true_peak_dbtp": round(true_peak_db(data), 1),
        "peak_db": round(peak, 1),
        "rms_db": round(rms, 1),
        "crest_db": round(peak - rms, 1),
        "clipped_samples": int(np.sum(np.abs(data) >= 0.9999)),
        "dc_offset": round(float(np.mean(mono)), 5),
    }


# Spectrum


def power_spectrum(mono: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch average power spectrum: (frequencies, power per bin)."""
    from scipy.signal import welch

    nperseg = min(8192, max(256, len(mono)))
    freqs, power = welch(mono, fs=rate, nperseg=nperseg, scaling="spectrum")
    return freqs, power


def band_levels(freqs: np.ndarray, power: np.ndarray, centres, *, width_octaves: float) -> list[float]:
    """Total power in each band around its centre, dB."""
    half = 2 ** (width_octaves / 2)
    return [round(db(float(power[(freqs >= c / half) & (freqs < c * half)].sum())), 1)
            for c in centres]


def spectrum_summary(data: np.ndarray, rate: int, resolution: str = "octave") -> dict:
    mono = data.mean(axis=1)
    freqs, power = power_spectrum(mono, rate)
    centres, width = (THIRD_CENTRES, 1 / 3) if resolution == "third" else (OCTAVE_CENTRES, 1.0)
    centres = [c for c in centres if c * 2 ** (width / 2) <= rate / 2 + 1]
    total = float(power.sum())
    if total <= 0:
        return {"bands_hz": centres, "band_db": [FLOOR_DB] * len(centres),
                "centroid_hz": 0, "rolloff85_hz": 0, "flatness": 0.0}
    centroid = float((freqs * power).sum() / total)
    cumulative = np.cumsum(power) / total
    rolloff = float(freqs[min(int(np.searchsorted(cumulative, 0.85)), len(freqs) - 1)])
    audible = power[(freqs >= 20) & (freqs <= 20000)] + 1e-20
    flatness = float(np.exp(np.mean(np.log(audible))) / np.mean(audible))
    return {
        "bands_hz": centres,
        "band_db": band_levels(freqs, power, centres, width_octaves=width),
        "centroid_hz": int(round(centroid)),
        "rolloff85_hz": int(round(rolloff)),
        "flatness": round(flatness, 3),
    }


# Stereo


def _bandpass(data: np.ndarray, rate: int, *, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyquist = rate / 2
    hi = min(hi, nyquist * 0.99)
    if lo >= hi:
        return np.zeros_like(data)
    sos = butter(4, [lo / nyquist, hi / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=0)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)))
    return None if denominator <= 1e-12 else float(np.sum(left * right) / denominator)


def _width_db(left: np.ndarray, right: np.ndarray) -> float:
    """Side energy relative to mid energy, dB: -inf for mono, 0 for one-sided or
    uncorrelated, positive when out of phase."""
    mid = float(np.mean(((left + right) / 2) ** 2))
    side = float(np.mean(((left - right) / 2) ** 2))
    if side <= 1e-12 * max(mid, 1e-30):
        return FLOOR_DB  # mono: no side signal worth a number
    return round(db(side) - db(mid), 1)


def mono_loss_db(data: np.ndarray) -> float:
    """Level change when summed to mono, dB: 0 for mono content, about -3 for
    uncorrelated sides, very negative when the channels cancel."""
    left, right = data[:, 0], data[:, 1]
    stereo_power = float(np.mean(left ** 2 + right ** 2))
    if stereo_power <= 0:
        return 0.0
    return round(db(float(np.mean((left + right) ** 2)) / (2 * stereo_power)), 1)


def stereo_summary(data: np.ndarray, rate: int) -> dict:
    left, right = data[:, 0], data[:, 1]
    overall = _correlation(left, right)
    total = db(float(np.mean(data ** 2)))
    bands = {}
    for name, (lo, hi) in STEREO_BANDS.items():
        if lo >= rate / 2:
            continue
        part = _bandpass(data, rate, lo=lo, hi=hi)
        if db(float(np.mean(part ** 2))) < total - 30:
            continue  # 30 dB under the whole signal: nothing here to judge
        correlation = _correlation(part[:, 0], part[:, 1])
        bands[name] = {"width_db": _width_db(part[:, 0], part[:, 1]),
                       "correlation": None if correlation is None else round(correlation, 2)}
    return {
        "correlation": None if overall is None else round(overall, 2),
        "width_db": _width_db(left, right),
        "mono_loss_db": mono_loss_db(data),
        "bands": bands,
    }


# Rhythm


def onset_summary(data: np.ndarray, rate: int) -> dict:
    import librosa

    mono = data.mean(axis=1).astype(np.float32)
    seconds = len(mono) / rate
    if not np.any(mono) or seconds <= 0:
        return {"onsets": 0, "onsets_per_second": 0.0}
    onsets = librosa.onset.onset_detect(y=mono, sr=rate, units="time")
    return {"onsets": int(len(onsets)), "onsets_per_second": round(len(onsets) / seconds, 2)}


# Several tracks against each other


def _band_frames(mono: np.ndarray, rate: int, centres, *, hop: int = 2048) -> np.ndarray:
    """Energy per (third-octave band, frame), linear power."""
    n_fft = 4096
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    frames = 1 + (len(mono) - n_fft) // hop
    window = np.hanning(n_fft)
    index = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spectra = np.abs(np.fft.rfft(mono[index] * window, axis=1)) ** 2  # (frames, bins)
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    half = 2 ** (1 / 6)
    bands = [spectra[:, (freqs >= c / half) & (freqs < c * half)].sum(axis=1) for c in centres]
    return np.array(bands)  # (bands, frames)


def masking(stems: dict[str, np.ndarray], rate: int) -> dict[str, dict]:
    """How much of each track is covered by louder tracks.

    Each track's energy is split into third-octave bands per ~43 ms frame.
    A cell of track A is covered when another track is louder there; covered_pct
    is the share of A's energy in covered cells, and worst names the track and
    band range that covers the most of it.
    """
    centres = [c for c in THIRD_CENTRES if c * 2 ** (1 / 6) <= rate / 2]
    energy = {name: _band_frames(stereo(data).mean(axis=1), rate, centres)
              for name, data in stems.items()}
    frames = min(e.shape[1] for e in energy.values())
    energy = {name: e[:, :frames] for name, e in energy.items()}
    result = {}
    for name, own in energy.items():
        total = float(own.sum())
        if total <= 0:
            result[name] = {"covered_pct": 0.0, "worst": None}
            continue
        others = {other: e for other, e in energy.items() if other != name}
        if not others:
            result[name] = {"covered_pct": 0.0, "worst": None}
            continue
        loudest = np.max(np.stack(list(others.values())), axis=0)
        covered = own * (loudest > own)
        worst = None
        best_share = 0.0
        for other, e in others.items():
            by_band = (own * (e > own)).sum(axis=1)  # per band
            share = float(by_band.sum()) / total
            if share > best_share:
                best_share = share
                lo, hi = _main_range(by_band, centres)
                worst = {"track": other, "pct": round(100 * share, 1), "range_hz": [lo, hi]}
        result[name] = {"covered_pct": round(100 * float(covered.sum()) / total, 1), "worst": worst}
    return result


def _main_range(by_band: np.ndarray, centres) -> tuple[int, int]:
    """The contiguous bands holding the central 80% of by_band's energy."""
    total = float(by_band.sum())
    cumulative = np.cumsum(by_band) / total
    lo = int(np.searchsorted(cumulative, 0.1))
    hi = int(np.searchsorted(cumulative, 0.9))
    hi = min(hi, len(centres) - 1)
    return int(round(centres[lo] / 2 ** (1 / 6))), int(round(centres[hi] * 2 ** (1 / 6)))


# One-shot samples


def sample_summary(data: np.ndarray, rate: int) -> dict:
    """What a drum or one-shot sample is like, for picking one from many.

    tail_s: time from the peak until the level stays 30 dB under it (50 ms
    RMS windows). attack_ms: time to reach the peak. pitch_hz: the strongest
    spectral peak between 30 and 500 Hz in the first 300 ms after the peak;
    the body of a kick, the tone of a snare.
    """
    mono = data.mean(axis=1)
    if not np.any(mono):
        return {"seconds": round(len(mono) / rate, 2), "peak_db": FLOOR_DB}
    peak_at = int(np.argmax(np.abs(mono)))
    peak = float(np.abs(mono[peak_at]))
    step = max(1, int(0.05 * rate))
    after = mono[peak_at:]
    n = max(1, len(after) // step)
    rms = np.sqrt(np.mean(after[: n * step].reshape(n, step) ** 2, axis=1)) if len(after) >= step else np.array([peak])
    loud = np.nonzero(rms > peak * 10 ** (-30 / 20))[0]
    tail = (int(loud[-1]) + 1) * step / rate if len(loud) else 0.0
    body = mono[peak_at:peak_at + int(0.3 * rate)]
    pitch = 0.0
    if len(body) >= 1024:
        spectrum = np.abs(np.fft.rfft(body * np.hanning(len(body)), n=max(len(body), 1 << 16)))
        freqs = np.fft.rfftfreq(max(len(body), 1 << 16), 1 / rate)
        band = (freqs >= 30) & (freqs <= 500)
        pitch = float(freqs[band][np.argmax(spectrum[band])])
    spectrum = spectrum_summary(data, rate)
    return {
        "seconds": round(len(mono) / rate, 2),
        "peak_db": round(amp_db(peak), 1),
        "attack_ms": round(1000 * peak_at / rate, 1),
        "tail_s": round(tail, 2),
        "rms_300ms_db": round(db(float(np.mean(body ** 2))), 1),
        "centroid_hz": spectrum["centroid_hz"],
        "pitch_hz": round(pitch, 1),
        "width_db": _width_db(data[:, 0], data[:, 1]),
    }


def pitch_note(hz: float) -> str:
    """The nearest note to a frequency and the offset in cents, e.g. 'F1+12'."""
    if hz <= 0:
        return ""
    midi = 69 + 12 * np.log2(hz / 440)
    nearest = int(round(midi))
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    cents = int(round(100 * (midi - nearest)))
    return f"{names[nearest % 12]}{nearest // 12 - 1}{cents:+d}"


# Tempo and key

# Krumhansl-Kessler key profiles, tonic first.
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
PITCH_CLASSES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def _analysis_mono(data: np.ndarray, rate: int, target: int = 22050) -> np.ndarray:
    from scipy.signal import resample_poly

    mono = data.mean(axis=1)
    if rate != target:
        g = np.gcd(rate, target)
        mono = resample_poly(mono, target // g, rate // g)
    return mono.astype(np.float32)


def tempo_bpm(data: np.ndarray, rate: int) -> float:
    """Estimated tempo, BPM (librosa onset autocorrelation)."""
    import librosa

    mono = _analysis_mono(data, rate)
    if not np.any(mono):
        return 0.0
    return round(float(np.atleast_1d(librosa.feature.tempo(y=mono, sr=22050))[0]), 1)


def key_summary(data: np.ndarray, rate: int) -> dict:
    """Most likely key by correlating the average chroma with major and minor
    profiles. chroma: share of pitch-class energy, C first, in percent."""
    import librosa

    mono = _analysis_mono(data, rate)
    if not np.any(mono):
        return {"key": None}
    chroma = librosa.feature.chroma_cqt(y=mono, sr=22050).mean(axis=1)
    scores = []
    for tonic in range(12):
        rolled = np.roll(chroma, -tonic)
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            scores.append((float(np.corrcoef(rolled, profile)[0, 1]), f"{PITCH_CLASSES[tonic]} {mode}"))
    scores.sort(reverse=True)
    return {"key": scores[0][1], "confidence": round(scores[0][0], 2),
            "runner_up": scores[1][1], "runner_up_confidence": round(scores[1][0], 2),
            "chroma_pct": [int(round(100 * c / chroma.sum())) for c in chroma]}
