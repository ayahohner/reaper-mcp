"""Pictures for a model to read: spectrograms and piano rolls as PNG files.

Each picture is drawn for reading, not decoration: labelled axes, bar lines
numbered from the project's bar 1, frequency gridlines at round numbers, and
a fixed dB scale so two renders of the same part can be compared side by side.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import numpy as np

DB_RANGE = 90  # spectrogram colour scale: the loudest cell down to 90 dB below it
FREQ_TICKS = (30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _pyplot():
    os.environ.setdefault("MPLBACKEND", "Agg")  # no window, no GUI toolkit
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def default_path(kind: str) -> Path:
    return Path(tempfile.gettempdir()) / "reaper-mcp" / f"{kind}-{uuid.uuid4().hex[:8]}.png"


def _freq_label(hz: float) -> str:
    return f"{hz / 1000:g}k" if hz >= 1000 else f"{hz:g}"


def _bar_lines(ax, *, start: float, end: float, bars: list[tuple[float, int]] | None) -> None:
    """Vertical lines at bar starts, labelled with bar numbers (every 4th when crowded)."""
    if not bars:
        return
    inside = [(t, n) for t, n in bars if start <= t <= end]
    every = 1 if len(inside) <= 24 else 4 if len(inside) <= 96 else 8
    for t, n in inside:
        major = (n - 1) % 4 == 0
        ax.axvline(t, color="white" if major else "#bbbbbb", lw=0.6 if major else 0.3, alpha=0.5)
        if (n - 1) % every == 0:
            ax.text(t, 1.0, str(n), transform=ax.get_xaxis_transform(), fontsize=6,
                    color="#333333", ha="left", va="bottom")


def spectrogram(data: np.ndarray, rate: int, path: Path, *, title: str,
                start: float = 0.0, bars: list[tuple[float, int]] | None = None) -> Path:
    """Level over time (top) and a log-frequency spectrogram (bottom) of a stereo render.

    start: project time of the first sample, so the time axis and bar lines
    read in project seconds.
    """
    plt = _pyplot()
    mono = data.mean(axis=1)
    seconds = len(mono) / rate
    times, freqs, power_db = _constant_q(mono, rate)
    top = float(power_db.max())
    times = times + start

    # Short-term level per channel on a 50 ms grid, dBFS RMS.
    step = max(1, int(0.05 * rate))
    n = len(mono) // step
    level = data[: n * step].reshape(n, step, 2)
    rms = 10 * np.log10(np.mean(level ** 2, axis=1) + 1e-12)
    level_t = start + (np.arange(n) + 0.5) * step / rate

    fig, (ax_level, ax_spec) = plt.subplots(
        2, 1, figsize=(12, 6.2), dpi=110, sharex=True,
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.06})
    ax_level.plot(level_t, rms[:, 0], lw=0.7, color="#1f77b4", label="L")
    ax_level.plot(level_t, rms[:, 1], lw=0.7, color="#d62728", label="R", alpha=0.8)
    ax_level.set_ylim(-60, 0)
    ax_level.set_yticks([-48, -24, -12, -6, 0])
    ax_level.set_ylabel("dBFS", fontsize=7)
    ax_level.grid(axis="y", lw=0.3, alpha=0.5)
    ax_level.legend(fontsize=6, loc="lower right", ncols=2)
    ax_level.tick_params(labelsize=7)
    ax_level.set_title(title, fontsize=9, loc="left")

    mesh = ax_spec.pcolormesh(times, freqs, power_db, shading="auto",
                              cmap="magma", vmin=top - DB_RANGE, vmax=top, rasterized=True)
    ax_spec.set_yscale("log")
    ax_spec.set_ylim(20, min(20000, rate / 2))
    ticks = [f for f in FREQ_TICKS if f <= rate / 2]
    ax_spec.set_yticks(ticks, [_freq_label(f) for f in ticks])
    ax_spec.set_ylabel("Hz", fontsize=7)
    ax_spec.set_xlabel("project time, s (numbers above: bars)", fontsize=7)
    ax_spec.tick_params(labelsize=7)
    for f in (100, 1000, 10000):
        ax_spec.axhline(f, color="#ffffff", lw=0.3, alpha=0.35)
    _bar_lines(ax_level, start=start, end=start + seconds, bars=bars)
    for t, _ in bars or []:
        if start <= t <= start + seconds:
            ax_spec.axvline(t, color="white", lw=0.3, alpha=0.3)
    ax_spec.set_xlim(start, start + seconds)
    bar = fig.colorbar(mesh, ax=[ax_level, ax_spec], pad=0.01, fraction=0.02)
    bar.set_label(f"dB (0 = loudest cell, {top:.0f} dB re 1)", fontsize=6)
    bar.ax.tick_params(labelsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _constant_q(mono: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-Q power in dB, 24 bins per octave from 20 Hz: a short window up
    high and a long one down low, so 44 Hz bass harmonics separate and hats
    stay sharp in time. Returns (frame times, bin frequencies, dB[bin, frame])."""
    import librosa

    fmin, per_octave = 20.0, 24
    n_bins = int(np.log2(min(20000, 0.45 * rate) / fmin) * per_octave)
    octaves = -(-n_bins // per_octave)
    hop = 2 ** (octaves - 1)  # librosa halves the rate once per octave
    hop *= max(1, round(len(mono) / hop / 4000))  # at most ~4000 frames
    spectrum = np.abs(librosa.cqt(mono.astype(np.float32), sr=rate, hop_length=hop, fmin=fmin,
                                  n_bins=n_bins, bins_per_octave=per_octave))
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=per_octave)
    times = librosa.frames_to_time(np.arange(spectrum.shape[1]), sr=rate, hop_length=hop)
    return times, freqs, 20 * np.log10(spectrum + 1e-10)


def note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def piano_roll(parts: dict[str, list[tuple[float, float, int, int]]], path: Path, *,
               title: str, beats_per_bar: int = 4,
               scale: set[int] | None = None) -> Path:
    """One lane per part, notes as bars on a beat axis.

    parts: name -> [(start_beat, length_beats, pitch, velocity)].
    Notes outside scale (pitch classes) are outlined in red; velocity sets opacity.
    """
    plt = _pyplot()
    from matplotlib.patches import Rectangle

    lanes = [(name, notes) for name, notes in parts.items() if notes]
    if not lanes:
        raise ValueError("no notes to draw")
    end = max(s + d for _, notes in lanes for s, d, _, _ in notes)
    heights = [max(4, max(p for _, _, p, _ in n) - min(p for _, _, p, _ in n) + 3) for _, n in lanes]
    fig, axes = plt.subplots(len(lanes), 1, figsize=(12, min(2 + 0.09 * sum(heights), 14)),
                             dpi=110, sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.15})
    colours = plt.get_cmap("tab10")
    for row, ((name, notes), ax) in enumerate(zip(lanes, axes[:, 0], strict=True)):
        low = min(p for _, _, p, _ in notes) - 1
        high = max(p for _, _, p, _ in notes) + 1
        for s, d, p, v in notes:
            outside = scale is not None and p % 12 not in scale
            ax.add_patch(Rectangle((s, p - 0.4), d, 0.8, facecolor=colours(row % 10),
                                   alpha=0.25 + 0.75 * v / 127,
                                   edgecolor="red" if outside else "none", lw=1.2))
        ax.set_ylim(low - 0.5, high + 0.5)
        ticks = [p for p in range(low, high + 1) if p % 12 in (0, 5, 7)] or [low, high]
        ax.set_yticks(ticks, [note_name(p) for p in ticks])
        ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
        for bar_beat in np.arange(0, end + beats_per_bar, beats_per_bar):
            ax.axvline(bar_beat, color="#888888", lw=0.5, alpha=0.6)
        ax.grid(axis="y", lw=0.2, alpha=0.4)
    axes[0, 0].set_title(title, fontsize=9, loc="left")
    axes[-1, 0].set_xlim(0, end)
    bar_ticks = np.arange(0, end + 1e-9, beats_per_bar * (1 if end <= 32 * beats_per_bar else 4))
    axes[-1, 0].set_xticks(bar_ticks, [str(int(b // beats_per_bar) + 1) for b in bar_ticks])
    axes[-1, 0].set_xlabel("bar", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
