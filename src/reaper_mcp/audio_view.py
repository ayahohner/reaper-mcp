"""See what the project sounds like: one render, every measurement, a picture.

analyze_audio renders once (the mix, a group of tracks, or one track; the
whole project or a time span) and measures level, level over time, spectrum,
stereo field and onsets from that one file. It also reads any audio file, so
a reference track or a saved earlier render is measured the same way.
analyze_tracks renders each listed track alone and reports how much each is
covered by the others.

The older analyze_* tools each render the whole project again for one number;
these replace that with one render per question.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from reaper_mcp import audio_measure as m
from reaper_mcp import pictures
from reaper_mcp.reaper import RPR, get_project
from reaper_mcp.render_tools import SOLO_IN_PLACE, _render

CUSTOM_SPAN = 0  # RENDER_BOUNDSFLAG: render RENDER_STARTPOS..RENDER_ENDPOS
SECTIONS = ("level", "time", "spectrum", "stereo", "rhythm", "key")
DEFAULT_SECTIONS = SECTIONS[:-1]  # key costs a few seconds; ask for it
RATE = 48000


def _bars(start: float, end: float) -> list[tuple[float, int]]:
    """(project time, bar number) for every bar starting in [start, end)."""
    bars = []
    for measure in range(10_000):
        result = RPR.TimeMap_GetMeasureInfo(0, measure, 0.0, 0.0, 0, 0, 0.0)
        time = float(result[0]) if isinstance(result, (list, tuple)) else float(result)
        if time >= end - 1e-9:
            break
        if time >= start:
            bars.append((time, measure + 1))
    return bars


def _bar_seconds() -> float:
    """Length of the first bar at the project's tempo, seconds."""
    first = RPR.TimeMap_GetMeasureInfo(0, 0, 0.0, 0.0, 0, 0, 0.0)
    second = RPR.TimeMap_GetMeasureInfo(0, 1, 0.0, 0.0, 0, 0, 0.0)
    pick = (lambda r: float(r[0])) if isinstance(first, (list, tuple)) else float
    return pick(second) - pick(first)


def render_span(tracks: list[int] | None, start: float | None, end: float | None, *,
                rate: int = RATE) -> tuple[Path, float, float]:
    """Render the mix, or only `tracks` soloed in place, over [start, end].

    Returns (temporary file, start, end). Solo states are restored afterwards.
    """
    project = get_project()
    length = project.length
    if length <= 0:
        raise RuntimeError("the project is empty; there is nothing to render")
    start = 0.0 if start is None else max(0.0, float(start))
    end = length if end is None else float(end)
    if not start < end:
        raise ValueError(f"need start < end; got start={start}, end={end}")
    all_tracks = project.tracks
    if tracks:
        for index in tracks:
            if not 0 <= index < len(all_tracks):
                raise IndexError(f"no track {index}; the project has {len(all_tracks)}")
    solos = [track.get_info_value("I_SOLO") for track in all_tracks] if tracks else []
    target = Path(tempfile.gettempdir()) / f"reaper-mcp-{uuid.uuid4().hex}.wav"
    try:
        if tracks:
            for i, track in enumerate(all_tracks):
                track.set_info_value("I_SOLO", SOLO_IN_PLACE if i in tracks else 0)
        RPR.GetSetProjectInfo(0, "RENDER_STARTPOS", start, True)
        RPR.GetSetProjectInfo(0, "RENDER_ENDPOS", end, True)
        _render(target, format="wav", sample_rate=rate, bit_depth=24, channels=2,
                bounds=CUSTOM_SPAN)
    finally:
        for track, solo in zip(all_tracks, solos, strict=False):
            track.set_info_value("I_SOLO", solo)
    if not target.is_file():
        raise RuntimeError(f"REAPER did not write {target}")
    return target, start, end


AUDIO_SUFFIXES = (".wav", ".aif", ".aiff", ".flac", ".caf", ".mp3", ".ogg")


def _read(path: Path) -> tuple[np.ndarray, int]:
    """Any audio file as (frames, 2) float. Formats libsndfile cannot decode
    (the AAC inside Apple Loops .caf files) go through macOS afconvert."""
    import soundfile as sf

    try:
        data, rate = sf.read(str(path), always_2d=True)
    except sf.LibsndfileError:
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "decoded.wav"
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEF32", str(path), str(wav)],
                           check=True, capture_output=True)
            data, rate = sf.read(str(wav), always_2d=True)
    return m.stereo(data), int(rate)


def _measure(data: np.ndarray, rate: int, include: tuple[str, ...], *, resolution: str,
             segment: float, segment_label: str) -> dict:
    result: dict = {}
    if "level" in include:
        result["level"] = m.level_summary(data, rate)
    if "time" in include:
        result["level_over_time"] = {"per": segment_label,
                                     "lufs": m.level_over_time(data, rate, segment)}
    if "spectrum" in include:
        result["spectrum"] = m.spectrum_summary(data, rate, resolution)
    if "stereo" in include:
        result["stereo"] = m.stereo_summary(data, rate)
    if "rhythm" in include:
        result["rhythm"] = {**m.onset_summary(data, rate), "tempo_bpm": m.tempo_bpm(data, rate)}
    if "key" in include:
        result["key"] = m.key_summary(data, rate)
    return result


def analyze_audio(
    *,
    tracks: list[int] | None = None,
    span: list[float] | None = None,
    file: str | None = None,
    include: list[str] | None = None,
    resolution: str = "octave",
    picture: bool = False,
    keep_render: str | None = None,
) -> dict:
    """
    Render once and measure it: the mix, or only `tracks` (soloed in place, so
    their sends to FX tracks are heard), over span = [start, end] seconds (default: whole project).
    file: measure an audio file instead (a reference, or a render kept earlier).
    include: any of level, time, spectrum, stereo, rhythm, key (default all but key).
      level: LUFS, true peak, peak, RMS, crest, clipped samples, DC.
      time: LUFS per bar (for files, per 2 s or longer so there are at most 64).
      spectrum: dB per octave band (resolution="third" for 1/3 octaves), centroid, rolloff, flatness.
      stereo: correlation, side/mid width dB, mono-sum loss dB, per band sub/low/mid/high/air.
      rhythm: onset count and rate, estimated tempo.
      key: most likely key and runner-up with correlation, pitch-class energy in % (C first).
    picture: also write a PNG (level + log spectrogram, bar lines) and return its path; Read it to look.
    keep_render: save the rendered WAV here, to compare against later with file=.
    """
    start, end = (span[0], span[1]) if span else (None, None)
    include_set = tuple(include) if include else DEFAULT_SECTIONS
    unknown = sorted(set(include_set) - set(SECTIONS))
    if unknown:
        return {"success": False, "error": f"unknown include {unknown}; use {list(SECTIONS)}"}
    if resolution not in ("octave", "third"):
        return {"success": False, "error": "resolution is 'octave' or 'third'"}
    rendered: Path | None = None
    if file:
        path = Path(file).expanduser()
        if not path.is_file():
            return {"success": False, "error": f"no file at {path}"}
        data, rate = _read(path)
        offset = 0.0
        if start is not None or end is not None:
            first = int((start or 0.0) * rate)
            last = len(data) if end is None else int(end * rate)
            data = data[first:last]
            offset = start or 0.0
        source = {"file": str(path)}
        segment = max(2.0, float(np.ceil(len(data) / rate / 64)))
        label, bars = f"{segment:g} s", None
    else:
        rendered, offset, stop = render_span(tracks, start, end)
        data, rate = _read(rendered)
        source = {"tracks": tracks or "mix", "span_s": [round(offset, 3), round(stop, 3)]}
        segment, label = _bar_seconds(), "bar"
        bars = _bars(offset, stop)
        if bars:
            source["bars"] = [bars[0][1], bars[-1][1]]
    try:
        if len(data) == 0:
            return {"success": False, "error": "nothing to measure: the span is empty"}
        result = {"success": True, **source, "seconds": round(len(data) / rate, 2)}
        result.update(_measure(data, rate, include_set, resolution=resolution,
                               segment=segment, segment_label=label))
        if picture:
            title = f"{source.get('file') or ('tracks ' + str(tracks) if tracks else 'mix')}"
            image = pictures.spectrogram(data, rate, pictures.default_path("spectrogram"),
                                         title=title, start=offset, bars=bars)
            result["picture"] = str(image)
        if keep_render and rendered is not None:
            kept = Path(keep_render).expanduser().resolve().with_suffix(".wav")
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rendered, kept)
            result["kept_render"] = str(kept)
        return result
    finally:
        if rendered is not None:
            rendered.unlink(missing_ok=True)


def analyze_tracks(
    *,
    tracks: list[int],
    start: float | None = None,
    end: float | None = None,
    picture: str | None = None,
) -> dict:
    """
    Render each listed track alone over start..end and compare them: per track LUFS,
    peak, spectral centroid and octave-band levels (dB, shared bands_hz header), plus
    masking: covered_pct = share of the track's energy where another listed track is
    louder (per 1/3 octave, ~43 ms), and worst = the track covering most and where.
    picture: "stack" writes a PNG with a row per track (spectrogram on one shared
    colour scale, level line, bar lines): the arrangement at once. "spectra"
    overlays each track's average spectrum in its own colour: where each sits,
    and where they overlap, for EQ.
    """
    if len(tracks) < 2:
        return {"success": False, "error": "list at least two tracks to compare"}
    project = get_project()
    names = {}
    stems = {}
    rate = RATE
    span = None
    for index in tracks:
        path, s, e = render_span([index], start, end)
        try:
            data, rate = _read(path)
        finally:
            path.unlink(missing_ok=True)
        name = project.tracks[index].name or f"track {index}"
        key = f"{index}: {name}"
        names[index] = key
        stems[key] = data
        span = (s, e)
    per_track = {}
    bands_hz = None
    for key, data in stems.items():
        spectrum = m.spectrum_summary(data, rate)
        bands_hz = spectrum["bands_hz"]
        level = m.level_summary(data, rate)
        per_track[key] = {"lufs": level["lufs"], "peak_db": level["peak_db"],
                          "centroid_hz": spectrum["centroid_hz"], "band_db": spectrum["band_db"]}
    covered = m.masking(stems, rate)
    for key, cover in covered.items():
        per_track[key].update(cover)
    result = {"success": True, "span_s": [round(span[0], 3), round(span[1], 3)],
              "bands_hz": bands_hz, "tracks": per_track}
    if picture == "stack":
        image = pictures.track_stack(stems, rate, pictures.default_path("tracks"),
                                     title=f"tracks {tracks}", start=span[0],
                                     bars=_bars(span[0], span[1]))
        result["picture"] = str(image)
    elif picture == "spectra":
        image = pictures.track_spectra(stems, rate, pictures.default_path("spectra"),
                                       title=f"tracks {tracks}, {span[0]:.1f}-{span[1]:.1f} s")
        result["picture"] = str(image)
    elif picture is not None:
        result["picture_error"] = "picture is 'stack' or 'spectra'"
    return result


MAX_FADER_DB = 12.0


def level_tracks(*, targets: dict[str, float], start: float | None = None,
                 end: float | None = None) -> dict:
    """
    Gain staging: set each track's fader so the track, soloed in place over
    start..end (its sends and ducking included, master FX bypassed), measures
    its target integrated LUFS. targets: {track index: LUFS}, e.g. a kick at -20
    and the rest placed around it, so the mix sums quietly and the master has headroom.
    Returns per track [measured before, fader dB after]; faders stop at +12 dB.
    """
    project = get_project()
    master_fx = project.master_track.fxs
    enabled = [fx.is_enabled for fx in master_fx]
    result: dict = {}
    try:
        for fx in master_fx:
            fx.disable()
        for key, target in targets.items():
            index = int(key)
            track = project.tracks[index]
            path, _, _ = render_span([index], start, end)
            try:
                data, rate = _read(path)
            finally:
                path.unlink(missing_ok=True)
            lufs = m.level_summary(data, rate)["lufs"]
            if lufs is None or lufs < -70:
                result[index] = "silent in this span"
                continue
            volume = track.get_info_value("D_VOL") * 10 ** ((target - lufs) / 20)
            volume = min(volume, 10 ** (MAX_FADER_DB / 20))
            track.set_info_value("D_VOL", volume)
            result[index] = [lufs, round(20 * np.log10(max(volume, 1e-9)), 1)]
    finally:
        for fx, was in zip(master_fx, enabled, strict=True):
            if was:
                fx.enable()
    return {"success": True, "columns": ["lufs_before", "fader_db"], "tracks": result}


SAMPLE_COLUMNS = ("file", "seconds", "peak_db", "attack_ms", "tail_s", "rms_300ms_db",
                  "centroid_hz", "pitch", "width_db")


def audition_samples(*, folder: str, match: str | None = None, limit: int = 40) -> dict:
    """
    Measure every audio file in a folder (not its subfolders) to pick
    one-shots without listening: one row per file, columns named once.
    match: keep files whose name contains this, case-insensitively ("kick").
    tail_s: seconds from the peak until 30 dB under it. pitch: the strongest
    peak between 30 and 500 Hz in the first 300 ms after the hit, as the
    nearest note and cents ("F1+12"); tune a kick or 808 by it.
    centroid_hz: brightness. width_db: side vs mid, -120 for mono.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        return {"success": False, "error": f"not a folder: {folder}"}
    files = sorted(p for p in root.iterdir()
                   if p.suffix.lower() in AUDIO_SUFFIXES
                   and (match is None or match.lower() in p.name.lower()))
    rows = []
    for path in files[:limit]:
        try:
            data, rate = _read(path)
        except Exception as e:  # one unreadable file should not hide the rest
            rows.append([path.name, f"unreadable: {e}"])
            continue
        s = m.sample_summary(data, rate)
        rows.append([path.name, s["seconds"], s["peak_db"], s.get("attack_ms"), s.get("tail_s"),
                     s.get("rms_300ms_db"), s.get("centroid_hz"),
                     m.pitch_note(s.get("pitch_hz", 0.0)), s.get("width_db")])
    result = {"success": True, "folder": str(root), "columns": list(SAMPLE_COLUMNS), "rows": rows}
    if len(files) > limit:
        result["not_shown"] = len(files) - limit
    return result


TOOLS = (analyze_audio, analyze_tracks, level_tracks, audition_samples)
