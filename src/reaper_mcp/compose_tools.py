"""Writing music in beats, in bulk: notes, automation, plugin settings, and a note view.

Everything here is positioned in quarter-note beats from the project start
(beat 0 = bar 1), because music is written in beats and the tempo may change.
Each tool takes a whole part per call: a part as text costs a few tokens per
note, where one call per note costs a few hundred.
"""

from __future__ import annotations

import re
from statistics import mean, pstdev

from reaper_mcp import pictures
from reaper_mcp.reaper import RPR, get_project, is_null

NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")
PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SCALES = {
    "major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10), "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11), "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10), "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10), "chromatic": tuple(range(12)),
}
ENVELOPE_SHAPES = {"linear": 0, "square": 1, "slow": 2, "fast_start": 3, "fast_end": 4, "bezier": 5}


# Parsing


def parse_pitch(text: str) -> int:
    """MIDI number from "60", "C4" (= 60), "F#1", "Bb0"."""
    text = text.strip()
    if re.fullmatch(r"\d{1,3}", text):
        pitch = int(text)
    else:
        match = NOTE_RE.match(text)
        if not match:
            raise ValueError(f"not a pitch: {text!r} (use 0-127 or a name like F#1)")
        letter, accidental, octave = match.groups()
        pitch = PITCH_CLASSES[letter.upper()] + {"#": 1, "b": -1, "": 0}[accidental]
        pitch += (int(octave) + 1) * 12
    if not 0 <= pitch <= 127:
        raise ValueError(f"pitch {text!r} is outside MIDI's 0-127")
    return pitch


def parse_notes(text: str, *, default_length: float = 1.0,
                default_velocity: int = 100) -> list[tuple[float, float, int, int]]:
    """Notes as "PITCH@BEAT[:LENGTH[:VELOCITY]]", separated by spaces, commas or
    newlines; several pitches share one onset with "+": "F1+C2@0:2".
    Returns [(beat, length, pitch, velocity)]."""
    notes = []
    for token in re.split(r"[\s,]+", text.strip()):
        if not token:
            continue
        if "@" not in token:
            raise ValueError(f"note {token!r} needs PITCH@BEAT, e.g. F1@0:0.5:110")
        pitches, timing = token.split("@", 1)
        fields = timing.split(":")
        if len(fields) > 3:
            raise ValueError(f"note {token!r}: at most BEAT:LENGTH:VELOCITY after @")
        beat = float(fields[0])
        length = float(fields[1]) if len(fields) > 1 and fields[1] else default_length
        velocity = int(fields[2]) if len(fields) > 2 and fields[2] else default_velocity
        if beat < 0 or length <= 0 or not 1 <= velocity <= 127:
            raise ValueError(f"note {token!r}: need beat >= 0, length > 0, velocity 1-127")
        notes.extend((beat, length, parse_pitch(p), velocity) for p in pitches.split("+"))
    return notes


def parse_key(key: str) -> tuple[int, set[int]]:
    """"F minor", "C# phrygian" -> (tonic pitch class, scale pitch classes)."""
    parts = key.strip().split()
    if len(parts) != 2:
        raise ValueError(f"key {key!r}: use TONIC MODE, e.g. 'F minor'")
    tonic = parse_pitch(parts[0] + "0") % 12
    mode = parts[1].lower()
    if mode not in SCALES:
        raise ValueError(f"mode {mode!r}: use one of {sorted(SCALES)}")
    return tonic, {(tonic + step) % 12 for step in SCALES[mode]}


# Tools


def write_notes(
    *,
    track_index: int,
    start_beat: float,
    length_beats: float,
    notes: str,
    repeat: int = 1,
    channel: int = 0,
) -> dict:
    """
    Write a part as one MIDI item: starts at start_beat (quarter notes from the
    project start; beat 0 = bar 1), lasts length_beats * repeat.
    notes: "PITCH@BEAT[:LENGTH[:VELOCITY]]" separated by spaces, BEAT relative to the
    item start; PITCH as 0-127 or a name (C4 = 60, F1, Bb0); chords as "F1+C2@0:2".
    Default length 1 beat, velocity 100. repeat: play the pattern that many times,
    every length_beats. Example, a bar of 8ths: "F1@0:0.5:120 F1@0.5:0.5:90 Ab1@1:1".
    """
    parsed = parse_notes(notes)
    if repeat < 1 or length_beats <= 0:
        return {"success": False, "error": "need repeat >= 1 and length_beats > 0"}
    late = [n for n in parsed if n[0] >= length_beats]
    if late:
        return {"success": False, "error": f"{len(late)} notes start at or after length_beats "
                                           f"({length_beats}); first at beat {late[0][0]}"}
    track = get_project().tracks[track_index]
    total = length_beats * repeat
    item = RPR.CreateNewMIDIItemInProj(track.id, start_beat, start_beat + total, True)
    if isinstance(item, (list, tuple)):
        item = item[0]
    take = RPR.GetActiveTake(item)
    for r in range(repeat):
        offset = start_beat + r * length_beats
        for beat, length, pitch, velocity in parsed:
            on = RPR.MIDI_GetPPQPosFromProjQN(take, offset + beat)
            off = RPR.MIDI_GetPPQPosFromProjQN(take, offset + min(beat + length, length_beats))
            RPR.MIDI_InsertNote(take, False, False, on, off, channel, pitch, velocity, True)
    RPR.MIDI_Sort(take)
    pitches = [p for _, _, p, _ in parsed]
    return {
        "success": True,
        "track_index": track_index,
        "item_index": track.n_items - 1,
        "beats": [start_beat, start_beat + total],
        "notes": len(parsed) * repeat,
        "range": [pictures.note_name(min(pitches)), pictures.note_name(max(pitches))] if pitches else [],
    }


def _param_names(track_id, fx_index: int) -> list[str]:
    count = RPR.TrackFX_GetNumParams(track_id, fx_index)
    return [str(RPR.TrackFX_GetParamName(track_id, fx_index, i, "", 256)[4]) for i in range(count)]


def _resolve_param(names: list[str], fx_index: int, param: str | int) -> int:
    """A parameter's index from its index or its name (exact, else a unique substring)."""
    if isinstance(param, int) or (isinstance(param, str) and param.isdigit()):
        index = int(param)
        if not 0 <= index < len(names):
            raise IndexError(f"FX {fx_index} has no parameter {index} (it has {len(names)})")
        return index
    wanted = param.strip().lower()
    exact = [i for i, n in enumerate(names) if n.lower() == wanted]
    if exact:
        return exact[0]
    partial = [i for i, n in enumerate(names) if wanted in n.lower()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise ValueError(f"parameter {param!r} matches several: {[names[i] for i in partial][:8]}")
    raise ValueError(f"FX {fx_index} has no parameter like {param!r}; some: {names[:12]}")


def set_fx(
    *,
    track_index: int,
    fx_index: int,
    params: dict[str, float] | None = None,
    config: dict[str, str] | None = None,
) -> dict:
    """
    Set several parameters of one FX at once, in the plugin's own units (a JSFX
    slider's value, a VST's 0-1). params: {name or index: value}; names match
    case-insensitively, whole or as a unique substring.
    config: named settings a parameter cannot reach, e.g. ReaSamplOmatic5000's
    {"FILE0": "/path/sample.wav", "DONE": ""} to load a sample.
    Returns each parameter's value as the plugin displays it.
    """
    track = get_project().tracks[track_index]
    if not 0 <= fx_index < track.n_fxs:
        return {"success": False, "error": f"track {track_index} has no FX {fx_index}"}
    shown = {}
    for name, value in (config or {}).items():
        if not RPR.TrackFX_SetNamedConfigParm(track.id, fx_index, name, str(value)):
            return {"success": False, "error": f"FX {fx_index} refused config {name!r}"}
    names = _param_names(track.id, fx_index) if params else []
    for name, value in (params or {}).items():
        index = _resolve_param(names, fx_index, name)
        RPR.TrackFX_SetParam(track.id, fx_index, index, float(value))
        label = names[index]
        formatted = RPR.TrackFX_GetFormattedParamValue(track.id, fx_index, index, "", 256)
        shown[label] = str(formatted[4]) if isinstance(formatted, (list, tuple)) else str(formatted)
    result = {"success": True, "track_index": track_index, "fx_index": fx_index, "params": shown}
    if config:
        result["config"] = sorted(config)
    return result


def _envelope(track, target: str, *, fx_index: int | None, param: str | int | None):
    if target in ("volume", "pan"):
        return RPR.GetTrackEnvelopeByName(track.id, target.capitalize())
    if target != "fx" or fx_index is None or param is None:
        raise ValueError("target is volume, pan, or fx (with fx_index and param)")
    index = _resolve_param(_param_names(track.id, fx_index), fx_index, param)
    return RPR.GetFXEnvelope(track.id, fx_index, index, True)


def write_envelope(
    *,
    track_index: int,
    points: str,
    target: str = "fx",
    fx_index: int | None = None,
    param: str | None = None,
    clear: bool = True,
) -> dict:
    """
    Write automation as "BEAT:VALUE[:SHAPE]" points (beats from the project start),
    separated by spaces. target: volume (VALUE in dB), pan (-1 left .. 1 right), or fx
    (fx_index + param name/index; VALUE in the plugin's units, as set_fx takes).
    SHAPE: linear (default), square, slow, fast_start, fast_end, bezier.
    clear: first delete existing points between the first and last beat written.
    """
    track = get_project().tracks[track_index]
    envelope = _envelope(track, target, fx_index=fx_index, param=param)
    if is_null(envelope):
        if target in ("volume", "pan"):
            # Show the track's envelope so it exists, then fetch it again.
            RPR.SetOnlyTrackSelected(track.id)
            RPR.Main_OnCommand(40406 if target == "volume" else 40407, 0)
            envelope = _envelope(track, target, fx_index=fx_index, param=param)
        if is_null(envelope):
            return {"success": False, "error": f"no {target} envelope on track {track_index}"}
    parsed = []
    for token in points.split():
        fields = token.split(":")
        if len(fields) not in (2, 3):
            return {"success": False, "error": f"point {token!r}: use BEAT:VALUE[:SHAPE]"}
        shape = fields[2] if len(fields) == 3 else "linear"
        if shape not in ENVELOPE_SHAPES:
            return {"success": False, "error": f"shape {shape!r}: use {sorted(ENVELOPE_SHAPES)}"}
        parsed.append((float(fields[0]), float(fields[1]), ENVELOPE_SHAPES[shape]))
    if not parsed:
        return {"success": False, "error": "no points"}
    times = [RPR.TimeMap2_QNToTime(0, beat) for beat, _, _ in parsed]
    if clear:
        RPR.DeleteEnvelopePointRange(envelope, min(times) - 1e-6, max(times) + 1e-6)
    mode = RPR.GetEnvelopeScalingMode(envelope)
    for time, (_, value, shape) in zip(times, parsed, strict=True):
        if target == "volume":
            value = RPR.ScaleToEnvelopeMode(mode, 10 ** (value / 20))
        elif target == "pan":
            value = -value  # REAPER's pan envelope runs opposite to track pan
        RPR.InsertEnvelopePoint(envelope, time, value, shape, 0.0, False, True)
    RPR.Envelope_SortPoints(envelope)
    return {"success": True, "track_index": track_index, "target": target,
            "points": len(parsed), "beats": [parsed[0][0], parsed[-1][0]]}


# Note view


def _notes_of(track) -> list[tuple[float, float, int, int]]:
    """All notes on a track as (beat, length, pitch, velocity), beats from project start."""
    notes = []
    for item_index in range(track.n_items):
        item = RPR.GetTrackMediaItem(track.id, item_index)
        take = RPR.GetActiveTake(item)
        if is_null(take) or not RPR.TakeIsMIDI(take):
            continue
        count = RPR.MIDI_CountEvts(take, 0, 0, 0)[2]
        for i in range(count):
            fields = RPR.MIDI_GetNote(take, i, False, False, 0.0, 0.0, 0, 0, 0)
            # (ok, take, index, selected, muted, start, end, channel, pitch, velocity)
            muted, start, end, pitch, velocity = fields[4], fields[5], fields[6], fields[8], fields[9]
            if muted:
                continue
            on = RPR.MIDI_GetProjQNFromPPQPos(take, start)
            off = RPR.MIDI_GetProjQNFromPPQPos(take, end)
            notes.append((round(on, 4), round(off - on, 4), int(pitch), int(velocity)))
    return sorted(notes)


def note_stats(notes: list[tuple[float, float, int, int]], *, scale: set[int] | None,
               beats_per_bar: int = 4) -> dict:
    """Numbers a part is checked by: range, key, velocity spread, grid, density, overlaps."""
    if not notes:
        return {"notes": 0}
    pitches = [p for _, _, p, _ in notes]
    velocities = [v for _, _, _, v in notes]
    run = best = 1
    for a, b in zip(velocities, velocities[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    offsets = [round((s * 4 - round(s * 4)) / 4, 4) for s, _, _, _ in notes]  # from the 16th grid
    off_grid = [o for o in offsets if abs(o) > 0.01]
    last_bar = int(max(s for s, _, _, _ in notes) // beats_per_bar)
    density = [0] * (last_bar + 1)
    for s, _, _, _ in notes:
        density[int(s // beats_per_bar)] += 1
    overlaps = 0
    by_pitch: dict[int, list[tuple[float, float]]] = {}
    for s, d, p, _ in notes:
        by_pitch.setdefault(p, []).append((s, s + d))
    for spans in by_pitch.values():
        overlaps += sum(1 for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False) if s2 < e1 - 1e-6)
    stats = {
        "notes": len(notes),
        "range": [pictures.note_name(min(pitches)), pictures.note_name(max(pitches))],
        "velocity": {"min": min(velocities), "max": max(velocities),
                     "mean": round(mean(velocities)), "sd": round(pstdev(velocities), 1),
                     "longest_same_run": best},
        "off_16th_grid": {"count": len(off_grid),
                          "mean_beats": round(mean(off_grid), 3) if off_grid else 0.0},
        "notes_per_bar": density,
        "same_pitch_overlaps": overlaps,
        "mean_length_beats": round(mean(d for _, d, _, _ in notes), 3),
    }
    if scale is not None:
        outside = sorted({pictures.note_name(p) for p in pitches if p % 12 not in scale})
        stats["outside_key"] = {"count": sum(1 for p in pitches if p % 12 not in scale),
                                "pitches": outside[:12]}
    return stats


def _shared_register(parts: dict[str, list[tuple[float, float, int, int]]]) -> list[dict]:
    """Pairs of parts sounding within an octave of each other at the same time."""
    names = list(parts)
    clashes = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            beats = 0.0
            for s1, d1, p1, _ in parts[a]:
                for s2, d2, p2, _ in parts[b]:
                    if abs(p1 - p2) < 12:
                        beats += max(0.0, min(s1 + d1, s2 + d2) - max(s1, s2))
            if beats > 0:
                clashes.append({"parts": [a, b], "beats_within_octave": round(beats, 2)})
    return sorted(clashes, key=lambda c: -c["beats_within_octave"])[:10]


def note_view(
    *,
    tracks: list[int],
    key: str | None = None,
    picture: bool = False,
    beats_per_bar: int = 4,
) -> dict:
    """
    Numbers (and optionally a piano roll PNG) for the written notes on tracks:
    per part range, notes outside key (key like "F minor"), velocity spread and
    longest run of one velocity, notes off the 16th grid, notes per bar, same-pitch
    overlaps; across parts, pairs that sound within an octave of each other and for
    how many beats. picture: write a piano roll (one lane per part, out-of-key notes
    outlined red) and return its path; Read it to look.
    """
    project = get_project()
    scale = parse_key(key)[1] if key else None
    parts = {}
    for index in tracks:
        track = project.tracks[index]
        parts[f"{index}: {track.name or 'track'}"] = _notes_of(track)
    result: dict = {"success": True,
                    "parts": {name: note_stats(notes, scale=scale, beats_per_bar=beats_per_bar)
                              for name, notes in parts.items()}}
    if len(parts) > 1:
        result["shared_register"] = _shared_register(parts)
    if picture and any(parts.values()):
        result["picture"] = str(pictures.piano_roll(
            parts, pictures.default_path("pianoroll"), title=f"notes, {key or 'no key given'}",
            beats_per_bar=beats_per_bar, scale=scale))
    return result


TOOLS = (write_notes, set_fx, write_envelope, note_view)
