"""Writing music in beats, in bulk: notes, automation, plugin settings, and a note view.

Everything here is positioned in quarter-note beats from the project start
(beat 0 = bar 1), because music is written in beats and the tempo may change.
Each tool takes a whole part per call: a part as text costs a few tokens per
note, where one call per note costs a few hundred.
"""

from __future__ import annotations

import re
from pathlib import Path
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
    replace: bool = True,
) -> dict:
    """
    Write a part as one MIDI item: starts at start_beat (quarter notes from the
    project start; beat 0 = bar 1), lasts length_beats * repeat. replace: first
    delete the track's items that overlap that span, so writing a part again
    rewrites it instead of stacking on it.
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
    replaced = _delete_items(track, start_beat, start_beat + total) if replace else 0
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
        **({"replaced_items": replaced} if replaced else {}),
        "range": [pictures.note_name(min(pitches)), pictures.note_name(max(pitches))] if pitches else [],
    }


def _delete_items(track, start_beat: float, end_beat: float) -> int:
    """Delete the track's items overlapping [start_beat, end_beat); how many."""
    start = RPR.TimeMap2_QNToTime(0, start_beat)
    end = RPR.TimeMap2_QNToTime(0, end_beat)
    doomed = [item for item in track.items
              if item.position < end - 1e-6 and item.position + item.length > start + 1e-6]
    for item in doomed:
        item.delete()
    return len(doomed)


def clear_items(*, tracks: list[int] | None = None, start_beat: float = 0.0,
                end_beat: float | None = None, envelopes: bool = True) -> dict:
    """
    Delete the items on `tracks` (default: every track) that overlap
    start_beat..end_beat (default: to the end), to rebuild an arrangement.
    envelopes: also delete every automation point (volume, pan, FX parameters,
    ducking) in that span.
    """
    project = get_project()
    indices = tracks if tracks is not None else range(project.n_tracks)
    end = 1e9 if end_beat is None else end_beat
    deleted = {i: _delete_items(project.tracks[i], start_beat, end) for i in indices}
    result = {"success": True, "deleted": {i: n for i, n in deleted.items() if n},
              "total": sum(deleted.values())}
    if envelopes:
        start_s = RPR.TimeMap2_QNToTime(0, start_beat)
        end_s = 1e9 if end_beat is None else RPR.TimeMap2_QNToTime(0, end_beat)
        cleared = 0
        for i in indices:
            track_id = project.tracks[i].id
            for e in range(RPR.CountTrackEnvelopes(track_id)):
                envelope = RPR.GetTrackEnvelope(track_id, e)
                RPR.DeleteEnvelopePointRange(envelope, start_s - 1e-6, end_s)
                cleared += 1
        result["envelopes_cleared"] = cleared
    return result


def fx_index_of(names: list[str], fx: int | str) -> int:
    """An FX's chain index from its index or its (renamed) name: case-insensitive,
    the whole name first, else a unique substring ("channel", "duck")."""
    if isinstance(fx, int) or (isinstance(fx, str) and fx.lstrip("-").isdigit()):
        index = int(fx)
        if not 0 <= index < len(names):
            raise IndexError(f"no FX {index}; the track has {len(names)}")
        return index
    wanted = fx.lower()
    exact = [i for i, n in enumerate(names) if n.lower() == wanted]
    if exact:
        return exact[0]
    partial = [i for i, n in enumerate(names) if wanted in n.lower()]
    if len(partial) == 1:
        return partial[0]
    found = "several" if partial else "no"
    raise ValueError(f"{found} FX named like {fx!r}: {sorted(set(names))[:12]}")


def _fx_names(track) -> list[str]:
    return [str(RPR.TrackFX_GetFXName(track.id, i, "", 256)[3]) for i in range(track.n_fxs)]


def _param_names(track_id, fx_index: int) -> list[str]:
    count = RPR.TrackFX_GetNumParams(track_id, fx_index)
    return [str(RPR.TrackFX_GetParamName(track_id, fx_index, i, "", 256)[4]) for i in range(count)]


def _resolve_param(names: list[str], fx_index: int, param: str | int) -> int:
    """A parameter's index from its index or its name. Names match case-insensitively,
    preferring in turn: the whole name, the name without a trailing "(unit)",
    a unique prefix, a unique substring."""
    if isinstance(param, int) or (isinstance(param, str) and param.isdigit()):
        index = int(param)
        if not 0 <= index < len(names):
            raise IndexError(f"FX {fx_index} has no parameter {index} (it has {len(names)})")
        return index
    wanted = param.strip().lower()
    lowered = [n.lower() for n in names]
    bare = [re.sub(r"\s*\([^)]*\)\s*$", "", n) for n in lowered]
    for candidates in ([i for i, n in enumerate(lowered) if n == wanted],
                       [i for i, n in enumerate(bare) if n == wanted]):
        if candidates:
            return candidates[0]
    for candidates in ([i for i, n in enumerate(lowered) if n.startswith(wanted)],
                       [i for i, n in enumerate(lowered) if wanted in n]):
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise ValueError(f"parameter {param!r} matches several: {[names[i] for i in candidates][:8]}")
    raise ValueError(f"FX {fx_index} has no parameter like {param!r}; some: {names[:12]}")


NUMBER_SHOWN = re.compile(r"([-+]?)(inf|\d*\.?\d+)", re.IGNORECASE)


def shown_number(text: str) -> float | None:
    """The first number in a parameter's display text ("-12.0 dB" -> -12.0,
    "-inf" -> -1e9), or None."""
    match = NUMBER_SHOWN.search(text)
    if not match:
        return None
    sign = -1.0 if match.group(1) == "-" else 1.0
    return sign * (1e9 if match.group(2).lower() == "inf" else float(match.group(2)))


def normalized_for(shown_at, target: float, *, steps: int = 24) -> float:
    """The normalized value (0-1) whose display shows `target`, by bisection.
    shown_at(v) sets the parameter to v and returns the number displayed;
    the display must rise or fall steadily with v."""
    low, high = 0.0, 1.0
    rising = shown_at(1.0) >= shown_at(0.0)
    for _ in range(steps):
        middle = (low + high) / 2
        if (shown_at(middle) < target) == rising:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _formatted(track_id, fx_index: int, index: int) -> str:
    formatted = RPR.TrackFX_GetFormattedParamValue(track_id, fx_index, index, "", 256)
    return str(formatted[4]) if isinstance(formatted, (list, tuple)) else str(formatted)


def _set_shown(track_id, fx_index: int, index: int, *, text: str) -> None:
    """Set a parameter to the value it displays as `text` ("-1", "-1 dB")."""
    target = shown_number(text)
    if target is None:
        raise ValueError(f"no number in {text!r}")

    def shown_at(value: float) -> float:
        RPR.TrackFX_SetParamNormalized(track_id, fx_index, index, value)
        number = shown_number(_formatted(track_id, fx_index, index))
        if number is None:
            raise ValueError(f"parameter {index} does not display a number")
        return number

    RPR.TrackFX_SetParamNormalized(track_id, fx_index, index, normalized_for(shown_at, target))


def set_fx(
    *,
    track_index: int,
    fx_index: int | str,
    params: dict[str, float | str] | None = None,
    config: dict[str, str] | None = None,
) -> dict:
    """
    Set several parameters of one FX at once. params: {name or index: value};
    names match case-insensitively, whole or as a unique substring. A number is
    in the plugin's own units (a JSFX slider's value, a VST's 0-1); a string is
    what the plugin displays ("-1" or "-1 dB" for a ceiling, "250 Hz"), found by search.
    config: named settings a parameter cannot reach, e.g. ReaSamplOmatic5000's
    {"FILE0": "/path/sample.wav", "DONE": ""} to load a sample.
    track_index -1 is the master track. fx_index: the FX's index or name ("channel").
    Returns each parameter's value as the plugin displays it.
    """
    project = get_project()
    track = project.master_track if track_index == -1 else project.tracks[track_index]
    try:
        fx_index = fx_index_of(_fx_names(track), fx_index)
    except (IndexError, ValueError) as e:
        return {"success": False, "error": f"track {track_index}: {e}"}
    shown = {}
    for name, value in (config or {}).items():
        if not RPR.TrackFX_SetNamedConfigParm(track.id, fx_index, name, str(value)):
            return {"success": False, "error": f"FX {fx_index} refused config {name!r}"}
    names = _param_names(track.id, fx_index) if params else []
    resolved, problems = {}, []
    for name in params or {}:
        try:
            resolved[name] = _resolve_param(names, fx_index, name)
        except (IndexError, ValueError) as e:
            problems.append(str(e))
    if problems:  # set nothing rather than half of what was asked
        return {"success": False, "error": "; ".join(problems)}
    for name, value in (params or {}).items():
        index = resolved[name]
        if isinstance(value, str):
            try:
                _set_shown(track.id, fx_index, index, text=value)
            except ValueError as e:
                return {"success": False, "error": f"{names[index]}: {e}", "set_so_far": shown}
        else:
            RPR.TrackFX_SetParam(track.id, fx_index, index, float(value))
        shown[names[index]] = _formatted(track.id, fx_index, index)
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
    fx_index: int | str | None = None,
    param: str | None = None,
    clear: bool = True,
) -> dict:
    """
    Write automation as "BEAT:VALUE[:SHAPE]" points (beats from the project start),
    separated by spaces. target: volume (VALUE in dB), pan (-1 left .. 1 right), or fx
    (fx_index or FX name + param name/index; VALUE in the plugin's units).
    SHAPE: linear (default), square, slow, fast_start, fast_end, bezier.
    clear: first delete existing points between the first and last beat written.
    """
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
    track = get_project().tracks[track_index]
    if target == "fx" and fx_index is not None:
        try:
            fx_index = fx_index_of(_fx_names(track), fx_index)
        except (IndexError, ValueError) as e:
            return {"success": False, "error": str(e)}
    _write_points(track, target, parsed, clear=clear, fx_index=fx_index, param=param)
    return {"success": True, "track_index": track_index, "target": target,
            "points": len(parsed), "beats": [parsed[0][0], parsed[-1][0]]}


def _write_points(track, target: str, points: list[tuple[float, float, int]], *, clear: bool,
                  fx_index: int | None = None, param: str | int | None = None) -> None:
    """Write (beat, value, shape code) points; volume values in dB, pan -1..1."""
    envelope = _envelope(track, target, fx_index=fx_index, param=param)
    if is_null(envelope) and target in ("volume", "pan"):
        # Show the track's envelope so it exists, then fetch it again.
        RPR.SetOnlyTrackSelected(track.id)
        RPR.Main_OnCommand(40406 if target == "volume" else 40407, 0)
        envelope = _envelope(track, target, fx_index=fx_index, param=param)
    if is_null(envelope):
        raise ValueError(f"no {target} envelope on track {track.index}")
    times = [RPR.TimeMap2_QNToTime(0, beat) for beat, _, _ in points]
    if clear:
        RPR.DeleteEnvelopePointRange(envelope, min(times) - 1e-6, max(times) + 1e-6)
    mode = RPR.GetEnvelopeScalingMode(envelope)
    for time, (_, value, shape) in zip(times, points, strict=True):
        if target == "volume":
            value = RPR.ScaleToEnvelopeMode(mode, 10 ** (value / 20))
        elif target == "pan":
            value = -value  # REAPER's pan envelope runs opposite to track pan
        RPR.InsertEnvelopePoint(envelope, time, value, shape, 0.0, False, True)
    RPR.Envelope_SortPoints(envelope)


def duck_points(hits: list[float], *, depth_db: float, release_beats: float,
                gap: float = 0.01) -> list[tuple[float, float, int]]:
    """Volume points (beat, dB, shape) that dip to depth_db at each hit and come
    back to 0 dB over release_beats, cut short by the next hit."""
    points: list[tuple[float, float, int]] = []
    hits = sorted(set(hits))
    for i, hit in enumerate(hits):
        following = hits[i + 1] if i + 1 < len(hits) else None
        recovered = not points or (points[-1][1] == 0.0 and points[-1][0] < hit - gap)
        if hit >= gap and recovered:
            points.append((hit - gap, 0.0, ENVELOPE_SHAPES["linear"]))
        points.append((hit, depth_db, ENVELOPE_SHAPES["slow"]))
        back = hit + release_beats
        if following is not None and back > following - 2 * gap:
            back = following - 2 * gap
            if back <= hit:
                continue
            points.append((back, depth_db * (1 - (back - hit) / release_beats), ENVELOPE_SHAPES["linear"]))
        else:
            points.append((back, 0.0, ENVELOPE_SHAPES["linear"]))
    return points


DUCK_FX = "JS: utility/volume_pan_sample_accurate_auto"
DUCK_NAME = "duck"


def _duck_fx(track) -> int:
    """Index of the track's gain plugin renamed "duck", appended if missing.

    Ducking lives on its own plugin because a track Volume envelope takes the
    fader over: with one in place, set_track_volume no longer changes anything.
    """
    for index in range(RPR.TrackFX_GetCount(track.id)):
        if RPR.TrackFX_GetNamedConfigParm(track.id, index, "renamed_name", "", 256)[4] == DUCK_NAME:
            return index
    index = RPR.TrackFX_AddByName(track.id, DUCK_FX, False, 1)  # 1: always add, at the end
    if index < 0:
        raise ValueError(f"could not add {DUCK_FX}")
    RPR.TrackFX_SetNamedConfigParm(track.id, index, "renamed_name", DUCK_NAME)
    return index


def duck(
    *,
    tracks: list[int],
    trigger: int,
    depth_db: float = -9.0,
    release_beats: float = 0.5,
    start_beat: float | None = None,
    end_beat: float | None = None,
) -> dict:
    """
    Sidechain-style ducking without routing: read the note starts on the trigger
    track (a kick) and, on each of tracks, automate a gain plugin named "duck"
    (added at the end of the chain if missing) to dip to depth_db at every hit
    and recover over release_beats. The fader stays free. Limit to
    [start_beat, end_beat) if given; replaces earlier duck points in that span.
    """
    project = get_project()
    hits = [beat for beat, _, _, _ in _notes_of(project.tracks[trigger])
            if (start_beat is None or beat >= start_beat) and (end_beat is None or beat < end_beat)]
    if not hits:
        return {"success": False, "error": f"no notes on track {trigger} in that span"}
    points = duck_points(hits, depth_db=depth_db, release_beats=release_beats)
    for index in tracks:
        track = project.tracks[index]
        _write_points(track, "fx", points, clear=True, fx_index=_duck_fx(track), param=0)
    return {"success": True, "tracks": tracks, "hits": len(set(hits)),
            "beats": [min(hits), max(hits)], "points_per_track": len(points)}


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



def place_audio(*, track_index: int, items: list[dict]) -> dict:
    """
    Put audio files on a track as items, positioned in beats (beat 0 = bar 1):
    field recordings, risers, vocal phrases, one-shots. Each item:
    {"file": path, "beat": start, "beats": length (default: the whole file at
    its rate), "offset_s": seconds skipped into the file, "gain_db": 0,
    "pitch": semitones (formant-free shift, tempo kept), "rate": playback
    rate (pitch preserved), "resample": true makes "pitch" play faster or
    slower like a sampler (clean transients, length changes), "pan": -1..1,
    "fade_in"/"fade_out": beats, "loop": repeat the file to fill "beats"}.
    Clear the span first with clear_items to rebuild.
    """
    track = get_project().tracks[track_index]
    placed, missing = 0, []
    for spec in items:
        path = Path(spec["file"]).expanduser()
        if not path.is_file():
            missing.append(str(path))
            continue
        source = RPR.PCM_Source_CreateFromFile(str(path))
        resample = bool(spec.get("resample"))
        pitch = float(spec.get("pitch", 0.0))
        rate = float(spec.get("rate", 1.0)) * (2 ** (pitch / 12) if resample else 1.0)
        source_s = RPR.GetMediaSourceLength(source, False)
        source_s = source_s[0] if isinstance(source_s, (list, tuple)) else source_s
        start = RPR.TimeMap2_QNToTime(0, spec["beat"])
        if "beats" in spec:
            length = RPR.TimeMap2_QNToTime(0, spec["beat"] + spec["beats"]) - start
        else:
            length = (source_s - spec.get("offset_s", 0.0)) / rate
        item = RPR.AddMediaItemToTrack(track.id)
        take = RPR.AddTakeToMediaItem(item)
        RPR.SetMediaItemTake_Source(take, source)
        beat_s = 60 / RPR.Master_GetTempo()
        for param, value in (("D_POSITION", start), ("D_LENGTH", length),
                             ("D_FADEINLEN", spec.get("fade_in", 0.0) * beat_s),
                             ("D_FADEOUTLEN", spec.get("fade_out", 0.0) * beat_s),
                             ("B_LOOPSRC", 1.0 if spec.get("loop") else 0.0)):
            RPR.SetMediaItemInfo_Value(item, param, value)
        for param, value in (("D_STARTOFFS", spec.get("offset_s", 0.0)),
                             ("D_VOL", 10 ** (spec.get("gain_db", 0.0) / 20)),
                             ("D_PITCH", 0.0 if resample else pitch), ("D_PLAYRATE", rate),
                             ("B_PPITCH", 0.0 if resample else 1.0),
                             ("D_PAN", float(spec.get("pan", 0.0)))):
            RPR.SetMediaItemTakeInfo_Value(take, param, value)
        placed += 1
    RPR.UpdateArrange()
    result = {"success": not missing, "track_index": track_index, "placed": placed}
    if missing:
        result["missing"] = missing[:5]
    return result


TOOLS = (write_notes, clear_items, place_audio, set_fx, write_envelope, duck, note_view)
